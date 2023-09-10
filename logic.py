import os
import re
import numpy as np
from pathlib import Path
from dotenv import load_dotenv
from load_env import load_all_classes
from faiss_search_kaggle import filter_indexes_by_train_type, retrieve_candidate
from transformers import pipeline
import librosa
import torch
import torchaudio
from omegaconf import OmegaConf
from num2words import num2words


load_dotenv()
df_qa, train_type_to_boarder, model, tfidf_vectorizer, tfidf_index_cos, bert_index_cos = load_all_classes()


SOUND_CONVERTER = {"Б": "бэ", "В": "вэ", "Г": "гэ", "Д": "дэ",
                   "Ж": "жэ", "З": "зэ", 'К': "ка", 'Л': "эл",
                   "М": "эм", "Н": "эн", 'П': "пэ", 'Р': "эр",
                   "С": "эс", "Т": "тэ", 'Ф': "эф", 'Х': "ха",
                   "Ц": "цэ", "Ч": "че", "Ш": "ша", 'Щ': "ща"}

MAPPER = {'2ЭС5К': [4317, 4783], 'ВЛ10К': [8150, 8227], 'ЭП1': [16005, 16194], 'ВЛ11М': [8228, 8730],
          '2ТЭ25А': [1013, 2021], 'ЭП1М': [16005, 16194], '2ТЭ10М': [364, 558], '2М62У': [0, 363],
          '2ТЭ10УК': [364, 558], '3ЭС5К': [4317, 4783], 'ЧС8': [15793, 16004], '2ЭС10': [3762, 4055],
          'ЧМЭ3': [12888, 13801], 'ВЛ85': [10351, 10845], '2ЭС4К': [4056, 4316], 'ТЭМ14': [10846, 11136],
          'ЧС6': [14983, 15371], 'ТЭМ18ДМ': [11137, 11659], 'ВЛ10': [7890, 8149], 'ЧС2': [13802, 13958],
          'ЧС200': [14983, 15371], '2ТЭ10У': [364, 558], 'ВЛ80Р': [9199, 9724], '2ЭС6': [4784, 5583],
          '2ТЭ25КМ': [2022, 3664], 'ВЛ80С': [9725, 10087], 'ТЭМ2': [11660, 11954], 'ЧС2К': [13959, 14103],
          '2ЭС7': [5584, 7889], '2М62': [0, 363], 'ЭП2К': [17994, 18857], '2ТЭ70': [3665, 3761],
          'ТЭМ18Д': [11137, 11659], 'ТЭМ7А': [11955, 12251], 'ВЛ80Т': [10088, 10350], 'ЧС2Т': [14104, 14480],
          'ВЛ10У': [7890, 8149], 'ВЛ15': [8731, 8951], 'ВЛ11': [8228, 8432], 'ЧС7': [15372, 15792],
          '2ТЭ116УД': [947, 1012], 'ЧС4Т': [14481, 14982], 'ЭП10': [16195, 16358], 'ТЭП70': [12252, 12466],
          'ЭП20': [16359, 17993], 'ТЭП70БС': [12467, 12887], 'ВЛ65': [8952, 9198], '2ТЭ116': [559, 946],
          '2ТЭ10МК': [364, 558]}

AVALIABLE_TRAIN_MODELS = MAPPER.keys()


torch.hub.download_url_to_file('https://raw.githubusercontent.com/snakers4/silero-models/master/models.yml',
                               'latest_silero_models.yml',
                               progress=False)
models = OmegaConf.load('latest_silero_models.yml')

language_tts = 'ru'
model_id_tts = 'v4_ru'
device_tts = torch.device('cpu')

model_tts, example_text = torch.hub.load(repo_or_dir='snakers4/silero-models',
                                     model='silero_tts',
                                     language=language_tts,
                                     speaker=model_id_tts)
model_tts.to(device_tts)  # gpu or cpu
sample_rate_tts = 24000
speaker_tts = 'baya'
put_accent_tts=True
put_yo_tts=True


model_whisper='openai/whisper-base'
tokenizer_whisper='openai/whisper-base'
pipe = pipeline('automatic-speech-recognition',model=model_whisper, tokenizer=tokenizer_whisper)


def whisper_inference(wav_path):
    wav, sr = librosa.load(wav_path, sr=16000, mono=True)
    result = pipe(wav, 
                chunk_length_s=30, 
                generate_kwargs={"language": "<|ru|>", "task": "transcribe"})['text']
    return result


def replace_english_with_russian(text): 
    english_chars = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz" 
    russian_chars = "АБЦДЕФГХИЖКЛМНОПКРСТУВВСЙЗабцдефгхижклмнопкрстуввсйз" 
    translation_table = str.maketrans(english_chars, russian_chars) 
    translated_text = text.translate(translation_table) 
    return translated_text 
 
 
def n2v(word): 
  if word.isdigit(): 
    return num2words(word, lang='ru') 
  else: 
    return word 
 
 
def replace_number_with_words(text): 
    pattern = r'(\D|\b)(\d+)(\D|\b)' 
    replaced_text = re.sub(pattern, r'\1 \2 \3', text) 
    replaced_text = re.sub(pattern, r'\1 \2 \3', replaced_text) 
    replaced_text = ' '.join([n2v(token) for token in replaced_text.split()]) 
    replaced_text = re.sub(r'\s+', r' ', replaced_text) 
    return replaced_text


def preprocess(text): 
  return replace_number_with_words(replace_english_with_russian(text))



def audio_input_processing(file_on_disk, train_model):
    user_input_text = whisper_inference(file_on_disk)
    if not user_input_text:
        return None, "Формат документа не поддерживается"
    else:
        reference = f"\n\nСогласно нормативным актам для поездов типа {train_model}."
        text_answer = get_reliable_answer_by_query(user_input_text,
                                                   train_model)
        preprocessed_text_answer = preprocess(text_answer)

        out_file = model_tts.apply_tts(text=preprocessed_text_answer,
                        speaker=speaker_tts,
                        sample_rate=sample_rate_tts,
                        put_accent=put_accent_tts,
                        put_yo=put_yo_tts)
        caption = text_answer.strip() + reference
        torchaudio.save('generated_audio.wav', out_file.reshape(1, -1), 24000)
        voice = open('generated_audio.wav', 'rb').read()

        os.remove('generated_audio.wav')
        os.remove(file_on_disk)

        return voice, caption


def text_input_processing(user_input_text, train_model):
    answer = get_reliable_answer_by_query(user_input_text, train_model)
    preprocessed_answer = preprocess(answer)

    out_filename = model_tts.apply_tts(text=preprocessed_answer,
                    speaker=speaker_tts,
                    sample_rate=sample_rate_tts,
                    put_accent=put_accent_tts,
                    put_yo=put_yo_tts)

    
    reference = f"\n\nСогласно нормативным актам для поездов типа {train_model}."

    caption = answer.strip() + reference
    torchaudio.save('generated_audio.wav', out_filename.reshape(1, -1), 24000)

    with open('generated_audio.wav', 'rb') as f:
        voice = f.read()

    os.remove('generated_audio.wav')

    try:
        os.remove('*.tmp')
    except FileNotFoundError:
        pass

    return voice, caption


def get_reliable_answer_by_query(query, train_type):
    tfidf_index_sub, bert_index_sub = filter_indexes_by_train_type(train_type, train_type_to_boarder,
                                                                   tfidf_index_cos, bert_index_cos)

    start_ind = train_type_to_boarder[train_type][0]

    candidates_ind = retrieve_candidate(query, model, tfidf_vectorizer, tfidf_index_sub,
                                        bert_index_sub, mode='cos', retrieve_topn=20, final_topn=1)

    candidates_df = df_qa.loc[np.array(candidates_ind) + start_ind, ['query', 'answer']]

    return candidates_df.iloc[0, 1]

