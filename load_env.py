from sentence_transformers import SentenceTransformer
import pickle
import pandas as pd
import faiss


def load_all_classes():
    device = 'cpu'

    # грузим берт енкодер
    model = SentenceTransformer("models/content/multilingual-e5-base")
    model = model.to(device)

    # грузим типы поедов
    with open('faiss_pipeline/train_type_to_boarder.pickle', 'rb') as f:
        train_type_to_boarder = pickle.load(f)

    df_qa = pd.read_csv('faiss_pipeline/query_to_answer_last_version.csv')
    df_qa['query'] = df_qa['query'].str.lower()

    # грузим tf-idf енкодер
    with open('faiss_pipeline/tfidf_word_vectorizer.pickle', 'rb') as f:
        tfidf_vectorizer = pickle.load(f)

    # грузим tf-idf индекс
    tfidf_index_cos = faiss.read_index("faiss_pipeline/tfidf_index_cos.index")

    # грузим берт индекс
    bert_index_cos = faiss.read_index("faiss_pipeline/bert_index_cos.index")

    return df_qa, train_type_to_boarder, model, tfidf_vectorizer, tfidf_index_cos, bert_index_cos