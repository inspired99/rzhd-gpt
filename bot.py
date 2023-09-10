import logging
import telebot
from logic import *


TELEGRAM_TOKEN = os.getenv("TELEGRAM_TOKEN")
bot = telebot.TeleBot(TELEGRAM_TOKEN)
MODELS_MARKUP = telebot.types.ReplyKeyboardMarkup(resize_keyboard=True, one_time_keyboard=True)
CURRENT_MODEL = None

logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO,
    filename="bot.log",
)


@bot.message_handler(commands=["start"])
def bot_start(message):
    bot_help(message)

    for model in AVALIABLE_TRAIN_MODELS:
        MODELS_MARKUP.row(model)

    reply = bot.send_message(
        message.chat.id,
        "Введите вашу модель поезда.", reply_markup=MODELS_MARKUP
    )

    bot.register_next_step_handler(reply, get_train_model)


def get_train_model(reply):
    global CURRENT_MODEL

    if reply and reply.text in AVALIABLE_TRAIN_MODELS:
        bot.send_message(
            reply.chat.id,
            "Такая модель поддерживается, всё хорошо! Можете задавать вопросы."
        )
        CURRENT_MODEL = reply.text
    else:
        model = bot.send_message(
            reply.chat.id,
            "Извините, эта модель недоступна. Пожалуйста, выберите из списка: " +
            ", ".join(AVALIABLE_TRAIN_MODELS)
        )
        bot.register_next_step_handler(model, get_train_model)


@bot.message_handler(commands=["help"])
def bot_help(message):
    bot.send_message(
        message.chat.id,
        "Привет! Меня зовут Марфуша, я - голосовой помощник машиниста. "
        "Вы можете задавать мне вопросы, а я постараюсь помочь!"
    )


@bot.message_handler(content_types=['text'])
def bot_text(message):
    global CURRENT_MODEL

    if CURRENT_MODEL is None:
        bot_start(message)

    else:
        bot.send_message(message.chat.id, "Текст получен, ожидайте ответ...")
        user_input_text = message.text

        voice, caption = text_input_processing(user_input_text, CURRENT_MODEL)

        bot.send_voice(
                message.chat.id,
                voice,
                caption=caption
            )


@bot.message_handler(content_types=[
    'voice',
    'audio',
    'document'
])
def voice_message_handler(message):
    global CURRENT_MODEL

    if CURRENT_MODEL is None:
        bot_start(message)
    else:

        if message.content_type == 'voice':
            file_info = bot.get_file(message.voice.file_id)
        elif message.content_type == 'audio':
            file_info = bot.get_file(message.audio.file_id)
        elif message.content_type == 'document':
            file_info = bot.get_file(message.document.file_id)
        else:
            bot.send_message(message.chat.id, "Формат документа не поддерживается")
            return

        file_path = file_info.file_path
        file_on_disk = Path("", f"{file_info.file_id}.tmp")
        file_data = bot.download_file(file_path)
        with open(file_on_disk, 'wb') as f:
            f.write(file_data)
        bot.send_message(message.chat.id, "Аудио получено, ожидайте ответ...")

        voice, caption = audio_input_processing(file_on_disk, CURRENT_MODEL)
        if voice is None:
            bot.send_message(message.chat.id, caption)
        else:
            bot.send_voice(
                    message.chat.id,
                    voice,
                    caption=caption
                )


if __name__ == "__main__":
    print("Запуск бота")
    bot.polling(none_stop=True)
