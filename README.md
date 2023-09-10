# [Голосовой Помощник Марфуша](https://t.me/machinist_helper_bot)

<p align="center">
<img src="https://github.com/inspired99/rzhd-gpt/assets/64794482/f246a14f-11e8-4a1b-8e63-0a2be98ae7f3" width="225">
</p>

## Описание

Перед вами голосовой помощник для водителей поездов - Марфуша. Марфуша способна отвечать на вопросы машиниста и выдавать инструкции в экстренных случаях при возникновении неисправностей или поломок.

Чтобы воспользоваться помощью Марфуши - нужно всего лишь начать с [ней](https://t.me/machinist_helper_bot) общение.
Для получения голосового ответа к Марфуше задавать вопрос можно голосом или в виде текста. Также Марфуша продублирует свой ответ в виде текстового обращения.

## Структура

`
faiss_pipeline
` - индексы FAISS для поиска в векторном пространстве с использованием эмбеддингов TF-IDF и трансформера [e5-base](https://huggingface.co/intfloat/multilingual-e5-base)

## Использование

К данному репозиторию прилагется дополнительный файл, который необходимо скачать для языковой модели e5-base - [ссылка](https://drive.google.com/file/d/1lRJTbZRJ-ZrRZfaKAeQtDWq_i_IlJv9w/view?usp=sharing). Данный файл необходимо расположить в директории по пути `./models/multilingual-e5-base`
