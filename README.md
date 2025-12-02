# Неформал-классификатор (TF-IDF + LinearSVC)


Проект классифицирует текстовые описания людей на 4 класса: `emo`, `punk`, `goth`, `normal`.


## Файлы

- dataset_10k.csv — сгенерированный датасет (10k записей)
- train.py — обучение TF-IDF + LinearSVC (с калибровкой для вероятностей)
- app.py — Gradio интерфейс с подсветкой ключевых слов, аватарками и историей запросов
- assets/ — аватарки для классов
- models/ — здесь сохраняется модель после тренировки
- Dockerfile — для контейнеризации

## Быстрый старт

1. Установить зависимости: `pip install -r requirements.txt`
2. Обучить модель: `python train.py` (создаст `models/neformal_svm_pipeline.joblib`)
3. Запустить приложение: `python app.py`
4. Откроется Gradio на локальном порту 7860

## Docker

Сборка: `docker build -t neformal-app .`
Запуск: `docker run -p 7860:7860 neformal-app`

