import gradio as gr
import joblib
import numpy as np
import html
import random

# -----------------------------------------
#  Загрузка обученной модели
# -----------------------------------------
MODEL_PATH = "models/neformal_svm_pipeline.joblib"
pipeline = joblib.load(MODEL_PATH)
tfidf = pipeline.named_steps["tfidf"]

CLASSES = pipeline.classes_
LABELS = {
    "emo": "Эмо",
    "punk": "Панк",
    "goth": "Гот",
    "normal": "Обычный",
}

# Аватарки
avatars = {
    "emo": "assets/emo.png",
    "punk": "assets/punk.png",
    "goth": "assets/goth.png",
    "normal": "assets/normal.png",
}

# -----------------------------------------
#  Кнопка "Случайное описание"
# -----------------------------------------
RANDOM_DESCRIPTIONS = {
    "emo": [
        "Чёрная челка закрывает глаза, слушает My Chemical Romance.",
        "Носит розово-чёрные браслеты и пишет стихи.",
        "Тёмная одежда, эмоциональность, грустные цитаты."
    ],
    "goth": [
        "Полностью чёрная одежда и серебряные кольца.",
        "Любит мистику, готик-рок и ночные прогулки.",
        "Бледная кожа, тёмный макияж, бархат."
    ],
    "punk": [
        "Ирокез, булавки, рваная одежда.",
        "Кожаная куртка с нашивками, яркие волосы.",
        "Панк-рок, протесты, цепи и металл."
    ],
    "normal": [
        "Обычная одежда: худи, кроссовки, джинсы.",
        "Спокойный стиль, слушает популярную музыку.",
        "Учёба, работа, сериалы — обычный образ жизни."
    ]
}


def random_description():
    group = random.choice(list(RANDOM_DESCRIPTIONS.keys()))
    return random.choice(RANDOM_DESCRIPTIONS[group])


# -----------------------------------------
#  Белая мягкая тема оформления
# -----------------------------------------
CSS = """
<style>
body, .gradio-container {
    background: #f7f7f9 !important;
    color: #222 !important;
    font-family: 'Inter', sans-serif;
}

/* Поля ввода */
textarea, input {
    background: #ffffff !important;
    border: 1px solid #d2d6dd !important;
    color: #222 !important;
    font-size: 16px !important;
    padding: 10px !important;
    border-radius: 8px !important;
}

/* Кнопки */
.gr-button {
    background: #4a6cf7 !important;
    color: white !important;
    border-radius: 10px !important;
    font-size: 17px !important;
    border: none !important;
}
.gr-button:hover {
    background: #6a86ff !important;
    transition: 0.2s;
}

/* Карточки */
.soft-box {
    background: #ffffff;
    border-radius: 14px;
    padding: 14px;
    border: 1px solid #e5e7ef;
    box-shadow: 0 3px 12px rgba(0,0,0,0.04);
}

/* Подсветка */
mark {
    background: #ffd54a;
    padding: 2px 4px;
    border-radius: 4px;
}

/* Прогресс-бары */
.bar-bg {
    background: #e2e4ec;
    height: 12px;
    border-radius: 6px;
}
.bar-fill {
    background: #4a6cf7;
    height: 12px;
    border-radius: 6px;
}
</style>
"""

# -----------------------------------------
#  HTML с вероятностями классов
# -----------------------------------------
def build_prob_html(prob_dict):
    out = "<div class='soft-box'>"
    for cls, val in prob_dict.items():
        pct = int(val * 100)
        out += f"""
        <div style='margin-bottom:10px'>
            <b>{cls}</b>: {pct}% 
            <div class="bar-bg">
                <div class="bar-fill" style="width:{pct}%"></div>
            </div>
        </div>
        """
    out += "</div>"
    return out


# -----------------------------------------
#  Предсказание модели
# -----------------------------------------
history_data = []


def classify(text):
    if not text.strip():
        return "Введите описание", "", "", None, history_data

    pred = pipeline.predict([text])[0]
    proba = pipeline.predict_proba([text])[0]

    probs = {LABELS[c]: float(p) for c, p in zip(CLASSES, proba)}

    # Подсветка слов
    vec = tfidf.transform([text])
    arr = vec.toarray()[0]
    fn = tfidf.get_feature_names_out()

    safe = html.escape(text)
    important = np.argsort(arr)[-5:][::-1]

    for idx in important:
        tok = fn[idx]
        if tok in text.lower():
            safe = safe.replace(tok, f"<mark>{tok}</mark>")

    # Добавление в историю
    history_data.append([text, LABELS[pred]])

    return LABELS[pred], build_prob_html(probs), safe, avatars[pred], history_data


# -----------------------------------------
#  Интерфейс Gradio
# -----------------------------------------
with gr.Blocks(title="Классификатор неформальных субкультур") as demo:

    gr.HTML(CSS)
    gr.Markdown("## 🎭 Классификатор неформальных субкультур")

    with gr.Row():
        # Левая колонка
        with gr.Column(scale=2):
            desc = gr.Textbox(
                label="Описание человека",
                lines=4,
                placeholder="Напишите описание..."
            )

            gr.Markdown("### Примеры:")

            with gr.Row():
                gr.Button("Носит чёрную одежду...").click(
                    lambda: "Носит чёрную одежду, слушает тяжёлую музыку.",
                    None, desc
                )
                gr.Button("Ирокез, заклёпки...").click(
                    lambda: "Ирокез, заклёпки, слушает панк.",
                    None, desc
                )

            with gr.Row():
                gr.Button("Любит мистику...").click(
                    lambda: "Любит мистику, бархатную одежду.",
                    None, desc
                )
                gr.Button("Простой стиль").click(
                    lambda: "Простая одежда, учёба, спокойный образ.",
                    None, desc
                )

            gr.Button("🎲 Случайное описание").click(
                lambda: random_description(), None, desc
            )

            gr.Button("🧹 Очистить текст").click(lambda: "", None, desc)

            btn = gr.Button("Определить")

            result = gr.Textbox(label="Результат", interactive=False)
            result_probs = gr.HTML()
            highlight = gr.HTML(label="Подсветка важных слов")

        # Правая колонка
        with gr.Column(scale=1):
            avatar = gr.Image(label="Аватар", type="filepath", height=250)

            history_table = gr.Dataframe(
                headers=["Текст", "Класс"],
                label="История запросов",
                interactive=False,
                row_count=5,
            )

            gr.Button("Очистить историю").click(
                lambda: history_data.clear() or [],
                None, history_table
            )

    btn.click(
        classify,
        inputs=desc,
        outputs=[result, result_probs, highlight, avatar, history_table]
    )

demo.launch(server_name="127.0.0.1", server_port=7861)
