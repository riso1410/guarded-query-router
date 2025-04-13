import os

import dspy
import fasttext
import gradio as gr
from dotenv import load_dotenv
from dspy import LM

load_dotenv()

# Healthcare Chatbot with Model Selection

# Define your answer function with an extra model_choice parameter
def classify_prompt(prompt: str, history: list, model_choice: str) -> list:
    # Using fastText for classification
    if model_choice == "fastText":
        fasttext_model = fasttext.load_model('models/fastText_law_fasttext.bin')
        prediction = fasttext_model.predict(prompt)
        print("fastText prediction:", prediction)
        if prediction[0][0] in ["__label__0", 0, False]:
            answer = "Not relevant to the domain, try another question."
        else:
            llm = LM(
                api_key=os.getenv("OPENAI_API_KEY"),
                model="gpt-4o-mini",
                api_base=os.getenv("PROXY_URL"),
                temperature=0.5,
            )
            qa_bot = dspy.Predict("question -> answer: str")
            answer = qa_bot(question=prompt).answer

    elif model_choice == "SVM":
        # Load SVM model (assuming it's saved as a pickle file)
        import pickle
        svm_model = pickle.load(open('models/svm_model.pkl', 'rb'))
        prediction = svm_model.predict([prompt])[0]
        if prediction in [0, False]:
            answer = "Not relevant to the domain, try another question."
        else:
            llm = LM(
                api_key=os.getenv("OPENAI_API_KEY"),
                model="gpt-4o-mini",
                api_base=os.getenv("PROXY_URL"),
                temperature=0.5,
            )
            qa_bot = dspy.Predict("question -> answer: str")
            answer = qa_bot(question=prompt).answer

    elif model_choice == "XGBoost":
        # Load XGBoost model
        import xgboost as xgb
        xgb_model = xgb.Booster()
        xgb_model.load_model('models/xgboost_model.json')
        prediction = xgb_model.predict(xgb.DMatrix([prompt]))[0]
        if prediction < 0.5:  # Assuming binary classification threshold
            answer = "Not relevant to the domain, try another question."
        else:
            llm = LM(
                api_key=os.getenv("OPENAI_API_KEY"),
                model="gpt-4o-mini",
                api_base=os.getenv("PROXY_URL"),
                temperature=0.5,
            )
            qa_bot = dspy.Predict("question -> answer: str")
            answer = qa_bot(question=prompt).answer

    elif model_choice == "ModernBERT":
        # Load BERT model
        from transformers import AutoModelForSequenceClassification, AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained('models/bert_model')
        model = AutoModelForSequenceClassification.from_pretrained('models/bert_model')
        inputs = tokenizer(prompt, return_tensors="pt")
        prediction = model(**inputs).logits.argmax().item()
        if prediction == 0:
            answer = "Not relevant to the domain, try another question."
        else:
            llm = LM(
                api_key=os.getenv("OPENAI_API_KEY"),
                model="gpt-4o-mini",
                api_base=os.getenv("PROXY_URL"),
                temperature=0.5,
            )
            qa_bot = dspy.Predict("question -> answer: str")
            answer = qa_bot(question=prompt).answer

    elif model_choice == "GPT-4o-mini":
        # Load GPT-4o-mini model
        llm = LM(
                api_key=os.getenv("OPENAI_API_KEY"),
                model="gpt-4o-mini",
                api_base=os.getenv("PROXY_URL"),
                temperature=0.5,
            )
        #TODO: load dspy model
        if prediction[0][0] in ["__label__0", 0, False]:
            answer = "Not relevant to the domain, try another question."
        else:
            qa_bot = dspy.Predict("question -> answer: str")
            answer = qa_bot(question=prompt).answer
    else:
        answer = "Invalid model selection."

    # Append the user prompt and the answer to the chat history and return it.
    history.append((prompt, answer))
    return history

# Build a Gradio Blocks interface including a model dropdown and chatbot.
with gr.Blocks() as demo:
    gr.Markdown("# Chat with Model Selection")
    with gr.Row():
        model_dropdown = gr.Dropdown(
            choices=["fastText", "SVM", "XGBoost", "LLM", "ModernBERT"],
            value="LLM",
            label="Choose Model"
        )
    chat = gr.Chatbot()
    txt = gr.Textbox(
        show_label=False,
        placeholder="Enter your message and press enter"
    )

    def respond(message: str, chat_history: list, model_choice: str) -> tuple:
        return classify_prompt(message, chat_history, model_choice), ""
    txt.submit(respond, inputs=[txt, chat, model_dropdown], outputs=[chat, txt])

# Launch the app
demo.launch()
