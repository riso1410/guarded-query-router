import os

import dspy
import gradio as gr
import joblib
from dotenv import load_dotenv
from dspy import LM
from sklearn.feature_extraction.text import TfidfVectorizer

load_dotenv()

def classify_prompt(prompt: str, histroy: list) -> str:
    model = joblib.load('models/SVM_law_tf_idf.pkl')

    prediction = model.predict([prompt])[0]

    if prediction == 0:
        prediction = "Not relevant to the domain, try another question."
    else:
        llm = LM(
            api_key=os.getenv("OPENAI_API_KEY"),
            model="gpt-4o-mini",
            api_base=os.getenv("PROXY_URL"),
            temperature=0.5,
        )
        llm.configure(lm=llm)

        qa_bot = dspy.Predict("question -> answer: str")
        return qa_bot(question=prompt).answer

demo = gr.ChatInterface(classify_prompt, type="messages").launch()
