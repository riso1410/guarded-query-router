import os

import dspy
import fasttext
import gradio as gr
from dotenv import load_dotenv
from dspy import LM, configure

load_dotenv()

def classify_prompt(prompt: str, history: list) -> str:
    fasttext_model = fasttext.load_model('models/fastText_law_fasttext.bin')

    # FastText prediction
    prediction = fasttext_model.predict(prompt)  # Get first prediction
    print(prediction)
    if prediction[0][0] in ["__label__0", 0, False]:
        return "Not relevant to the domain, try another question."
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

demo = gr.ChatInterface(classify_prompt, type="messages", share=False).launch()
