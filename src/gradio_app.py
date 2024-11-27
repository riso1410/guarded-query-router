import gradio as gr
from langchain_openai import ChatOpenAI
from langchain.schema import AIMessage, HumanMessage
from llm_classifier import ClassificationModule
import os 
import dotenv
import fasttext
import joblib
from fastembed import TextEmbedding
from dspy import LM, configure

dotenv.load_dotenv()

api_key = os.getenv("OPENAI_API_KEY")
proxy_url = os.getenv("PROXY_URL")

llm = ChatOpenAI(
    model="gpt-4o-mini",
    base_url=proxy_url,
    api_key=api_key,
    temperature=0.5,
    max_tokens=250,
    )

lm = LM(
    api_key=api_key,
    model="gpt-4o-mini",
    api_base=proxy_url,
    temperature=0,
    max_tokens=250
    )
configure(lm=lm)

#classifier = fasttext.load_model("C:/Users/riso/Desktop/Prompt-Classification/models/Fasttext.bin")
classifier = joblib.load("C:/Users/riso/Desktop/Prompt-Classification/models/SVM_TFIDF.joblib")
#classifier = ClassificationModule()
#classifier.load("C:/Users/riso/Desktop/Prompt-Classification/models/gpt-4o-mini.json")

embedding = TextEmbedding(model_name="BAAI/bge-small-en-v1.5")

def classify(msg: str) -> bool:
    msg = list(embedding.embed(msg))
    label = classifier.predict(msg)

    #label = classifier(prompt=msg, domain="law").label
    return label in ["__label__1", "1", True]

def respond(message: str, history: list) -> str:
    if classify(message):
        history_langchain_format = []

        for msg in history:
            if msg['role'] == "user":
                history_langchain_format.append(HumanMessage(content=msg['content']))
            elif msg['role'] == "assistant":
                history_langchain_format.append(AIMessage(content=msg['content']))

        history_langchain_format.append(HumanMessage(content=message))
        gpt_response = llm.invoke(history_langchain_format)
        return gpt_response.content

    return "The question is out of my domain."

gr.ChatInterface(respond, type="messages").launch()