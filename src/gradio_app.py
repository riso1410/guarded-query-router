import gradio as gr
from langchain_openai import ChatOpenAI
from langchain.schema import AIMessage, HumanMessage
import os 
import dotenv

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

def classify() -> bool:
    return True

def respond(message: str, history: list) -> str:
    # TODO 
    # 1. Load model and embedding model 
    # 2. Decide on the query to be sent to the model or not 
    # 3. Send the query to the model
    # 4. Return the response from the model

    if classify():
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