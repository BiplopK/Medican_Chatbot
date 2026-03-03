from flask import Flask, render_template,jsonify,request
from langchain_pinecone import PineconeVectorStore
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.runnables import RunnablePassthrough
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from dotenv import load_dotenv

from src.helper import initial_embedding
from src.prompts import *
import os

load_dotenv()

app=Flask(__name__)


PINECONE_API_KEY=os.getenv("PINECONE_API_KEY")
GOOGLE_API_KEY=os.getenv("GOOGLE_API_KEY")

os.environ["PINECONE_API_KEY"]=PINECONE_API_KEY
os.environ["GOOGLE_API_KEY"]=GOOGLE_API_KEY

embeddings=initial_embedding()

index_name="medical-chatbot"

docsearch=PineconeVectorStore.from_existing_index(
    embedding=embeddings,
    index_name=index_name
)

retriever=docsearch.as_retriever(search_type="similarity",search_kwargs={'k':3})
chatModel=ChatGoogleGenerativeAI(model="gemini-2.5-flash")
prompt=ChatPromptTemplate(
    [
        ('system',system_prompts),
        ('human','{input}'),
    ]
)

rag_chain=(
    {
        'context':lambda x: retriever.invoke(x['input']),
        'input':RunnablePassthrough()
    }|prompt | chatModel | StrOutputParser()
)

@app.route("/")
def index():
    return render_template("chat.html")


@app.route("/get",methods=["GET","POST"])
def chat():
    data = request.get_json()  # parse JSON from request body
    msg = data.get('msg')      # get 'msg' safely
    if not msg:
        return jsonify({"answer": "No message received!"})
    input=msg
    print(input)
    response=rag_chain.invoke({'input':msg})
    print("Response: ",response)
    return jsonify({"answer": response})

if __name__ == "__main__":
    app.run(host="0.0.0.0",port=8000,debug=True)