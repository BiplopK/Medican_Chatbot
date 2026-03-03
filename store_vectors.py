from dotenv import load_dotenv
import os
from pinecone import Pinecone,ServerlessSpec
from langchain_pinecone import PineconeVectorStore

from src.helper import load_pdf,text_split,initial_embedding,filter_doc


load_dotenv()

extracted_doc=load_pdf(r"C:\Users\Acer\Documents\Medical_chatbot\Medican_Chatbot\data")
filtered_documents=filter_doc(extracted_doc)
chunk_text=text_split(filtered_documents)

embeddings=initial_embedding()

PINECONE_API_KEY=os.getenv("PINECONE_API_KEY")
GOOGLE_API_KEY=os.getenv("GOOGLE_API_KEY")

os.environ["PINECONE_API_KEY"]=PINECONE_API_KEY
os.environ["GOOGLE_API_KEY"]=GOOGLE_API_KEY

pinecone_api_key=PINECONE_API_KEY
pc=Pinecone(api_key=pinecone_api_key)

index_name="medical-chatbot"
if not pc.has_index(index_name):
    pc.create_index(
        name=index_name,
        dimension=384,
        spec=ServerlessSpec(cloud='aws',region="us-east-1"),
        metric="cosine"
    )

index=pc.Index(index_name)

docsearch=PineconeVectorStore.from_documents(
    documents=chunk_text,
    embedding=embeddings,
    index_name=index_name
)