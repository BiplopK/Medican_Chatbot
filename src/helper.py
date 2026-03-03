from langchain_community.document_loaders import PyPDFLoader,DirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings


def load_pdf(data_path):
    loader=DirectoryLoader(
        data_path,
        glob="*.pdf",
        loader_cls=PyPDFLoader
    )

    documents=loader.load()
    return documents


def filter_doc(extracted_data):
    filtered=[]
    for doc in extracted_data:
        src=doc.metadata.get("source")
        filtered.append(
            Document(
                page_content=doc.page_content,
                meta_data={'source':src}
            )
        )
    
    return filtered

def text_split(filtered_doc):
    text_splitter=RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=20
    )
    texts_chunk=text_splitter.split_documents(filtered_doc)
    return texts_chunk



def initial_embedding():
    model_name= "sentence-transformers/all-MiniLM-L6-v2"
    embeddings=HuggingFaceEmbeddings(
        model_name=model_name
    )
    return embeddings

