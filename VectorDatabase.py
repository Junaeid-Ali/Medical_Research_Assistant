
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_community.document_loaders import DirectoryLoader, PyPDFLoader
from dotenv import load_dotenv

load_dotenv()




loader = DirectoryLoader(
    path="data",          # Root folder
    glob="**/*.pdf",      # Search all subfolders
    loader_cls=PyPDFLoader
)

docs = loader.load()

print(f"Total pages loaded: {len(docs)}")
# print(documents[0].page_content)

splitter=RecursiveCharacterTextSplitter(
    chunk_size=800,
    chunk_overlap=100
)

chunks=splitter.split_documents(docs)




embedding_model = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

vectorstore=Chroma.from_documents(
    embedding=embedding_model,
    documents=chunks,
    persist_directory="Medical-db"
)
