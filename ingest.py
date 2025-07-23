# this script is like one time process to prepare the documents for interpretation (framework for live app)
# 1. loads docs from /data
# 2. splits them into parts
# 3. converts into embeddings using OpenAI

from langchain_community.document_loaders import PyPDFLoader, CSVLoader, UnstructuredExcelLoader
from langchain_openai import OpenAIEmbeddings
from dotenv import load_dotenv
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
import os

load_dotenv()
openai_api_key = os.getenv('OPENAI_API_KEY') # loads .env file 

def ingest(file_path: str, save_path: str = 'faiss_index'):
    ext = os.path.splitext(file_path)[-1].lower() # extracting file extension from the path; [-1] takes last item in split - ext

    if ext == '.pdf':
        loader = PyPDFLoader(file_path)
    elif ext == '.csv':
        loader = CSVLoader(file_path)
    elif ext in ['.xls', '.xlsx']:
        loader = UnstructuredExcelLoader(file_path) 
    else: 
        raise ValueError(f"Unsupported file type: '{ext}'")
    
    documents = loader.load()
    splitter = RecursiveCharacterTextSplitter( #recursive for smart text splitting 
        chunk_size=500,
        chunk_overlap=100
    ) #500 and 100 are a good starting point for PDFs
    docs = splitter.split_documents(documents) #splits the doc chosen
    embeddings = OpenAIEmbeddings(openai_api_key=openai_api_key) #chunks into embeddings (OpenAI key)
    vectors = FAISS.from_documents(docs, embeddings) #storing in vector database
    vectors.save_local(save_path) #storing locally
    print(f"FAISS index saved locally as '{save_path}'")



