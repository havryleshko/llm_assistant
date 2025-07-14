# this script is like one time process to prepare the documents for interpretation (framework for live app)
# 1. loads docs from /data
# 2. splits them into parts
# 3. converts into embeddings using OpenAI

from langchain_community.document_loaders import PyPDFLoader
from langchain_openai import OpenAIEmbeddings
from dotenv import load_dotenv
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
import os

load_dotenv()
openai_api_key = os.getenv('OPENAI_API_KEY') # loads .env file 

filename = input('Enter the path to PDF: ') #loading the doc
loader = PyPDFLoader(filename) #reading the doc
documents = loader.load()

splitter = RecursiveCharacterTextSplitter( #recursive for smart text splitting 
    chunk_size=500,
    chunk_overlap=100
) #500 and 100 are a good starting point for PDFs

docs = splitter.split_documents(documents) #splits the doc chosen
embeddings = OpenAIEmbeddings(openai_api_key=openai_api_key) #chunks into embeddings (OpenAI key)

vectors = FAISS.from_documents(docs, embeddings) #storing in vector database
vectors.save_local('faiss_index') #storing locally
print("FAISS index saved locally as 'faiss_index'")



