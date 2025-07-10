# this script is like one time process to prepare the documents for interpretation (framework for live app)
# 1. loads docs from /data
# 2. splits them into parts
# 3. converts into embeddings using OpenAI

from langchain.document_loaders import TextLoader
from langchain.document_loaders import PyPDFLoader
from langchain.embeddings import OpenAIEmbeddings
from dotenv import load_dotenv
from langchain.text_splitter import RecursiveCharacterTextSplitterTextSplitter
from langchain.vectorstores import FAISS

load_dotenv()

filename = input('Enter the path to PDF: ') #loading the doc
loader = PyPDFLoader(filename) #reading the doc

splitter = RecursiveCharacterTextSplitterTextSplitter( #recursive for smart text splitting 
    chunk_size=500,
    chunk_overlap=100
) #500 and 100 are a good starting point for PDFs

docs = splitter.split_documents(loader) #splits the doc chosen
embeddings = OpenAIEmbeddings() #chunks into embeddings (OpenAI key)

vectors = FAISS.from_documents(docs, embeddings) #storing in vector database
vectors.save_local('faiss_index') #storing locally


