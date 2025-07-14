from dotenv import load_dotenv
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.chains import RetrievalQA
import os
from langchain_openai import ChatOpenAI

load_dotenv()
openai_api_key = os.getenv('OPENAI_API_KEY') # getting the key from .env file
embeddings = OpenAIEmbeddings(openai_api_key=openai_api_key)

db = FAISS.load_local('faiss_index', embeddings, allow_dangerous_deserialization=True) # getting the faiss from local folder

llm = ChatOpenAI(openai_api_key=openai_api_key, model='gpt-3.5-turbo') # setting up llm model for reasoning
retriever = db.as_retriever() # turns memory into search engine
qs = RetrievalQA.from_chain_type(llm=llm, retriever=retriever) # combining memory + brain

while True: # to keep asking question forever
    query = input('\nLearn fast, ask (or "exit"): ')
    if query.lower() == 'exit':
        break
    answer = qs.run(query)
    print(answer)