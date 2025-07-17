import streamlit as st
from langchain_community.vectorstores import FAISS
from langchain.chains import ConversationRetrievalChain
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain.memory import ConversationBufferMemory
from dotenv import load_dotenv
import os

load_dotenv()

llm = ChatOpenAI()
embeddings = OpenAIEmbeddings()
db = FAISS.load_local('long_term_memory', embeddings)
retriever = db.as_retriever()
memory = ConversationBufferMemory(memory_key='chat_history', return_messages=True)

qa_chain = ConversationRetrievalChain.from_llm(
    llm=llm,
    memory=memory,
    retriever=retriever
)