from fastapi import FastAPI
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware
from main import core_components
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"]
)

class Question(BaseModel): # to validate requests
    question: str #str because user asks question in text form
    chat_id: str # to track memory per user 

llm, retriever, reasoning_template, embeddings, db = core_components() # unpacking core_components
memory_store = {} # for chat_id -> creating empty dict to store memory objects

def get_memory(chat_id): # takes key from dict
    if chat_id not in memory_store:
        memory_store[chat_id] = ConversationBufferMemory(memory_key='chat_history', return_messages=True)
    return memory_store[chat_id]

@app.post('/ask')
def ask_q(q: Question, callbacks=None):
    user_memory = get_memory(q.chat_id)
    qa_chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=retriever,
        memory=user_memory,
        combine_docs_chain_kwargs={'prompt': reasoning_template},
        callbacks=callbacks
    )
    answer = qa_chain.invoke(q.question)
    return answer
