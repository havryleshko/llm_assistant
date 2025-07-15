from dotenv import load_dotenv
from langchain_openai import OpenAIEmbeddings
from langchain.chains import ConversationalRetrievalChain
from langchain_community.vectorstores import FAISS
import os
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate, SystemMessagePromptTemplate, HumanMessagePromptTemplate
from langchain.schema import Document
from langchain.memory import ConversationBufferMemory

load_dotenv()
openai_api_key = os.getenv('OPENAI_API_KEY') # getting the key from .env file
embeddings = OpenAIEmbeddings(openai_api_key=openai_api_key)

db = FAISS.load_local('faiss_index', embeddings, allow_dangerous_deserialization=True) # getting the faiss from local folder

llm = ChatOpenAI(openai_api_key=openai_api_key, model='gpt-3.5-turbo') # setting up llm model for reasonin
retriever = db.as_retriever() # turns memory into search engine

reasoning_template = ChatPromptTemplate.from_messages([
    SystemMessagePromptTemplate.from_template(
        "You are a thoughtful AI assistant. When you answer a question, always think step by step, explain each step clearly, and reason out loud before giving the final answer."
    ),
    HumanMessagePromptTemplate.from_template(
        'Here is the context:\n{context}\n\nAnswer this question:\n{question}'
    )
])

# CREATING SHORT-TERM MEMORY 

short_memory = ConversationBufferMemory(
    memory_key='chat_history',
    return_messages=True
) # recalls past answers and questions and stores in DB

qa_chain = ConversationalRetrievalChain.from_llm(
    llm=llm,
    retriever=retriever,
    memory=short_memory,
    combine_docs_chain_kwargs={'prompt': reasoning_template}
)

#CREATING LONG-TERM MEMORY

#this try/except block tries to load FAISS_index from local, but in case there is none it creates a new one not to raise error
try:
    long_term_memory = FAISS.load_local(
        'long_term_memory', embeddings=embeddings, allow_dangerous_deserialization=True) #loading faiss_index
except:
    dummy = Document(page_content='Initialise FAISS doc', metadata={})
    long_term_memory = FAISS.from_documents([dummy], embeddings) #creates new if none


while True: # to keep asking question forever
    query = input('\nLearn fast, ask (or "exit"): ')
    if query.lower() == 'exit':
        break

    answer = qa_chain.run(query)
    print(answer)

    #storing qa chain long-term
    qa_doc = Document(
        page_content=f'Q: {query}\nA: {answer}',
        metadata={'source': 'chat'}
    )
    long_term_memory.add_documents([qa_doc])
    long_term_memory.save_local('long_term_memory')
