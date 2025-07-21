from dotenv import load_dotenv
from langchain_openai import OpenAIEmbeddings
from langchain.chains import ConversationalRetrievalChain
from langchain_community.vectorstores import FAISS
import os
from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate, SystemMessagePromptTemplate, HumanMessagePromptTemplate
from langchain.schema import Document
from langchain.memory import ConversationBufferMemory
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler

def core_components():
    load_dotenv()
    openai_api_key = os.getenv('OPENAI_API_KEY') # getting the key from .env file
    embeddings = OpenAIEmbeddings(openai_api_key=openai_api_key)

    db = FAISS.load_local('faiss_index', embeddings, allow_dangerous_deserialization=True) # getting the faiss from local folder

    llm = ChatOpenAI(openai_api_key=openai_api_key, model='gpt-4o', streaming=True, callbacks=[StreamingStdOutCallbackHandler()]) # setting up llm model for reasonin, additionally adding streaming
    retriever = db.as_retriever() # turns memory into search engine

    reasoning_template = PromptTemplate.from_messages([
        SystemMessagePromptTemplate.from_template(
            """You are a thoughtful AI assistant. When you answer a question, always think step by step, explain each step clearly, and reason out loud before giving the final answer.
            Use both the user's chat history and retrieved documents to answer the question. Use chain-of-thought reasoning. Be logical, but give answer in a human-like manner.
            You are an insightful and friendly AI who is not afraid of giving your own opinion when asked. Use human-like tone and give your opinions when asked and don't be afraid to say whatever you want and try to justify it"""
        ),
        HumanMessagePromptTemplate.from_template(
            'Conversation history:\n{chat_history}\n\nHere is the context (doc):\n{context}\n\nAnswer this question:\n{question}'
        )
    ])
    return llm, retriever, reasoning_template, embeddings, db

def long_term_m():
    llm, retriever, reasoning_template, embeddings = core_components()
    memory = ConversationBufferMemory(memory_key='chat_history', return_messages=True)
    qa_chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=retriever,
        memory=memory,

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

        return long_term_memory

if __name__ == '__main__':
    long_term_m()
