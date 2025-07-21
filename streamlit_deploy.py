import streamlit as st
from api import ask_q
from main import core_components
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from langchain.callbacks.base import BaseCallbackHandler
from api import Question

if 'chat_history' not in st.session_state:
    st.session_state.chat_history = [] # if ther is no history, create empty listt

llm, retriever, reasoning_template, embeddings, db = core_components() #unpacking components (for streaming)

memory = ConversationBufferMemory(
    memory_key='chat_history',
    return_messages=True
)

qa_chain = ConversationalRetrievalChain.from_llm(
    llm=llm,
    retriever=retriever,
    memory=memory,
    combine_docs_chain_kwargs={'prompt': reasoning_template}
)

st.set_page_config(page_title='AI for personal docs', page_icon='📔') # sets page title and favicon
st.title('AI for personal docs') # the main title on the page
for m in st.session_state.chat_history:
    with st.chat_message(m['role']): # role => user or AI
        st.markdown(m['content']) # st.markdown to print the text out

#to figure out user's name for later use in the chat
if 'name' not in st.session_state:
    st.session_state.name = st.text_input("What's your name?")
    st.stop()

upload = st.file_uploader("Choose the doc") # for uploading own docss

inp = st.chat_input(f'What do you want to know, {st.session_state.name}?')
if inp: #trigerring chat response
    st.chat_message('user').markdown(inp) # show user input
    st.session_state.chat_history.append({'role': 'user', 'content': inp}) # adding new message to the list that stores converstion
    with st.chat_message('AI'):
        r_container = st.empty() # this creates container for streaming; it is space that streaming will gradually be filling up: for showing on page

        partial = '' #each answer will begin from empty string and then grow in it, every time from strach; for scratching before showing

        class StreamHandler(BaseCallbackHandler):
            def __init__(self, container):
                self.container = container
                self.partial = ""

            def on_llm_new_token(self, token: str, **kwargs):
                self.partial += token
                self.container.markdown(partial)
                return super().on_llm_new_token(token, **kwargs)
                
        with st.spinner('Reasoning...'): # loading animation
            chat_id = st.session_state.name
            answer = ask_q(Question(question=inp, chat_id=chat_id), callbacks=[StreamHandler(r_container)]) # calling ASK_Q function from API module
            st.markdown(answer) # returning result
    st.session_state.chat_history.append({'role': 'assistant', 'content': answer}) # saving ASSISTANT's response to chat history