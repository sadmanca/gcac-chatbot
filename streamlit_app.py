import streamlit as st
from llama_index import VectorStoreIndex, ServiceContext, Document
from llama_index.llms import OpenAI
import openai
from llama_index import SimpleDirectoryReader
import time

from langchain.memory import ConversationBufferMemory
memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)

from langchain.chat_models import ChatOpenAI
from langchain.chains import ConversationChain
from langchain.chains.conversation.memory import ConversationBufferWindowMemory
from langchain.prompts import (
    SystemMessagePromptTemplate,
    HumanMessagePromptTemplate,
    ChatPromptTemplate,
    MessagesPlaceholder
)

st.set_page_config(page_title="Chatbot | GCAC Wayfinding Project", page_icon="🔵", layout="centered", initial_sidebar_state="auto", menu_items=None)
st.title("UofT Graduate Centre for Academic Communication (GCAC) Chatbot 💬")
st.info("Check out everything there is know at the [GCAC website](https://www.sgs.utoronto.ca/resources-supports/gcac/)", icon="📃")

openai.api_key = st.secrets["OPENAI_API_KEY"]
from langchain.chat_models import ChatOpenAI
from langchain.chains import RetrievalQA
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler

import pinecone
pinecone.init(  
    api_key="39de8dc5-781a-4a69-949f-742dc27c6161",  # find at app.pinecone.io
    environment="gcp-starter"  # next to api key in console
)
from langchain.embeddings.openai import OpenAIEmbeddings
embeddings = OpenAIEmbeddings()
from langchain.vectorstores import Pinecone
index_name = "test"
docsearch = Pinecone.from_existing_index(index_name, embeddings)
from langchain.chains import RetrievalQA
from langchain.llms import OpenAI
# llm=OpenAI(
#     model="gpt-3.5-turbo",
# )


from langchain.callbacks.base import BaseCallbackHandler
class StreamHandler(BaseCallbackHandler):
    def __init__(self, container):
        self.message_placeholder = container
        self.full_response = ""

    def on_llm_new_token(self, token: str, **kwargs):
        self.full_response += token + ""
        self.message_placeholder = self.message_placeholder.markdown(self.full_response + "▌")

    def on_llm_end(self, finish, **kwargs):
        self.message_placeholder = self.message_placeholder.markdown(self.full_response)

llm = ChatOpenAI(
    model_name="gpt-3.5-turbo",
    streaming=True,
    temperature=0.5,
    openai_api_key=st.secrets["OPENAI_API_KEY"],
)

from langchain.memory import ConversationBufferMemory
memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True, input_key='query', output_key='result')

qa_with_sources = RetrievalQA.from_chain_type(
    llm=llm,
    chain_type="stuff",
    retriever=docsearch.as_retriever(),
    return_source_documents=True,
    memory=memory
)

# with open( "app\style.css" ) as css:
#     st.markdown( f'<style>{css.read()}</style>' , unsafe_allow_html= True)

if "messages" not in st.session_state.keys(): # Initialize the chat messages history
    st.session_state.messages = [
        {"role": "assistant", "content": "Ask me a question about the GCAC's course offerings!"},
        {"role": "assistant", "content": "What are you looking for today?"}
    ]

# @st.cache_resource(show_spinner=False)
# def load_data():
#     with st.spinner(text="Loading and indexing GCAC course metadata – hang tight! This should take 1-2 minutes."):
#         reader = SimpleDirectoryReader(input_dir="./data", recursive=True)
#         docs = reader.load_data()
#         service_context = ServiceContext.from_defaults(llm=OpenAI(model="gpt-3.5-turbo", temperature=0.2, system_prompt="You are an expert on communications development course offerings at the University of Toronto (UofT) Graduate Centre for Academic Communication (GCAC) and your job is to help students figure out what courses they should take as well as to discover the different types of courses that are available to them. Assume that all questions are related to the GCAC communications development courses. Keep your answers helpful and based on facts - do not hallucinate features."))
#         index = VectorStoreIndex.from_documents(docs, service_context=service_context)
#         return index

# index = load_data()
# # chat_engine = index.as_chat_engine(chat_mode="condense_question", verbose=True, system_prompt="You are an expert on the Streamlit Python library and your job is to answer technical questions. Assume that all questions are related to the Streamlit Python library. Keep your answers technical and based on facts – do not hallucinate features.")
# chat_engine = index.as_chat_engine(chat_mode="condense_question", verbose=True, streaming=True)

# def my_callback(text_to_display):
#     prompt = text_to_display
#     print(prompt)

if prompt := st.chat_input("Your question"): # Prompt for user input and save to chat history
    st.session_state.messages.append({"role": "user", "content": prompt})

for message in st.session_state.messages:  # Display the prior chat messages
    with st.chat_message(message["role"]):
        st.write(message["content"])
        if 'links' in message:
            for link in message['links']:
                st.link_button(link['title'], link['source_url'])

# st.button("test", on_click=my_callback, args=("hi",))

# If last message is not from assistant, generate a new response
if st.session_state.messages[-1]["role"] != "assistant":
    st.markdown("""
        <style>
            @keyframes highlight {
                0% { background-color: white; }
                25% { background-color: #ff6c6c; }
                50% { background-color: white; }
                75% { background-color: #ff6c6c; }
                100% { background-color: white; }
            }

            .stChatFloatingInputContainer {
                animation: highlight 3s ease-in-out infinite;
                padding: 4px !important;
                margin: 0 !important;
                margin-bottom: 60px !important;
                border-radius: 10px !important;
            }
        </style>
        """, unsafe_allow_html=True)
        
    with st.chat_message("assistant"):

        message_placeholder = st.container()
        handler = StreamHandler(message_placeholder)

        # prompt += "; if mentioning courses, provide links with the course information in a markdown table (use columns like title, abbreviation, type of course, division description, link) after providing information via text (always provide information via text first). If mentioning FAS or FASE, include their courses as well. Keep your answers helpful and based on facts - do not hallucinate features."
        # callback = StreamHandler(st.empty())
        # result = qa_with_sources({"query": prompt}, callbacks=callback)
        links_container = st.container()
        result = qa_with_sources({"query": prompt}, callbacks = [handler])
        links_container.caption("Sources:")
        for document in result["source_documents"]:
            source_url = document.metadata['source']
            title = document.metadata['title']
            links_container.link_button(title, source_url)
            
        links = [{'title': document.metadata['title'], 'source_url': document.metadata['source']} for document in result["source_documents"]]
        st.session_state.messages.append({"role": "assistant", "content": handler.full_response, "links": links})

        # response = result["result"]
        # response = chat_engine.stream_chat(prompt)

        # for token in response.response_gen:
            # full_response += token + ""
            # Add a blinking cursor to simulate typing
            # message_placeholder.markdown(full_response + "▌")

        # full_response = response

        # message_placeholder.markdown(full_response)

    # Add assistant response to chat history
    # st.session_state.messages.append({"role": "assistant", "content": full_response})

    # st.write(response.response)
    # message = {"role": "assistant", "content": response.response}
    # st.session_state.messages.append(message) # Add response to message history

from streamlit_extras.stylable_container import stylable_container

# if 'animation_played' not in st.session_state:
#     st.session_state['animation_played'] = False

# if not st.session_state['animation_played']:
#     st.markdown("""
#         <style>
#             @keyframes highlight {
#                 0% { background-color: white; }
#                 25% { background-color: #002a5c; }
#                 50% { background-color: white; }
#                 75% { background-color: #002a5c; }
#                 100% { background-color: white; }
#             }

#             .stChatFloatingInputContainer {
#                 animation: highlight 15s ease-in-out infinite;
#                 padding: 4px !important;
#                 margin: 0 !important;
#                 margin-bottom: 60px !important;
#                 border-radius: 10px !important;
#             }
#         </style>
#         """, unsafe_allow_html=True)
#     st.session_state['animation_played'] = True

if 'animation_played' not in st.session_state:
    st.session_state['animation_played'] = False

if st.session_state.messages[-1]["role"] != "user":
    st.markdown("""
        <style>
            @keyframes highlight {
                0% { background-color: white; }
                25% { background-color: #73b9ff; }
                50% { background-color: white; }
                75% { background-color: #73b9ff; }
                100% { background-color: white; }
            }

            .stChatFloatingInputContainer {
                animation: highlight 12s ease-in-out infinite;
                padding: 4px !important;
                margin: 0 !important;
                margin-bottom: 60px !important;
                border-radius: 10px !important;
            }
        </style>
        """, unsafe_allow_html=True)