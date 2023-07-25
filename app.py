import streamlit as st
import os
import csv
from dotenv import load_dotenv
from PyPDF2 import PdfReader
from langchain.text_splitter import CharacterTextSplitter
from langchain.agents import create_csv_agent
from langchain.embeddings import OpenAIEmbeddings, HuggingFaceInstructEmbeddings
from langchain.vectorstores import FAISS
from langchain.chat_models import ChatOpenAI
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from htmlTemplates import css, bot_template, user_template
from langchain.llms import HuggingFaceHub


def get_file_text(docs):
    text = ""
    for doc in docs:
        if doc.type == 'application/pdf':
            pdf_reader = PdfReader(doc)
            for page in pdf_reader.pages:
                text += page.extract_text()
        elif doc.type == 'text/csv':
            with open(doc.name, newline='') as csvfile:
                csv_reader = csv.reader(csvfile, delimiter=',')
                for row in csv_reader:
                    text += row[0]

    return text


def get_text_chunks(text):
    text_splitter = CharacterTextSplitter(
        separator="\n",
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len
    )
    chunks = text_splitter.split_text(text)
    return chunks


def get_vectorstore(text_chunks):
    embeddings = OpenAIEmbeddings()
    # embeddings = HuggingFaceInstructEmbeddings(model_name="hkunlp/instructor-xl")
    vectorstore = FAISS.from_texts(texts=text_chunks, embedding=embeddings)
    return vectorstore


def get_conversation_chain(vectorstore):
    llm = ChatOpenAI()
    # llm = HuggingFaceHub(repo_id="google/flan-t5-xxl",
    #                model_kwargs={"temperature": 0.5, "max_length": 512})

    memory = ConversationBufferMemory(
        memory_key='chat_history', return_messages=True)
    conversation_chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=vectorstore.as_retriever(),
        memory=memory
    )
    return conversation_chain


def handle_userinput(user_question):
    response = st.session_state.conversation({'question': user_question})

    st.session_state.chat_history = response['chat_history']

    for i, message in enumerate(reversed(st.session_state.chat_history)):
        if i % 2 == 0:
            st.write(bot_template.replace(
                "{{MSG}}", message.content), unsafe_allow_html=True)
        else:
            st.write(user_template.replace(
                "{{MSG}}", message.content), unsafe_allow_html=True)


def main():
    load_dotenv()

    # Load the OpenAI API key from the environment variable
    if os.getenv("OPENAI_API_KEY") is None or os.getenv("OPENAI_API_KEY") == "":
        st.header(
            "Please set your OpenAI API key in the .env file and refresh the page.")
        exit(1)

    st.set_page_config(page_title="Chat with DOCs",
                       page_icon=":books:")
    st.write(css, unsafe_allow_html=True)

    if "conversation" not in st.session_state:
        st.session_state.conversation = None
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = None
    if "text_prompt" not in st.session_state:
        st.session_state.text_prompt = ""

    st.header("Chat with multiple DOCs :books::question:")

    user_question = st.text_input(
        "label text visibility set to hidden", label_visibility="hidden", placeholder="Enter your question here", key='text_prompt')

    with st.sidebar:
        st.subheader("Your documents")
        docs = st.file_uploader(
            "Upload your PDFs here and click on 'Process'", type=['pdf', 'csv'], accept_multiple_files=True)
        if st.button("Process"):
            with st.spinner("Processing"):
                # get pdf text
                raw_text = get_file_text(docs)

                # get the text chunks
                text_chunks = get_text_chunks(raw_text)

                # create vector store
                vectorstore = get_vectorstore(text_chunks)

                # create conversation chain
                st.session_state.conversation = get_conversation_chain(
                    vectorstore)

        # checkbox to view state values
        view_state = st.checkbox("View state values")
        if view_state:
            # st.write("st.session_state chat_history:",st.session_state['chat_history'])
            st.write("text prompt:",
                     st.session_state["text_prompt"])

    if user_question:
        if st.session_state.conversation is None:
            st.write(
                "Please upload and process your PDF documents before asking a question.")
        else:
            handle_userinput(user_question)


if __name__ == '__main__':
    main()
