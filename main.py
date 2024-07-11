import os
import streamlit as st
import pickle
import time
import requests
from bs4 import BeautifulSoup
from langchain import OpenAI
from langchain.chains import RetrievalQAWithSourcesChain
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.document_loaders import UnstructuredURLLoader
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from dotenv import load_dotenv
import pathway as pw

load_dotenv()  

st.title("Let me Help You 🤝😊")
st.sidebar.title("FAQ Bot \n Link to FAQ section can also be added here")

urls = [
    "https://www.amazon.in/gp/help/customer/display.html?nodeId=GDF5PQP4Z6SUH4CQ"
]

file_path = "faiss_store_openai.pkl"

main_placeholder = st.empty()
llm = OpenAI(temperature=0.9, max_tokens=500)

try:
    loader = UnstructuredURLLoader(urls=urls)
    main_placeholder.text("Data Loading...")
    data = loader.load()

    text_splitter = RecursiveCharacterTextSplitter(
        separators=['\n\n', '\n', '.', ','],
        chunk_size=1000
    )
    main_placeholder.text("Please Wait")
    docs = text_splitter.split_documents(data)

    if docs:
        df = pw.DataFrame(docs)

        embeddings = OpenAIEmbeddings()
        vectorstore_openai = FAISS.from_documents(df, embeddings)
        main_placeholder.text("Please wait")
        time.sleep(2)

        with open(file_path, "wb") as f:
            pickle.dump(vectorstore_openai, f)
    else:
        main_placeholder.text("No documents to process")
except Exception as e:
        main_placeholder.text(f"An error occurred: {e}")

query = main_placeholder.text_input("Question: ")
if query:
    if os.path.exists(file_path):
        try:
            with open(file_path, "rb") as f:
                vectorstore = pickle.load(f)
                chain = RetrievalQAWithSourcesChain.from_llm(llm=llm, retriever=vectorstore.as_retriever())
                result = chain({"question": query}, return_only_outputs=True)

                st.header("Answer")
                personalized_answer = f"Hi there! 😊 Here's what I found for your question:\n\n{result['answer']}\n\nI hope this helps🤝! If you have any more questions, feel free to ask. 😊"
                st.write(personalized_answer)

        except Exception as e:
            st.write(f"An error occurred while retrieving the answer: {e}")
    else:
        st.write("FAISS index not found. URL not processed! Please double-check the API")
