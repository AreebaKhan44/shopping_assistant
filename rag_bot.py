import os
import streamlit as st
from dotenv import load_dotenv

# ✅ Updated imports
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import SentenceTransformerEmbeddings
from langchain_core.documents import Document
from langchain_community.llms import OpenAI
from langchain.chains import RetrievalQA

# Load environment variables
load_dotenv()

# ✅ Load documents from a text file
def load_docs(file_path='products.txt'):
    with open(file_path, 'r') as f:
        lines = f.readlines()
    return [Document(page_content=line.strip()) for line in lines if line.strip()]

# ✅ Build vector index with embeddings
def build_retriever():
    docs = load_docs()
    embeddings = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")
    db = FAISS.from_documents(docs, embeddings)
    retriever = db.as_retriever(search_kwargs={"k": 2})
    return retriever

# ✅ Create Retrieval-Augmented Generation (RAG) bot
def create_rag_bot():
    retriever = build_retriever()

    # ✅ Use Streamlit secret for API key in deployment
    openai_key = st.secrets["OPENAI_API_KEY"] if "OPENAI_API_KEY" in st.secrets else os.getenv("OPENAI_API_KEY")

    if not openai_key:
        raise ValueError("OpenAI API key not found in secrets or environment variables.")

    llm = OpenAI(temperature=0, openai_api_key=openai_key)
    qa_chain = RetrievalQA.from_chain_type(llm=llm, retriever=retriever)
    return qa_chain

# ✅ Ask the bot a question and return the response
def ask_question(question):
    bot = create_rag_bot()
    return bot.run(question)
