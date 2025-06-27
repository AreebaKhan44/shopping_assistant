import os
import streamlit as st
from dotenv import load_dotenv

# ✅ Force PyTorch-only (avoids TensorFlow DLL errors)
os.environ["USE_TF"] = "0"

# ✅ Updated imports
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.documents import Document
from langchain_community.llms import OpenAI
from langchain.chains import RetrievalQA

load_dotenv()


# Load knowledge base
def load_docs(file_path='products.txt'):
    with open(file_path, 'r') as f:
        lines = f.readlines()
    return [Document(page_content=line.strip()) for line in lines if line.strip()]

# Build vector index
def build_retriever():
    docs = load_docs()
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    db = FAISS.from_documents(docs, embeddings)
    retriever = db.as_retriever(search_kwargs={"k": 2})
    return retriever

# Create RAG pipeline
def create_rag_bot():
    retriever = build_retriever()
    llm = OpenAI(temperature=0, openai_api_key=os.getenv("OPENAI_API_KEY"))
    qa_chain = RetrievalQA.from_chain_type(llm=llm, retriever=retriever)
    return qa_chain

# Get response
def ask_question(question):
    bot = create_rag_bot()
    return bot.run(question)
