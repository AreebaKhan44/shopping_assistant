

import os
import streamlit as st
from dotenv import load_dotenv

# ✅ Only needed for local development (ignored on Streamlit Cloud)
load_dotenv()

# ✅ Prevent TensorFlow-related warnings if using torch-only
os.environ["USE_TF"] = "0"

# LangChain + FAISS + Embeddings
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.documents import Document
from langchain_community.llms import OpenAI
from langchain.chains import RetrievalQA


# Load product knowledge base
def load_docs(file_path='products.txt'):
    with open(file_path, 'r') as f:
        lines = f.readlines()
    return [Document(page_content=line.strip()) for line in lines if line.strip()]


# Build retriever with FAISS + embeddings
def build_retriever():
    docs = load_docs()
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    db = FAISS.from_documents(docs, embeddings)
    retriever = db.as_retriever(search_kwargs={"k": 2})
    return retriever


# Create RAG bot using OpenAI + retriever
def create_rag_bot():
    retriever = build_retriever()

    # ✅ Use Streamlit secrets on deployment; fallback to .env for local
    api_key = st.secrets.get("OPENAI_API_KEY") or os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise ValueError("❌ OPENAI_API_KEY not found. Set it in .env (for local) or secrets.toml (for deployment).")

    # Instantiate OpenAI LLM
    llm = OpenAI(temperature=0, openai_api_key=api_key)

    # Create Retrieval QA chain
    qa_chain = RetrievalQA.from_chain_type(llm=llm, retriever=retriever)
    return qa_chain


# Ask the bot a question
def ask_question(question):
    bot = create_rag_bot()
    return bot.run(question)

