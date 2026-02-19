import streamlit as st
from langchain_community.llms import Ollama
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.vectorstores import FAISS

st.title("🤖 Hybrid RAG Chatbot (PDF + General)")

# -------- Load PDF --------
loader = PyPDFLoader("AI Question Bank.pdf")
documents = loader.load()

splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=50
)
docs = splitter.split_documents(documents)

# -------- Embeddings --------
embeddings = OllamaEmbeddings(model="nomic-embed-text")
vectorstore = FAISS.from_documents(docs, embeddings)

# -------- LLM --------
llm = Ollama(
    model="phi3",
    base_url="http://localhost:11434"
)

# -------- UI --------
question = st.text_input("Ask any question")

if question:
    # Retrieve relevant docs
    retrieved_docs = vectorstore.similarity_search(question, k=3)

    if retrieved_docs:
        context = "\n\n".join([doc.page_content for doc in retrieved_docs])

        prompt = f"""
You are a helpful assistant.
Use the context if it is relevant.
If the context is not helpful, answer from your own knowledge.

Context:
{context}

Question:
{question}
"""
    else:
        # No context → general question
        prompt = question

    response = llm.invoke(prompt)
    st.write(response)
