import os
import time
import streamlit as st
from dotenv import load_dotenv 

from langchain_community.document_loaders import TextLoader, PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from src.config import CHROMA_DIR, DATA_DIR

load_dotenv()

def build_vector_db():
    """Reads documents from the data directory and compiles them safely into ChromaDB."""
    api_key = os.getenv("GOOGLE_API_KEY")
    
    embeddings = GoogleGenerativeAIEmbeddings(
        model="gemini-embedding-2-preview", 
        google_api_key=api_key
    )
    
    # 1. Process all text and PDF research paper sources from /data
    documents = []
    for file in os.listdir(DATA_DIR):
        file_path = os.path.join(DATA_DIR, file)
        
        # Keep your original text file loader
        if file.endswith(".txt"):
            print(f"Reading Text: {file}...")
            loader = TextLoader(file_path, encoding='utf-8')
            documents.extend(loader.load())
            
        # Add this new loader to handle your Andrew Ng research papers
        elif file.endswith(".pdf"):
            print(f"Reading PDF Paper: {file}...")
            loader = PyPDFLoader(file_path)
            documents.extend(loader.load())
            
    if not documents:
        print("Data directory is empty. Place your text or PDF files in /data.")
        return None

    # 2. Slice text up into optimized chunks
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=700, chunk_overlap=100)
    docs = text_splitter.split_documents(documents)
    total_chunks = len(docs)
    print(f"\nTotal chunks generated: {total_chunks}")
    print("Beginning throttled ingestion to avoid 429 Quota Exceeded...")

    # 3. Initialize Chroma manually
    db = Chroma(persist_directory=CHROMA_DIR, embedding_function=embeddings)

    # 4. Safely process in smaller batches of 20 strings with adaptive retry logic
    batch_size = 20
    total_batches = (total_chunks + batch_size - 1) // batch_size
    
    for i in range(0, total_chunks, batch_size):
        batch = docs[i:i + batch_size]
        current_batch_num = i // batch_size + 1
        print(f"Uploading batch {current_batch_num}/{total_batches} (Chunks {i} to {i + len(batch)})...")
        
        success = False
        delay = 90  # Start with a strong 90-second recovery window if blocked
        
        while not success:
            try:
                db.add_documents(batch)
                success = True
            except Exception as e:
                if "429" in str(e) or "RESOURCE_EXHAUSTED" in str(e):
                    print(f"\n⚠️ Quota completely empty. Sleeping for {delay} seconds...")
                    time.sleep(delay)
                    delay = min(delay + 30, 180)  # Increase delay incrementally if still blocked
                    print("Retrying current batch...")
                else:
                    print(f"Unexpected error encountered: {e}")
                    raise e
        
        # Baseline pacing break between successful batches to protect the tier metrics
        if i + batch_size < total_chunks:
            time.sleep(10)
            
    print(f"\n🎉 Success! Vectorized all {total_chunks} chunks to local workspace at {CHROMA_DIR}")
    return db

@st.cache_resource
def get_retriever():
    api_key = os.getenv("GOOGLE_API_KEY")
    embeddings = GoogleGenerativeAIEmbeddings(
        model="gemini-embedding-2-preview", 
        google_api_key=api_key
    )
    return Chroma(persist_directory=CHROMA_DIR, embedding_function=embeddings).as_retriever(
        search_kwargs={"k": 3}
    )

if __name__ == "__main__":
    build_vector_db()