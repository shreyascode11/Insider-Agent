import os
from langchain_community.embeddings import HuggingFaceEmbeddings
from src.vector_store import ClubVectorStore
from src.config import Config

from src.tools import get_vector_store

def load_all_documents():
    print("🚀 Initializing Knowledge Ingestion Pipeline...")
    
    # Initialize the embedding model and vector store
    store = get_vector_store()
    
    # Check if the database is already populated to prevent duplicating chunks on every new Streamlit session
    if store.collection.count() > 0:
        print("✅ Database already populated. Skipping duplicate ingestion.")
        return
        
    # Locate the raw documents folder
    docs_dir = Config.RAW_DOCS_PATH
    os.makedirs(docs_dir, exist_ok=True) # Ensures the folder exists
    
    # Find all text and markdown files
    files = [f for f in os.listdir(docs_dir) if f.endswith('.txt') or f.endswith('.md')]
    
    if not files:
        print(f"⚠️ No .txt or .md files found in {docs_dir}")
        print("Please save your official club manuals there and run again.")
        return

    # 3. Read, chunk, and ingest each file
    for filename in files:
        filepath = os.path.join(docs_dir, filename)
        print(f"\n📄 Reading {filename}...")
        
        with open(filepath, 'r', encoding='utf-8') as file:
            content = file.read()
            
        print(f"🧠 Chunking and vectorizing {filename}...")
        store.chunk_and_add_text(raw_text=content, source_name=filename)
        
    print("\n✅ All official knowledge successfully ingested into the AI's memory bank!")

if __name__ == "__main__":
    # Fix the pathing issue just like we did in test_pipeline
    import sys
    sys.path.append(os.path.dirname(os.path.abspath(__file__)))
    load_all_documents()