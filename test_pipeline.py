import os
import sys

# Crucial Fix: Add the current root directory to the Python search path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from langchain_community.embeddings import HuggingFaceEmbeddings
from src.vector_store import ClubVectorStore

def run_test():
    print("🚀 Initializing Test Pipeline Components...")
    
    # 1. Initialize local embedding framework 
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    
    # 2. Instantiate our vector store manager
    store = ClubVectorStore(embedding_model=embeddings)
    
    # 3. Add mock data based on your specific requirements text
    sample_club_data = """
    Insiders Club Roles Opening Notice:
    - Secretary (1 position): Maintain documentation, track activities, manage official emails.
    - Web Development Associate (2 positions): Assist members with frontend/backend projects. Required: React.js, Full-Stack.
    - AI/ML Associate (2 positions): Assist with ML concepts and projects. Required: Machine Learning, LLMs.
    
    The deadline for submitting application forms is June 10, 2026 at 11:59 PM. Evaluation is marked out of 50 total points.
    """
    
    print("\n📥 Ingesting sample text data into local storage...")
    store.chunk_and_add_text(sample_club_data, source_name="roles_and_deadlines_2026")
    
    # 4. Perform a validation query search
    print("\n🔍 Simulating an Agent Query search...")
    query = "When is the absolute last date to apply for positions?"
    results = store.similarity_search(query, k=1)
    
    print("\n=== Best Vector Match Found ===")
    for doc in results:
        print(f"Source Document: {doc.metadata['source']}")
        print(f"Content Context:\n{doc.page_content.strip()}")

if __name__ == "__main__":
    run_test()