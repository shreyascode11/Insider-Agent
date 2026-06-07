from langchain_core.tools import tool
from src.vector_store import ClubVectorStore
from langchain_community.embeddings import HuggingFaceEmbeddings

from src.config import Config

_vector_store = None

def get_vector_store():
    global _vector_store
    if _vector_store is None:
        embeddings = HuggingFaceEmbeddings(model_name=Config.EMBEDDING_MODEL_NAME)
        _vector_store = ClubVectorStore(embedding_model=embeddings)
    return _vector_store

@tool
def search_club_policies(query: str) -> str:
    """
    Use this tool to search the official Insiders Club documentation.
    Pass in a specific natural language query to find rules, deadlines, roles, or responsibilities.
    """
    print(f"\n🛠️ [Agent Tool Execution] Searching internal docs for: '{query}'")
    
    vs = get_vector_store()
    results = vs.similarity_search(query, k=5)
    
    if not results:
        print("⚠️ [Tool DB Status] No matching text blocks found in ChromaDB.")
        return "No relevant information found in the official club documents regarding this query."
    
    # DEBUG PRINT: Let's see what the database is actually giving the LLM
    print(f"📖 [Tool DB Status] Found {len(results)} relevant text block(s):")
    for i, doc in enumerate(results):
        print(f"   -> Chunk {i+1} (Source: {doc.metadata.get('source')}): {doc.page_content.strip()[:120]}...")

    formatted_results = "\n\n".join([
        f"[Source: {doc.metadata.get('source', 'Unknown')}]\n{doc.page_content}" 
        for doc in results
    ])
    
    return formatted_results

AGENT_TOOLS = [search_club_policies]