import os
from dotenv import load_dotenv

load_dotenv()

class Config:
    """Centralized configuration management for the Insiders Agent."""
    
    # --- Directory Paths ---
    # Automatically resolve the root directory (insiders_agent/)
    BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    
    import sys
    # Use /tmp for Streamlit Cloud (Linux) to avoid SQLite locking and read-only filesystem issues
    if sys.platform.startswith("linux"):
        VECTOR_STORE_PATH = "/tmp/insiders_vector_store"
    else:
        VECTOR_STORE_PATH = os.path.join(BASE_DIR, "data", "vector_store")
    RAW_DOCS_PATH = os.path.join(BASE_DIR, "data", "raw_docs")
    
    # --- Model Configuration ---
    # Using HuggingFace 
    EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2" 
    
    LLM_PROVIDER = os.getenv("LLM_PROVIDER", "groq") 
    OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
    GROQ_API_KEY = os.getenv("GROQ_API_KEY") 
    
    @classmethod
    def validate_keys(cls):
        """Ensure the required API keys are present based on the chosen provider."""
        if cls.LLM_PROVIDER == "openai" and not cls.OPENAI_API_KEY:
            print("⚠️ Warning: OPENAI_API_KEY is not set in the .env file.")
        elif cls.LLM_PROVIDER == "groq" and not cls.GROQ_API_KEY:
            print("⚠️ Warning: GROQ_API_KEY is not set in the .env file.")
        else:
            print(f"✅ Configuration loaded. Active Provider: {cls.LLM_PROVIDER.upper()}")

# Run basic validation when the module is imported
Config.validate_keys()