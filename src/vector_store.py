import os
import uuid
from typing import List, Dict, Any
import chromadb
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from .config import Config

class ClubVectorStore:
    """Manages document chunking, embedding generation, and vector storage using ChromaDB."""
    
    def __init__(self, embedding_model: Embeddings):
        """
        Initializes the persistent ChromaDB collection and the embedding model.
        """
        self.embedding_model = embedding_model
        self.collection_name = "insiders_club_knowledge"
        self.persist_directory = Config.VECTOR_STORE_PATH
        self.client = None
        self.collection = None
        
        self._initialize_store()

    def _initialize_store(self) -> None:
        """Initializes the persistent ChromaDB client and fetches/creates the collection."""
        try:
            import sys
            if sys.platform.startswith("linux"):
                # Use in-memory client on Streamlit Cloud to bypass all SQLite/Rust filesystem restrictions
                self.client = chromadb.EphemeralClient()
            else:
                os.makedirs(self.persist_directory, exist_ok=True)
                # Create a persistent local database client
                self.client = chromadb.PersistentClient(path=self.persist_directory)
            
            # Fetch or create the specific collection for club data
            self.collection = self.client.get_or_create_collection(
                name=self.collection_name,
                metadata={"description": "Grounded knowledge base for Insiders Club operations"}
            )
            print(f"✅ Vector store successfully initialized at: {self.persist_directory}")
            print(f"📊 Current unique chunks in store: {self.collection.count()}")
            
        except Exception as e:
            print(f"❌ Error during vector store initialization: {e}")
            raise

    def chunk_and_add_text(self, raw_text: str, source_name: str) -> None:
        """
        Takes raw text from a document, splits it cleanly into overlapping chunks,
        generates embeddings, and saves them to the store.
        """
        # Using a specialized text splitter to preserve paragraph context
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=500,
            chunk_overlap=50,
            length_function=len,
            separators=["\n\n", "\n", " ", ""]
        )
        
        # Split text into segments
        chunks = splitter.split_text(raw_text)
        
        # Convert raw strings to structured LangChain Document objects
        documents = [
            Document(page_content=chunk, metadata={"source": source_name})
            for chunk in chunks
        ]
        
        self.add_documents(documents)

    def add_documents(self, documents: List[Document]) -> None:
        """Extracts content, computes embeddings, and updates ChromaDB."""
        if not documents:
            print("⚠️ No documents provided to add.")
            return

        print(f"🔄 Processing {len(documents)} document chunks...")
        
        ids: List[str] = []
        texts: List[str] = []
        metadatas: List[Dict[str, Any]] = []
        
        for i, doc in enumerate(documents):
            unique_id = f"chunk_{uuid.uuid4().hex[:8]}_{i}"
            ids.append(unique_id)
            texts.append(doc.page_content)
            
            meta = dict(doc.metadata) if doc.metadata else {}
            meta["source"] = meta.get("source", "unknown_source")
            meta["char_length"] = len(doc.page_content)
            metadatas.append(meta)

        print("🧠 Generating local text embeddings vectors...")
        try:
            embeddings_list = self.embedding_model.embed_documents(texts)
        except Exception as e:
            print(f"❌ Failed to generate embeddings: {e}")
            raise

        try:
            self.collection.add(
                ids=ids,
                embeddings=embeddings_list,
                metadatas=metadatas,
                documents=texts
            )
            print(f"✨ Successfully ingested {len(documents)} blocks into vector database.")
            print(f"📈 Total combined chunks now: {self.collection.count()}")
        except Exception as e:
            print(f"❌ Database write operation failed: {e}")
            raise

    def similarity_search(self, query: str, k: int = 3) -> List[Document]:
        """Queries the vector store for the most contextually relevant segments."""
        try:
            query_embedding = self.embedding_model.embed_query(query)
            
            results = self.collection.query(
                query_embeddings=[query_embedding],
                n_results=k
            )
            
            retrieved_docs = []
            if results and results["documents"] and len(results["documents"]) > 0:
                for text, meta in zip(results["documents"][0], results["metadatas"][0]):
                    retrieved_docs.append(Document(page_content=text, metadata=meta))
                    
            return retrieved_docs
            
        except Exception as e:
            print(f"❌ Error occurred during similarity vector search: {e}")
            return []