# 🚀 Insiders Club Agentic Orchestrator
> **A high-speed, stateful AI agent for intelligent policy and document retrieval.**

## 📖 Overview
The **Insiders Club Agentic Orchestrator** is an advanced Agentic RAG (Retrieval-Augmented Generation) system built to serve as the official AI Assistant for the Insiders Club. It effortlessly reasons through user queries regarding club policies, roles, and deadlines. By leveraging a stateful agent graph and a local vector database, the agent dynamically fetches factual context to deliver highly accurate, hallucination-free answers. 

## ✨ Key Features
- **Stateful Reasoning:** Built with LangGraph to maintain conversational memory and execute complex tool-calling loops.
- **Lightning-Fast Inference:** Powered by Groq and the massive `llama-3.3-70b-versatile` model for incredibly fast and precise responses.
- **Local Vector Search:** Uses a persistent ChromaDB instance paired with HuggingFace embeddings (`all-MiniLM-L6-v2`) for entirely local, cost-free document indexing.
- **Anti-Hallucination Guardrails:** Strict system prompts and tool constraints guarantee that the AI only answers based on official documentation.
- **Modern Web UI:** An intuitive, chat-based interface built with Streamlit for seamless member interaction.

## 🧠 Architecture
The system operates on an Agentic RAG architecture powered by **LangGraph**:
1. **Query Ingestion:** The user asks a question via the Streamlit UI or Terminal loop.
2. **Stateful Graph Evaluation:** The LangGraph agent receives the query along with the conversation history.
3. **Tool Invocation:** If the agent identifies that it needs factual information, it calls the `search_club_policies` tool.
4. **Vector Retrieval:** The tool converts the query into an embedding, searches the local ChromaDB, and returns the top matching context blocks.
5. **Final Generation:** The LLM synthesizes the retrieved documents to provide a precise answer, terminating the loop immediately to prevent hallucination.

## 🛠️ Tech Stack
- **Core Orchestration:** [LangChain](https://langchain.com/) & [LangGraph](https://langchain.com/langgraph)
- **LLM Inference:** [Groq API](https://groq.com/) (`llama-3.3-70b-versatile`)
- **Vector Database:** [ChromaDB](https://www.trychroma.com/)
- **Embeddings:** HuggingFace (`all-MiniLM-L6-v2`) via `sentence-transformers`
- **Frontend App:** [Streamlit](https://streamlit.io/)
- **Package Management:** `uv` / `pip`

## ⚙️ Prerequisites
Before setting up the project, ensure you have:
- Python 3.13 or higher installed.
- A **Groq API Key**. You can obtain a free API key by signing up at the [Groq Console](https://console.groq.com/).

## 🚀 Installation & Setup
Follow these steps to clone and configure the project locally:

**1. Clone the repository:**
```bash
git clone https://github.com/shreyascode11/Insider-Agent.git
cd Insider-Agent
```

**2. Create and activate a virtual environment (Optional but recommended):**
```bash
python -m venv .venv
source .venv/bin/activate  # On macOS/Linux
# .venv\Scripts\activate   # On Windows
```

**3. Install dependencies:**
```bash
# If using uv (recommended for speed)
uv pip install -r requirements.txt

# Or using standard pip
pip install -r requirements.txt
```

**4. Set up environment variables:**
Create a `.env` file in the root directory and add your Groq API key:
```env
GROQ_API_KEY=your_groq_api_key_here
LLM_PROVIDER=groq
```

## 🎮 Usage
**1. Ingest Data into the Vector Database:**
Before querying the agent, you need to populate the local vector store with the club's raw documents. Place any `.txt` or `.md` files into the `data/raw_docs/` folder, then run:
```bash
python ingest.py
```
*This script automatically reads your files, generates embeddings, and saves them to the persistent local `data/vector_store/` directory.*

**2. Run the Interactive Terminal (Testing):**
You can chat with the agent directly in your terminal to observe its thought process and node evaluation:
```bash
python main.py
```

**3. Launch the Web UI:**
For the complete user experience, start the Streamlit application:
```bash
streamlit run app.py
```

## 📁 Project Structure
```text
Insiders-Agent/
├── .env                  # Environment variables (ignored by git)
├── .gitignore            # Git ignore rules
├── app.py                # Streamlit web interface
├── main.py               # Terminal-based execution loop
├── requirements.txt      # Project dependencies
├── ingest.py             # Script to automatically load and vectorize raw docs
├── pyproject.toml        # Project metadata
├── data/
│   ├── raw_docs/         # Source markdown/PDF files (ignored by git)
│   └── vector_store/     # Persistent ChromaDB files (ignored by git)
└── src/
    ├── __init__.py
    ├── agent.py          # Core LangGraph state machine (InsidersAgent)
    ├── config.py         # Centralized configuration management
    ├── tools.py          # Vector database tool wrappers
    └── vector_store.py   # ChromaDB ingestion and embedding logic
```
