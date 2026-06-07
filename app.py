import streamlit as st
from src.agent import InsidersAgent
from ingest import load_all_documents  

# --- Page Configuration ---
st.set_page_config(
    page_title="Insiders Club AI",
    page_icon="🚀",
    layout="centered"
)

st.title("🚀 Insiders Club Orchestrator")
st.markdown("Welcome to the official AI Assistant for the Insiders Club. Ask me about roles, deadlines, and club policies!")

# --- Initialize Agent and Session State ---
# We store the agent in session state so it doesn't reload on every button click
if "agent" not in st.session_state:
    with st.spinner("Initializing AI Engine and building memory bank..."):
        try:
            # 1. Force the cloud server to ingest the raw text files and build a fresh database
            load_all_documents()
            
            # 2. Boot up the agent
            st.session_state.agent = InsidersAgent()
            st.success("System Online.")
        except Exception as e:
            st.error(f"Failed to load agent: {e}")
            st.stop()

# We store the chat history in session state to display it on the screen
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# --- Render Chat History ---
for msg in st.session_state.chat_history:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# --- Chat Input Box ---
user_query = st.chat_input("Ask a question (e.g., 'Who is the President?')")

if user_query:
    # 1. Display user message immediately
    with st.chat_message("user"):
        st.markdown(user_query)
    
    # 2. Add to history
    st.session_state.chat_history.append({"role": "user", "content": user_query})
    
    # 3. Process with LangGraph Agent
    with st.chat_message("assistant"):
        with st.spinner("Searching club documents..."):
            try:
                # CRITICAL CHANGE: Pass the full chat history list to maintain memory
                response = st.session_state.agent.invoke(st.session_state.chat_history)
                st.markdown(response)
                
                # Add to history
                st.session_state.chat_history.append({"role": "assistant", "content": response})
            except Exception as e:
                st.error(f"Execution error: {e}")