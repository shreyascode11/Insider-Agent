import streamlit as st
from src.agent import InsidersAgent

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
    with st.spinner("Initializing AI Engine and connecting to Groq..."):
        try:
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
user_query = st.chat_input("Ask a question (e.g., 'What are the duties of the Secretary?')")

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
                # Call your exact LangGraph invoke method
                response = st.session_state.agent.invoke(user_query)
                st.markdown(response)
                
                # Add to history
                st.session_state.chat_history.append({"role": "assistant", "content": response})
            except Exception as e:
                st.error(f"Execution error: {e}")