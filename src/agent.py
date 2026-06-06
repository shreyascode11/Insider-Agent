import os
from typing import Annotated, Sequence, TypedDict
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage, SystemMessage
from langchain_groq import ChatGroq
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode

from .config import Config
from .tools import AGENT_TOOLS

class AgentState(TypedDict):
    """
    The global state object shared across all nodes in the graph.
    Tracks the conversational history using append-only message semantics.
    """
    messages: Annotated[Sequence[BaseMessage], add_messages]

class InsidersAgent:
    """Orchestrates the LangGraph execution flow for the Insiders Club Assistant."""
    
    def __init__(self):
        # Initializing Groq llama3 model
        if not Config.GROQ_API_KEY:
            raise ValueError("❌ GROQ_API_KEY is missing from your configuration or .env file.")
            
        self.model = ChatGroq(
            groq_api_key=Config.GROQ_API_KEY,
            model_name="llama-3.3-70b-versatile",
            temperature=0.1  # Very low temperature to prevent hallucinations
        )
        
        # Binding the custom RAG tools directly to the Groq model
        self.model_with_tools = self.model.bind_tools(AGENT_TOOLS)
        
        # Compiling the graph workflow architecture
        self.graph = self._build_workflow()

    def _call_model(self, state: AgentState) -> dict:
        """Node: Pass the active conversation state messages to the LLM engine."""
        print("[Graph Node: LLM Thinking] Evaluating user query...")
        messages = state["messages"]
        
        from langchain_core.messages import SystemMessage
        
        # Inject an authoritative system context instruction if it's not already there
        if not messages or getattr(messages[0], "type", "") != "system":
            system_instruction = (
                "You are the official AI Assistant for the Insiders Club. Your job is to answer member questions "
                "with absolute accuracy. You have access to specialized tools to search official documents. "
                "Always look up documents before answering policy or eligibility questions. "
                "If the information is not present in the documents, state clearly that you do not know. "
                "CRITICAL: You are ONLY allowed to use the tools explicitly provided to you. "
                "Do NOT attempt to use hallucinated tools like 'brave_search'. "
                "Once you have found the answer, output the final answer immediately and DO NOT call any more tools."
            )
            messages = [SystemMessage(content=system_instruction)] + list(messages)
            
        response = self.model_with_tools.invoke(messages)
        return {"messages": [response]}

    def _should_continue(self, state: AgentState) -> str:
        """Conditional Edge: Inspects the last message to route to a tool or terminate."""
        last_message = state["messages"][-1]
        
        # If the LLM requested a function/tool execution, route to the tools runner node
        if hasattr(last_message, "tool_calls") and last_message.tool_calls:
            print(f"🔄 [Graph Edge: Routing] Model requested tool call: {last_message.tool_calls[0]['name']}")
            return "execute_tools"
            
        print("🏁 [Graph Edge: Routing] Complete execution. Returning response to user.")
        return END

    def _build_workflow(self):
        """Constructs and compiles the multi-node state graph machine layout."""
        workflow = StateGraph(AgentState)

        # Defining structural computing nodes
        workflow.add_node("agent_brain", self._call_model)
        
        # Prebuilt ToolNode handles automatic extraction and firing of bound functions
        tool_node = ToolNode(AGENT_TOOLS)
        workflow.add_node("execute_tools", tool_node)

        # Set execution entry point
        workflow.add_edge(START, "agent_brain")

        # Establish conditional decision pathways after the model thinks
        workflow.add_conditional_edges(
            "agent_brain",
            self._should_continue,
            {
                "execute_tools": "execute_tools",
                END: END
            }
        )

        # Loop back tool outputs into the brain for structural comprehension
        workflow.add_edge("execute_tools", "agent_brain")

        # Compile graph structure into an executable runner module
        return workflow.compile()

    def invoke(self, query: str) -> str:
        """Public execution method wrapper to invoke the compiled agent engine."""
        system_instruction = SystemMessage(
            content=(
                "You are a strict, precise QA Assistant for the Insiders Club.\n"
                "Your single goal is to answer the user's prompt directly.\n\n"
                "RULES:\n"
                "1. Look up information using your tool.\n"
                "2. As soon as a tool response contains the direct answer to the user's question, "
                "STOP making tool calls immediately. Do not look up secondary or mentioned topics.\n"
                "3. Formulate your final response using the retrieved information and answer the user."
            )
        )
        
        inputs = {
            "messages": [
                system_instruction,
                HumanMessage(content=query)
            ]
        }
        
        output_state = self.graph.invoke(inputs)
        final_message = output_state["messages"][-1]
        return final_message.content