from src.agent import InsidersAgent
agent = InsidersAgent()
for step in agent.graph.stream({"messages": [("user", "When is the exact deadline to apply for the club roles?")]}, stream_mode="values"):
    print(step["messages"][-1])
