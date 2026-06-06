import sys
from src.agent import InsidersAgent

def main():
    print("="*60)
    print("🚀 Booting up Insiders Club Agentic Orchestrator...")
    print("="*60)

    try:
        # 1. This creates the instance and saves it to lowercase 'agent'
        agent = InsidersAgent() 
        print("✅ System Online: Agent successfully compiled and connected to Groq.")
        print("💡 (Type 'exit' or 'quit' to close the terminal.)\n")
    except Exception as e:
        print(f"\n❌ Failed to initialize agent: {e}")
        sys.exit(1)

    # 2. Interactive Terminal Execution Loop
    while True:
        try:
            user_input = input("\n👤 You: ")
            
            if user_input.lower() in ['quit', 'exit', 'q']:
                print("\n👋 Shutting down operations... Good luck on your submission!")
                break

            if not user_input.strip():
                continue

            print("🤖 Agent is processing... (Evaluating Nodes)")
            
            # CRUCIAL FIX HERE: Must be lowercase 'agent', NOT 'InsidersAgent'
            response = agent.invoke(user_input) 

            print("\n✨ Insiders Agent:")
            print(f"{response}\n")
            print("-" * 60)

        except KeyboardInterrupt:
            print("\n\n👋 Force quit detected. Shutting down system...")
            break
        except Exception as e:
            print(f"\n❌ Execution Error during graph traversal: {e}")

if __name__ == "__main__":
    main()