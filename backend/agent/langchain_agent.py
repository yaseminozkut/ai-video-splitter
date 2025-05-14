from langchain_ollama import ChatOllama
from langchain.agents import initialize_agent, AgentType
from langchain_core.tools import Tool

# Dummy object detector tool
def fake_object_detector(input_str: str) -> str:
    print(f"👀 [object_detector called with] {input_str}")
    # Simulate that "dog" appears at 32s
    return "✅ Object 'dog' found at 32 seconds in demo.mp4"

# Dummy video splitter tool
def fake_video_splitter(input_str: str) -> str:
    print(f"✂️ [video_splitter called with] {input_str}")
    return f"✅ Video split simulated at {input_str.split('::')[1].strip()} seconds."

def create_agent():
    PREFIX = """You are a helpful assistant that uses tools to solve user problems.

    When you are finished, always say:
    Final Answer: [your final answer here]
    """

    # Use the chat-style Ollama interface
    llm = ChatOllama(model="mistral", temperature=0)

    # Define tools
    tools = [
        Tool(
            name="object_detector",
            func=fake_object_detector,
            description="Detects when an object appears in a video. Input format: 'object_name :: video_path'"
        ),
        Tool(
            name="video_splitter",
            func=fake_video_splitter,
            description="Splits the video at a given timestamp. Input format: 'video_path :: timestamp_in_seconds'"
        ),
    ]

    agent = initialize_agent(
        tools=tools,
        llm=llm,
        agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
        verbose=True,
        handle_parsing_errors=True,
        agent_kwargs={"prefix": PREFIX}
    )

    return agent

# Run a prompt
if __name__ == "__main__":
    agent = create_agent()
    print("🎬 Type a prompt or 'exit' to quit.")
    while True:
        prompt = input("🧠 Prompt: ")
        if prompt.lower() in ("exit", "quit"):
            break
        response = agent.run(prompt)
        print(f"\n🎯 Final Output: {response}")