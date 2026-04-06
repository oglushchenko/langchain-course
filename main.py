from dotenv import load_dotenv
from langchain_core.messages import HumanMessage
from langgraph.graph import END, MessagesState, StateGraph

from nodes import run_agent_reasoning, tool_node

load_dotenv()  # Load environment variables from .env file

AGEN_REASON = "agen_reason"
ACT = "act"
LAST = -1

def should_continue(state: MessagesState) -> str:
    """
    Determine whether to continue reasoning or to act.
    """
    if not state["messages"][LAST].tool_calls:
        return END
    return ACT

flow = StateGraph(MessagesState)

flow.add_node(AGEN_REASON, run_agent_reasoning)
flow.set_entry_point(AGEN_REASON)
flow.add_node(ACT, tool_node)

flow.add_conditional_edges(AGEN_REASON, should_continue, {
    END:END,
    ACT:ACT
})

flow.add_edge(ACT, AGEN_REASON)

app = flow.compile()
app.get_graph().draw_mermaid_png(output_file_path="flow.png")

if __name__ == "__main__":
    print("Hello from langchain-course!")
    res = app.invoke({"messages": [HumanMessage(content="What is the weather in New York? List it and then triple the temperature")]})
    print(res["messages"][LAST].content)
