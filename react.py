from dotenv import load_dotenv
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI
from langchain_tavily import TavilySearch

load_dotenv()  # Load environment variables from .env file

@tool
def triple(num: float) -> float:
    """
    param num: A number to be tripled.
    returns: The triple of the input number.
    """
    return 3 * num

tools = [TavilySearch(max_results=1), triple]

llm = ChatOpenAI(model="gpt-4o-mini", temperature=0).bind_tools(tools)