from dotenv import load_dotenv

load_dotenv()

from langchain.chat_models import init_chat_model
from langchain.tools import tool
from langchain_core.messages import HumanMessage, SystemMessage, ToolMessage
from langsmith import traceable

MAX_ITERATIONS = 10
MODEL = 'qwen3.5:2b'

@tool
def get_product_price(product: str) -> float:
    """Lookup the price of a product in the catalog."""
    print(f"    >> Looking up price for: {product}")

    prices = {
        "laptop": 999,
        "keyboard": 49,
        "smartphone": 499,
        "headphones": 199,
        "Lenovo ThinkPad T14 G6 Black": 1200
    }
    
    prices_lower = {k.lower(): v for k, v in prices.items()}
    return prices_lower.get(product.lower(), 0)

@tool
def apply_discount(price: float, discount_tier: str) -> float:
    """Apply a discount tier to the price. Available discount tiers are: bronze, silver, gold."""
    print(f"    >> Applying discount: {discount_tier} to price: {price}")
    discount_percentages = {
        "bronze": 10,
        "silver": 20,
        "gold": 30
    }
    discount = discount_percentages.get(discount_tier.lower(), 0)
    return round(price * (1 - discount / 100), 2)

@traceable(name="langchain agent loop with tool calling")
def run_agent(question: str):
    tools = [get_product_price, apply_discount]
    tools_dict = {tool.name: tool for tool in tools}
    llm = init_chat_model(f'ollama:{MODEL}', temperature=0)
    llm_with_tools = llm.bind_tools(tools)
    print(f"Question: {question}")
    print("=" * 60)

    messages = [
        SystemMessage(
            content=(
                "You are a helpful shopping assistant."
                "You have access to the following tools:"
                "- get_product_price(product_name: str) -> float: Lookup the price of a product in the catalog."
                "- apply_discount(price: float, discount_tier: str) -> float: Apply a discount tier to the price. Available discount tiers are: bronze, silver, gold."
                "Use the tools to answer the user's question. If you need to call a tool, use the following format:"
                "ToolName(arg1=value1, arg2=value2)"
                "STRICT RULES - you must follow this exactly:\n"
                "1. NEVER guess or assume any product price.\n"
                "2. ALWAYS use the get_product_price tool to lookup product prices.\n"
                "3. ALWAYS use the apply_discount tool to apply discounts.\n"
                "4. ONLY call apply discount after you have the price from get_product_price. "
                "Pass the exact price returned from get_product_price to apply_discount, do NOT pass made up number.\n"
                "5. If the user does not specify a discount tier, ask them which tier to use - do NOT assume a default tier.\n"
            )
        ),
        HumanMessage(content=question)
    ]

    for i in range(1, MAX_ITERATIONS + 1):
        print(f"Iteration {i}:")

        ai_message = llm_with_tools.invoke(messages)

        tool_calls = ai_message.tool_calls

        if not tool_calls:
            print("Final answer:")
            print(ai_message.content)
            print("-" * 60)
            return ai_message.content
        
        tool_call = tool_calls[0]
        tool_name = tool_call.get("name")
        tool_args = tool_call.get("args", {})
        tool_call_id = tool_call.get("id")

        print(f"  >> Tool call: {tool_name} with args: {tool_args}")
        
        tool_to_use = tools_dict.get(tool_name)
        if tool_to_use is None:
            print(f"Error: Tool {tool_name} not found.")
            print("-" * 60)
            return
        
        obseravation = tool_to_use.invoke(tool_args)

        print(f"  >> Tool observation: {obseravation}")

        messages.append(ai_message)
        messages.append(
            ToolMessage(
                content=obseravation,
                tool_call_id=tool_call_id
            )
        )

    print("ERROR: Max iterations reached without a final answer.")
    print("-" * 60)
    return None


if __name__ == "__main__":
    run_agent("What is the price of a laptop \"Lenovo ThinkPad T14 G6 Black\" after applying a gold discount?")