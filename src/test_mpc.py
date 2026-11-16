import asyncio
from fastmcp import Client

ASK_CASES = [
    "What was Microsoft's depreciation and amortization expense in FY2023?",
    "",
]

OTHER_CASES = [
    ("get_financial_metric", {"company": "3M", "metric": "capex", "fiscal_year": 2018}),
    ("get_financial_metric", {"company": "MICROSOFT", "metric": "depreciation_amortization", "fiscal_year": 2023}),
    ("get_financial_metric", {"company": "FERRARI", "metric": "revenue", "fiscal_year": 2022}),
    ("search_filings", {"query": "depreciation and amortization in the cash flow statement",
                        "company": "MICROSOFT", "fiscal_year": 2023, "k": 3}),
    ("calculator", {"expression": "(4258 + 56 + 4947) / 10936"}),
    ("calculator", {"expression": "1 / 0"}),
    ("calculator", {"expression": "import os"}),
]

def show(name, args, result):
    print(f"== {name} {args}")
    if result.is_error:
        print("ERROR:", result.content[0].text)
    else:
        print("OK:", result.data)
    print("-" * 80)

async def main():
    async with Client("http://127.0.0.1:8000/mcp") as client:
        tools = await client.list_tools()
        print([t.name for t in tools])

        for q in ASK_CASES:
            result = await client.call_tool("ask_finrag", {"question": q}, raise_on_error=False)
            show("ask_finrag", q, result)

        for tool, args in OTHER_CASES:
            result = await client.call_tool(tool,args,raise_on_error=False)
            show(tool, args, result)

asyncio.run(main())