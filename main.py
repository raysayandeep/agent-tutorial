from dotenv import load_dotenv
from pydantic import BaseModel
from langchain_anthropic import ChatAnthropic
from langchain_aws import ChatBedrock
from langchain_ollama import ChatOllama
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import PydanticOutputParser
from langchain.agents import create_tool_calling_agent, AgentExecutor
from tools import search_tool, custom_math_tool, custom_retriever_tool

load_dotenv()


# class ResearchResponse(BaseModel):
#     topic: str
#     summary: str
#     sources: list[str]
#     tools_used: list[str]
class ResearchResponse(BaseModel):
    answer: str


# llm = ChatAnthropic(model='claude-3-5-sonnet-20241022')
# llm2 = ChatBedrock(model='meta.llama3-8b-instruct-v1:0')
# llm2 = ChatBedrock(model='anthropic.claude-3-5-sonnet-20240620-v1:0')
llm3 = ChatOllama(model='qwen2.5:latest')

parser = PydanticOutputParser(pydantic_object=ResearchResponse)

prompt = ChatPromptTemplate.from_messages(
    [
        ("system",
         """
        You are a research assistant that will help generate a research paper.
        Answer the user query only using the supplied tools.
        Wrap the output in the format and provide no other text\n{format_instructions}
        """,
         ),
        ("placeholder", "{chat_history}"),
        ("human", "{query}"),
        ("placeholder", "{agent_scratchpad}"),
    ]
).partial(format_instructions=parser.get_format_instructions())

tools = [custom_retriever_tool]
agent = create_tool_calling_agent(
    llm=llm3,
    prompt=prompt,
    tools=tools
)

agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=False)
#query = input("Enter your query: ")
#raw_response = agent_executor.invoke({"query": f"{query}"})
#print(raw_response.get('output'))
# try:
#     structured_response = parser.parse(raw_response.get('output'))
#     print(structured_response)
# except Exception as e:
#     print(f"Error parsing response:{e}")
while True:
    query = input("Enter your query:\n")
    if query.lower() in ["exit", "quit", "bye"]:
        print("\nAI Assistant: Goodbye! Have a great day! 👋\n")
        break

    # Process input through the agent
    response = agent_executor.invoke({"query": f"{query}"})

    # Display AI response
    print("\nAI Assistant:\n", response["output"], "\n")