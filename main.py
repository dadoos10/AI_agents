import os
from pydantic import BaseModel
from dotenv import load_dotenv
load_dotenv()
from langchain_core.output_parsers import PydanticOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain.agents import create_tool_calling_agent, AgentExecutor
from langchain_openai import ChatOpenAI
import os
from dotenv import load_dotenv
from tools import search_tool, wiki_tool, save_tool

load_dotenv()
# class of the research response, which will be used to store the response from the LLM.
# base model ensures that the response is in the correct format.
class ResearchResponse(BaseModel):
    topic: str
    summary: str
    sources: list[str]
    tool_used: str

# create a prompt for the LLM to generate a research response.
parser = PydanticOutputParser(pydantic_object=ResearchResponse)
format_instructions = parser.get_format_instructions()

prompt = ChatPromptTemplate.from_messages(
    [
        ("system",
        """
        You are a research assistant that will help generate a research paper answer the user query and use the necessary tools. 
        Wrap the output in the following format and provide no other text:\n{format_instructions}
        provide a link to the source of the information in the sources field.
        """,
        ),
        ("placeholder", "{chat_history}"),
        ("human", "{query}"),
        ("placeholder", "{agent_scratchpad}"),
    ]
).partial(
    format_instructions=format_instructions,
)

# now we will create a prompt template for the LLM to generate a research response.


llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0, api_key=os.environ.get("OPENAI_API_KEY"))
response = llm.invoke("What is the capital of Israel?")
# print(response)
tools = [search_tool, wiki_tool, save_tool]
agent = create_tool_calling_agent(llm = llm, tools = tools, prompt = prompt)
# verbose is set to True to print the process  of the LLM.
agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True)

user_query = input("How can I help you with your research? ")
raw_response = agent_executor.invoke({"query": user_query})

print(raw_response)

try:
    structured_response = parser.parse(raw_response["output"])
    print("Structured Response:")
    print(f"topic: {structured_response.topic}")
    print(f"summary: {structured_response.summary}")
    print(f"sources: {structured_response.sources}")
    print(f"tools_used: {structured_response.tool_used}")
except Exception as e:
    print("Error parsing response:", e)
    print("Raw response:", raw_response["output"])
