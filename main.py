import os
from dotenv import load_dotenv
from langchain.agents import create_agent
from langchain.tools import tool
from langchain.messages import HumanMessage, SystemMessage
from langchain_ollama import ChatOllama
from requests import Response
from langfuse.langchain import CallbackHandler
from langfuse import get_client
from langchain_tavily import TavilySearch
from pydantic import BaseModel, Field


class Source(BaseModel):
    '''Schema for a source used by the agent'''
    url: str = Field(description="The url of the source")

class AgentResponse(BaseModel):
    '''Schema for the response of the agent'''
    answer: str = Field(description="The answer to the question")
    sources: list[Source] = Field(description="The sources used to answer the question")







load_dotenv()
# tavily=TavilyClient()

langfuse = get_client()
langfuse_handler = CallbackHandler()

# @tool
# def search(query: str) -> str:
#     '''
#     Search the web for the query
#     Args:
#         query: The query to search for
#     Returns:
#         The search results
#     '''
#     print(f"Searching the web for: {query}")
#     return tavily.search(query=query)

llm=ChatOllama(model="gemma4:12b")
tools=[TavilySearch(max_results=3)]
agent=create_agent(model=llm, tools=tools, response_format=AgentResponse)

def main():

    response=agent.invoke({"messages": [
        SystemMessage(content="You are a helpful assistant. Always use the search tool to look up information before answering any question."),
        HumanMessage(content="Search 3 jobs openings for AI engineer using ai agents in india with 1 year of experience")
    ]},
    config={"callbacks": [langfuse_handler]}
    )
    print(response["messages"][-1].content)
    langfuse.flush()

if __name__ == "__main__":
    main()
