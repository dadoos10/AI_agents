# this document is for the tools used in the ai agent
#1. search in web tool
from langchain_community.tools import DuckDuckGoResultsResults
from langchain.tools import Tool

search = DuckDuckGoResultsResults()
search_tool = Tool(
    name="search_in_web",
    func=search.run,
    description="Search the web, useful for when you need to answer questions about current events. You should ask targeted questions that are likely to have a specific answer.",
)
