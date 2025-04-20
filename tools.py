# this document is for the tools used in the ai agent
from datetime import datetime
from langchain_community.tools import DuckDuckGoSearchRun
from langchain_community.tools import WikipediaQueryRun
from langchain.tools import Tool
from langchain_community.utilities import WikipediaAPIWrapper

#1. search in web tool
search = DuckDuckGoSearchRun()
search_tool = Tool(
    name="search_in_web",
    func=search.run,
    description="Search the web, useful for when you need to answer questions about" \
    " current events. You should ask targeted questions that are likely to have a" \
    " specific answer.",
)

#2. search in wiki tool
wikipedia_api_wrapper = WikipediaAPIWrapper(top_k_results=1, doc_content_chars_max=100, )
wiki_tool = WikipediaQueryRun(api_wrapper=wikipedia_api_wrapper)

def save_to_txt(data: str, filename: str = "research_output.txt"):
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    formatted_text = f"--- Research Output ---\nTimestamp: {timestamp}\n\n{data}\n\n"

    with open(filename, "a", encoding="utf-8") as f:
        f.write(formatted_text)

    return f"Data successfully saved to {filename}"

save_tool = Tool.from_function(
    func=save_to_txt,
    name="save_text_to_file",
    description="save structured research data to a text file. Input should be the data to save.",
)
