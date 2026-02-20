import os
from dotenv import load_dotenv

# vector db 
import chromadb
from chromadb.utils import embedding_functions

# langchain related
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_community.tools import DuckDuckGoSearchResults
from langchain.agents import create_agent 
from langchain.agents.middleware import ModelRetryMiddleware, ModelCallLimitMiddleware
from langchain_core.rate_limiters import InMemoryRateLimiter 

load_dotenv()

# defining path for persistence
chroma_client = chromadb.PersistentClient(path="./research_memory")

# defining brain for DB
# links embedding function
google_ef = embedding_functions.GoogleGenerativeAiEmbeddingFunction(
    api_key=os.getenv("GOOGLE_API_KEY")
)

# every time you add or query, it calls Google's API to do the math
# saving what it learns
collection = chroma_client.get_or_create_collection(
    name="research_vault",
    embedding_function=google_ef # 🚨 Link established here
)


rate_limiter = InMemoryRateLimiter(
    requests_per_second=0.1, 
    check_every_n_seconds=0.1, 
    max_bucket_size=10
)

# setup the LLM
primary_llm = ChatGoogleGenerativeAI(model="gemini-2.0-flash", temperature=0, rate_limiter=rate_limiter)

fallback_llm = ChatGoogleGenerativeAI(model="gemini-2.0-flash-lite", temperature = 0)

resilient_llm = primary_llm.with_fallbacks([fallback_llm])

# setup Tools
tools = [DuckDuckGoSearchResults()]

# create the Agent using the v1.0 standard
# This pre-built function now handles the loop and execution logic
research_agent = create_agent(
    model=resilient_llm,
    tools=tools,
    system_prompt="You are a helpful research assistant. When asked for a specific number of items"
                  "(e.g. TOP 3) you MUST find and list exactly that many. do not provide vague lists."
                  "Always synthesize information from multiple search results to ensure accuracy."
    ,
    debug=True,
    # middleware to catch 429 (req limit error) and retry automatically
    middleware=[
        # Stop the agent after 5 model calls
        ModelCallLimitMiddleware(run_limit=5, exit_behavior="end"),
        ModelRetryMiddleware(max_retries=3, initial_delay=2.0, backoff_factor=2.0)
    ]
)

async def run_research_agent(query: str):
    try:
        # v1.0 uses a messages-based invoke
        result = await research_agent.ainvoke({
            "messages": [{"role": "user", "content": query}]
        })
        # Extract the final AI message content
        return result["messages"][-1].content
    except Exception as e:
        return f"Error running agent: {str(e)}"