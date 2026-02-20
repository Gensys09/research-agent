from fastapi import FastAPI, HTTPException
from schemas import ResearchRequest, ResearchResponse
from agent.agent import run_research_agent

app = FastAPI(title="AI Research Assistant")

@app.post("/research", response_model=ResearchResponse)
async def research_topic(request: ResearchRequest):
    """
    Endpoint that takes a query, runs the AI agent to    search web,
    and returns a summarised answer
    """
    
    print(f"Received query: {request.query}")
    
    # run agent logic
    ai_answer = await run_research_agent(request.query)
    
    # for now, mocking source list (instead of parsing)
    return ResearchResponse(
        query = request.query,
        answer = ai_answer,
        sources = ["DuckDuckGo Search"]
    )
    
@app.get("/health")
def health_check():
    return {"status" : "running"}