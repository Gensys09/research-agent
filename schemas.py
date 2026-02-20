from pydantic import BaseModel

class ResponseObject(BaseModel):
    title: str
    reason: str
    
class ResearchRequest(BaseModel): 
    query: str
    max_results: int = 3

class ResearchResponse(BaseModel):
    query: str
    answer: list[ResponseObject]
    sources: list[str]