from typing import TypedDict, List
from pydantic import BaseModel, Field

class TavilyResult(TypedDict):
    title: str
    content: str
    url: str

class AgentState(TypedDict):
    user_input: str
    project_type: str
    city: str
    geo_state: str
    legal_info_found: bool
    legal_summary: str
    suggested_websites: List[str]
    project_roadmap: str
    route_decision: str
    tavily_search_results: List[TavilyResult]
    query_type: str

class ProjectLocation(BaseModel):
    project_type: str = Field(...)
    city: str = Field(...)
    geo_state: str = Field(...)

class QueryClassifier(BaseModel):
    query_type: str = Field(...)