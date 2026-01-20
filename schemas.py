from pydantic import BaseModel, Field
from typing import List

class SkillMatch(BaseModel):
    skill: str
    match_status: str = Field(description="One of: 'Matched', 'Missing', or 'Partial'")
    justification: str = Field(description="Brief reason for this status")

class REASMAnalysis(BaseModel):
    candidate_name: str
    overall_score: float = Field(ge=0, le=100)
    verdict: str = Field(description="Must be 'STRONG HIRE', 'POTENTIAL MATCH', or 'SKILL GAP DETECTED'")
    detailed_matches: List[SkillMatch]
    improvement_tips: List[str]