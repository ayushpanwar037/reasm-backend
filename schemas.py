from pydantic import BaseModel, Field
from typing import List


class SkillMatch(BaseModel):
    """Individual skill matching result"""
    skill: str = Field(description="The skill or technology being evaluated")
    match_status: str = Field(
        description="One of: 'Matched' (skill found), 'Partial' (related experience), or 'Missing' (not found)"
    )
    justification: str = Field(
        description="Specific reason for this status, referencing actual resume content"
    )


class REASMAnalysis(BaseModel):
    """Complete resume analysis result"""
    candidate_name: str = Field(
        description="The actual name of the candidate extracted from the resume header/top section"
    )
    overall_score: float = Field(
        ge=0, 
        le=100,
        description="Overall match percentage from 0-100 based on skill alignment"
    )
    verdict: str = Field(
        description="Must be one of: 'STRONG HIRE' (>=80), 'POTENTIAL MATCH' (50-79), or 'SKILL GAP DETECTED' (<50)"
    )
    detailed_matches: List[SkillMatch] = Field(
        description="List of all skills from JD with their match status and justification"
    )
    improvement_tips: List[str] = Field(
        description="2-4 actionable suggestions for the candidate to improve their fit for this role"
    )