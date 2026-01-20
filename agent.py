import os
from pydantic_ai import Agent
from schemas import REASMAnalysis
from dotenv import load_dotenv

load_dotenv()

# We define the agent with a specific result_type to ensure structured output
reasm_agent = Agent(
    'google-gla:gemini-2.5-flash', 
    result_type=REASMAnalysis,
    system_prompt=(
        "You are REASM, a high-level Technical Recruiter AI. "
        "Analyze the provided Resume against the Job Description."
    )
)