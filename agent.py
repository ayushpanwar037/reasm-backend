import os
from pydantic_ai import Agent
from schemas import REASMAnalysis
from dotenv import load_dotenv

load_dotenv()

# We define the agent with a specific output_type to ensure structured output
# Using 'google-gla' provider with Gemini model
reasm_agent = Agent(
    'google-gla:gemini-2.0-flash',  # Updated model name
    output_type=REASMAnalysis,       # Changed from result_type to output_type
    system_prompt=(
        "You are REASM, a high-level Technical Recruiter AI. "
        "Analyze the provided Resume against the Job Description."
    )
)