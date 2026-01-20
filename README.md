# REASM Backend

AI-powered Resume Extraction & Skill Matcher API built with FastAPI and Pydantic AI.

## Environment Variables

Set these in Railway:
- `GOOGLE_API_KEY` - Your Google AI API key for Gemini

## Endpoints

- `POST /analyze` - Analyze a resume against a job description
  - Form data: `file` (PDF), `jd` (string)
