from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import PyPDF2
import io
import re
from agent import reasm_agent

app = FastAPI(
    title="REASM API",
    description="AI-powered Resume Extraction & Skill Matcher with Semantic Matching"
)

# CORS for frontend communication
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


def extract_pdf_text(file_bytes: bytes) -> str:
    """Extract text from PDF with better formatting"""
    try:
        pdf_reader = PyPDF2.PdfReader(io.BytesIO(file_bytes))
        text_parts = []
        
        for page in pdf_reader.pages:
            page_text = page.extract_text()
            if page_text:
                # Clean up the text
                page_text = re.sub(r'\s+', ' ', page_text)  # Normalize whitespace
                text_parts.append(page_text.strip())
        
        return "\n\n".join(text_parts)
    except Exception as e:
        raise ValueError(f"PDF extraction failed: {str(e)}")


def clean_job_description(jd: str) -> str:
    """Clean and format job description for better parsing"""
    # Remove excessive whitespace
    jd = re.sub(r'\s+', ' ', jd)
    # Preserve list formatting
    jd = re.sub(r'[•·▪▸►]\s*', '\n• ', jd)
    jd = re.sub(r'[-–—]\s+', '\n- ', jd)
    return jd.strip()


@app.get("/")
async def root():
    return {
        "status": "ok", 
        "message": "REASM API is running",
        "version": "2.0",
        "model": "gemini-2.5-flash"
    }


@app.get("/health")
async def health():
    return {"status": "healthy"}


@app.post("/analyze")
async def analyze_resume(
    file: UploadFile = File(...), 
    jd: str = Form(...)
):
    """
    Analyze a resume against a job description.
    
    - file: PDF resume file
    - jd: Job description text
    
    Returns structured analysis with skill matching and verdict.
    """
    try:
        # 1. Validate file
        if not file.filename.lower().endswith('.pdf'):
            raise HTTPException(status_code=400, detail="Only PDF files are supported")
        
        # 2. Extract text from PDF
        file_bytes = await file.read()
        resume_text = extract_pdf_text(file_bytes)
        
        if not resume_text or len(resume_text.strip()) < 50:
            raise HTTPException(
                status_code=400, 
                detail="Could not extract sufficient text from PDF. Please ensure it's not a scanned image."
            )

        # 3. Clean job description
        clean_jd = clean_job_description(jd)
        
        if len(clean_jd.strip()) < 20:
            raise HTTPException(
                status_code=400,
                detail="Job description is too short. Please provide more details."
            )

        # 4. Build the analysis prompt
        prompt = f"""
## RESUME CONTENT:
{resume_text}

---

## JOB DESCRIPTION:
{clean_jd}

---

## TASK:
Analyze this resume against the job description following your instructions.

1. First, find and extract the candidate's ACTUAL NAME from the resume (usually at the top)
2. Extract ALL required skills/technologies from the job description
3. For each skill, check the resume and determine: Matched, Partial, or Missing
4. Provide specific justifications referencing actual resume content
5. Calculate an overall match score (0-100)
6. Determine verdict: STRONG HIRE, POTENTIAL MATCH, or SKILL GAP DETECTED
7. Provide actionable improvement tips

Be thorough and accurate. The candidate's career depends on this analysis.
"""

        # 5. Run AI analysis
        result = await reasm_agent.run(prompt)
        
        # 6. Return the structured output
        return result.output
        
    except HTTPException:
        raise
    except Exception as e:
        error_msg = str(e)
        
        # Provide helpful error messages
        if "429" in error_msg or "quota" in error_msg.lower():
            raise HTTPException(
                status_code=429,
                detail="AI rate limit reached. Please wait 30 seconds and try again."
            )
        elif "api_key" in error_msg.lower():
            raise HTTPException(
                status_code=500,
                detail="API configuration error. Please contact support."
            )
        else:
            raise HTTPException(
                status_code=500, 
                detail=f"Analysis failed: {error_msg}"
            )