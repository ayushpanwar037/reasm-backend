from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import PyPDF2
import io
from agent import reasm_agent

app = FastAPI(
    title="REASM API",
    description="AI-powered Resume Extraction & Skill Matcher"
)

# Important for Vercel/Railway communication
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
async def root():
    return {"status": "ok", "message": "REASM API is running"}

@app.post("/analyze")
async def analyze_resume(file: UploadFile = File(...), jd: str = Form(...)):
    try:
        # 1. Extract text from PDF
        pdf_reader = PyPDF2.PdfReader(io.BytesIO(await file.read()))
        resume_text = ""
        for page in pdf_reader.pages:
            resume_text += page.extract_text() or ""

        if not resume_text.strip():
            raise HTTPException(status_code=400, detail="Could not extract text from PDF")

        # 2. Run the Pydantic AI Agent
        # The 'result' will automatically be a REASMAnalysis object
        prompt = f"RESUME:\n{resume_text}\n\nJOB DESCRIPTION:\n{jd}"
        result = await reasm_agent.run(prompt)
        
        # Return the output (new API uses .output instead of .data)
        return result.output
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Analysis failed: {str(e)}")