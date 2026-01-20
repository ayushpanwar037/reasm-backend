from fastapi import FastAPI, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
import PyPDF2
import io
from agent import reasm_agent

app = FastAPI()

# Important for Vercel/Railway communication
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.post("/analyze")
async def analyze_resume(file: UploadFile = File(...), jd: str = Form(...)):
    # 1. Extract text from PDF
    pdf_reader = PyPDF2.PdfReader(io.BytesIO(await file.read()))
    resume_text = ""
    for page in pdf_reader.pages:
        resume_text += page.extract_text()

    # 2. Run the Pydantic AI Agent
    # The 'result' will automatically be a REASMAnalysis object
    prompt = f"RESUME:\n{resume_text}\n\nJOB DESCRIPTION:\n{jd}"
    result = await reasm_agent.run(prompt)
    
    return result.data