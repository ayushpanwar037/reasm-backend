"""
Pinecone Vector Database Integration for Semantic Skill Matching
Uses embeddings to find similar skills and improve matching accuracy
"""
import os
from typing import List, Dict, Optional
from pinecone import Pinecone, ServerlessSpec
import google.generativeai as genai
from dotenv import load_dotenv

load_dotenv()

# Initialize Pinecone
pc = None
index = None
INDEX_NAME = "reasm-skills"

def init_pinecone():
    """Initialize Pinecone client and index"""
    global pc, index
    
    api_key = os.getenv("PINECONE_API_KEY")
    if not api_key:
        print("Warning: PINECONE_API_KEY not set. Semantic matching disabled.")
        return False
    
    try:
        pc = Pinecone(api_key=api_key)
        
        # Check if index exists, create if not
        existing_indexes = [idx.name for idx in pc.list_indexes()]
        
        if INDEX_NAME not in existing_indexes:
            pc.create_index(
                name=INDEX_NAME,
                dimension=768,  # Gemini embedding dimension
                metric="cosine",
                spec=ServerlessSpec(
                    cloud="aws",
                    region="us-east-1"
                )
            )
        
        index = pc.Index(INDEX_NAME)
        return True
    except Exception as e:
        print(f"Pinecone initialization error: {e}")
        return False


def get_embedding(text: str) -> Optional[List[float]]:
    """Generate embedding using Google's embedding model"""
    try:
        genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))
        result = genai.embed_content(
            model="models/text-embedding-004",
            content=text
        )
        return result['embedding']
    except Exception as e:
        print(f"Embedding error: {e}")
        return None


def store_skill_embeddings(skills: List[str], namespace: str = "jd_skills") -> bool:
    """Store skill embeddings in Pinecone for semantic matching"""
    if not index:
        if not init_pinecone():
            return False
    
    try:
        vectors = []
        for i, skill in enumerate(skills):
            embedding = get_embedding(skill)
            if embedding:
                vectors.append({
                    "id": f"{namespace}_{i}",
                    "values": embedding,
                    "metadata": {"skill": skill, "source": namespace}
                })
        
        if vectors:
            index.upsert(vectors=vectors, namespace=namespace)
        return True
    except Exception as e:
        print(f"Error storing embeddings: {e}")
        return False


def find_similar_skills(
    resume_skill: str, 
    namespace: str = "jd_skills",
    top_k: int = 3,
    threshold: float = 0.7
) -> List[Dict]:
    """
    Find similar skills from JD based on resume skill
    Returns list of similar skills with similarity scores
    """
    if not index:
        if not init_pinecone():
            return []
    
    try:
        # Get embedding for resume skill
        embedding = get_embedding(resume_skill)
        if not embedding:
            return []
        
        # Query Pinecone
        results = index.query(
            vector=embedding,
            top_k=top_k,
            include_metadata=True,
            namespace=namespace
        )
        
        # Filter by threshold and return
        similar_skills = []
        for match in results.matches:
            if match.score >= threshold:
                similar_skills.append({
                    "skill": match.metadata.get("skill", ""),
                    "similarity": round(match.score, 3)
                })
        
        return similar_skills
    except Exception as e:
        print(f"Error finding similar skills: {e}")
        return []


def semantic_skill_match(
    resume_skills: List[str],
    jd_skills: List[str],
    threshold: float = 0.75
) -> Dict[str, Dict]:
    """
    Perform semantic matching between resume skills and JD skills
    Returns a mapping of JD skills to their match status
    """
    # Store JD skills in Pinecone
    store_skill_embeddings(jd_skills, namespace="jd_skills")
    
    # Match each resume skill against JD skills
    matches = {}
    matched_jd_skills = set()
    
    for resume_skill in resume_skills:
        similar = find_similar_skills(resume_skill, namespace="jd_skills", threshold=threshold)
        for s in similar:
            jd_skill = s["skill"]
            if jd_skill not in matched_jd_skills:
                matched_jd_skills.add(jd_skill)
                matches[jd_skill] = {
                    "matched_with": resume_skill,
                    "similarity": s["similarity"],
                    "status": "Matched" if s["similarity"] >= 0.85 else "Partial"
                }
    
    # Mark unmatched JD skills as missing
    for jd_skill in jd_skills:
        if jd_skill not in matches:
            matches[jd_skill] = {
                "matched_with": None,
                "similarity": 0,
                "status": "Missing"
            }
    
    return matches


def cleanup_namespace(namespace: str = "jd_skills"):
    """Clean up vectors in a namespace after analysis"""
    if index:
        try:
            index.delete(delete_all=True, namespace=namespace)
        except Exception as e:
            print(f"Cleanup error: {e}")
