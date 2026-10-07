from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from verify_agent import run_verifact_agent

app = FastAPI(
    title="Agentic VeriFact API",
    description="Explainable News Verification combining Deep Learning (LSTM) with Live Retrieval (Agentic RAG)",
    version="1.0.0"
)

# Request schema
class NewsItem(BaseModel):
    text: str

# Health check endpoint
@app.get("/health")
def health_check():
    return {"status": "healthy", "service": "Agentic VeriFact Engine"}

# Primary verification endpoint
@app.post("/api/verify")
def verify_claim(item: NewsItem):
    if not item.text.strip():
        raise HTTPException(status_code=400, detail="Text field cannot be empty.")
    
    result = run_verifact_agent(item.text)
    return result

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app:app", host="127.0.0.1", port=8000, reload=True)