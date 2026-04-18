from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from assistant_engine import AssistantEngine
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI()

# Enable CORS for React frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

engine = AssistantEngine()

class Query(BaseModel):
    text: str

@app.post("/ask")
async def ask_assistant(query: Query):
    try:
        response = engine.process_query(query.text)
        return {"response": response}
    except Exception as e:
        print(f"API Error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    # Changed port to 8001 to avoid WinError 10048
    uvicorn.run(app, host="0.0.0.0", port=8001)
