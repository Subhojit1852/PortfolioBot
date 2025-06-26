from fastapi import FastAPI
from pydantic import BaseModel
from connect_memory_with_LLM import qa_chain
# from dotenv import load_dotenv
from fastapi.middleware.cors import CORSMiddleware


# load_dotenv()

app = FastAPI()
origins = ["*"]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class ChatRequest(BaseModel):
    query: str

@app.post("/chat")
def chat(req: ChatRequest):
    result = qa_chain.invoke({"query": req.query})
    return {"response": result["result"]}

