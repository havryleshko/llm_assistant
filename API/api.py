from fastapi import FastAPI
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"]
)

class Question(BaseModel): # to validate requests
    question: str #str because user asks question in text form

@app.get('/')
async def root(): # async to handle many requests
    return {'message': 'API working'}

@app.post('/ask') #test 
async def ask_q(q: Question):
    return {'answer': q.question}
