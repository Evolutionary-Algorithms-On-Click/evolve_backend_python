# main.py

from fastapi import FastAPI
from typing import Optional
from pydantic import BaseModel

app = FastAPI()

class rootRes(BaseModel):
    message: str
    description: Optional[str] = None

@app.get("/", response_model=rootRes)
async def root():
    return rootRes(message="Hello World", description="This is a FastAPI example")