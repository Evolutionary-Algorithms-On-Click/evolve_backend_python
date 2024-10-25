# main.py

from fastapi import FastAPI
from controller import apiRouter    

app = FastAPI()

app.include_router(apiRouter)