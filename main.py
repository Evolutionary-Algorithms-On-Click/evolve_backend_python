from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from controller import apiRouter
import os
import shutil


from dotenv import load_dotenv

load_dotenv()

# shutil.rmtree("plots/")
# shutil.rmtree("population/")
os.makedirs("plots/", exist_ok=True)
os.makedirs("population/", exist_ok=True)


app = FastAPI(
    title="Evolve",
    description="Tool for running Evlutionary Algorithms in Python",
    version="0.1.0",
    docs_url="/docs",
    redoc_url=None,
)

# Allow all origins CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.mount("/api/plots", StaticFiles(directory="plots"), name="plots")
app.mount("/api/population", StaticFiles(directory="population"), name="population")

app.include_router(apiRouter)
