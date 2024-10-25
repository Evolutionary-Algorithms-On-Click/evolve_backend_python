from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from controller import apiRouter
import os

from dotenv import load_dotenv
load_dotenv()

os.makedirs("plots/", exist_ok=True)


app = FastAPI(
    title="Genetic Algorithm",
    description="Genetic Algorithm API",
    version="0.1.0",
    docs_url="/docs",
    redoc_url=None
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



