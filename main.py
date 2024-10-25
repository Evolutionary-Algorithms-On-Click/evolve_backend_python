from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
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

app.mount("/api/plots", StaticFiles(directory="plots"), name="plots")

app.include_router(apiRouter)



