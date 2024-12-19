from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from controllers.runnerController import apiRouter as runnerRouter
from controllers.authController import apiRouter as authRouter
import os
import shutil
from contextlib import asynccontextmanager
from db.dbSession import databaseInstance, initDatabase
from mailer.mailSession import mailerInstance


from dotenv import load_dotenv

load_dotenv()

# shutil.rmtree('plots/')
# shutil.rmtree('population/')
os.makedirs("plots/", exist_ok=True)
os.makedirs("population/", exist_ok=True)


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Initialize the database pool
    await databaseInstance.connect()
    await initDatabase()
    print("Database pool created and initialized.")

    # Yield control back to the application
    yield

    # Close the database pool
    print("Database pool closed.")
    mailerInstance.close()
    print("Closed Mailer.")


app = FastAPI(
    title="Evolve",
    description="Tool for running Evlutionary Algorithms in Python",
    version="0.1.0",
    docs_url="/docs",
    redoc_url=None,
    lifespan=lifespan,
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

app.include_router(runnerRouter)
app.include_router(authRouter)
