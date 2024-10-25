# main.py

from fastapi import FastAPI
from fastapi.responses import JSONResponse
from fastapi.encoders import jsonable_encoder

app = FastAPI()

@app.get("/")
async def root():
    return JSONResponse(
        status_code=200,
        content=jsonable_encoder({"message": "Hello World"}),
    )

@app.get("/algorithm")
async def algorithm():
    return JSONResponse(
        status_code=200,
        content=jsonable_encoder({"message": "Algorithm", "data": [
            {"name": "eaSimple", "type": "deap"},
            {"name": "eaMuPlusLambda", "type": "deap"},
            {"name": "eaMuCommaLambda", "type": "deap"},
            {"name": "eaGenerateUpdate", "type": "deap"},
        ]}),
    )
