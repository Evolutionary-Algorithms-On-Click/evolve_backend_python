from fastapi.responses import JSONResponse
from fastapi.encoders import jsonable_encoder
from fastapi import APIRouter
from models import *

apiRouter = APIRouter(prefix="/api")

@apiRouter.get("/test")
async def root():
    return JSONResponse(
        status_code=200,
        content=jsonable_encoder({"message": "Hello World"}),
    )

@apiRouter.get("/algorithm")
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

