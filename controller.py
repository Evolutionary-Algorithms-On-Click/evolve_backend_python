from fastapi.responses import JSONResponse
from fastapi.encoders import jsonable_encoder
from fastapi import APIRouter
from models import *
from runner import Runner
import uuid
import os

apiRouter = APIRouter(prefix="/api")

backend_url = os.getenv('BASE_URL')+"/api"

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


@apiRouter.get("/runAlgo")
async def runAlgo():
    runner = Runner(id = str(uuid.uuid4()))

    runner.create()

    pop, log, hof = runner.run()
    print("Best individual is: %s\nwith fitness: %s" % (hof[0], hof[0].fitness))
    
    gen, avg, min_, max_ = log.select("gen", "avg", "min", "max")

    runner.createPlot(gen, avg, min_, max_)


    return JSONResponse(
        status_code=200,
        content=jsonable_encoder({"message": "Run Algorithm", "data": {
            "best": hof[0],
            "generation": gen,
            "average": avg,
            "minimum": min_,
            "maximum": max_,
            "plot": f"{backend_url}/plots/{runner.id}/fitness_plot.png"
        }}),
    )

