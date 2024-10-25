from fastapi import File, UploadFile, HTTPException
from fastapi.responses import JSONResponse
from fastapi.encoders import jsonable_encoder
from fastapi import APIRouter
from models import *
from runner import Runner
import uuid
import os
import pickle


apiRouter = APIRouter(prefix="/api")

backend_url = os.getenv("BASE_URL", "http://localhost:8000") + "/api"

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


@apiRouter.post("/runAlgo")
async def runAlgo(runAlgoModel: RunAlgoModel):

    runner = Runner(id = str(uuid.uuid4()))

    runner.create(
        individual = "binaryString",
        populationFunction = "initRepeat",
        weights=(1.0,),
        individualSize=10,
        indpb=0.10,
        randomRange = [0, 100]
        )

    log, hof = runner.run(
        poputlationSize=5000,
        generations=10,
        cxpb=0.5,
        mutpb=0.2
    )

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
            "plot": f"{backend_url}/plots/{runner.id}/fitness_plot.png",
            "population": f"{backend_url}/population/{runner.id}/population.pkl"
        }}),
    )



# NOTE TO DEVELOPERS : USE SWAGGER UI TO test this endpoint. {BASE_URL}/docs
@apiRouter.post(
        "/unpickleFile/", 
        response_model=UnpickleFileModel, 
        summary="Unpickle File and Return Data", 
        description="Accepts a pickle file upload, unpickles it, and returns the data as JSON."
        )
async def upload_file(file: UploadFile = File(..., description="Upload a pickled (.pkl) file")):
    try:        
        contents = await file.read()
        data = pickle.loads(contents)

        return {"data": data}

    except pickle.UnpicklingError:
        raise HTTPException(status_code=400, detail="Failed to unpickle file. Invalid file format.")
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))