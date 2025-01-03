from fastapi import File, UploadFile, HTTPException, Depends
from fastapi.responses import JSONResponse
from fastapi.encoders import jsonable_encoder
from fastapi import APIRouter
from models import *
from runner import Runner
from gpRunner import GpRunner
from validator import validateRunAlgoRequest
from config import ParamsList
import uuid
import os
import pickle


apiRouter = APIRouter(prefix="/api")

backend_url = os.getenv("BASE_URL", "http://localhost:8000") + "/api"

paramsList = ParamsList()


@apiRouter.get(
    "/test",
    summary="API Healthcheck",
    description="Returns a simple message to test the API Health.",
)
async def root():
    return JSONResponse(
        status_code=200,
        content=jsonable_encoder({"message": "Hello World"}),
    )


@apiRouter.get(
    "/validParams",
    summary="Endpoint to fetch valid parameters",
    description="Returns the valid parameters for the algorithm.",
)
async def algorithm():
    return JSONResponse(
        status_code=200,
        content=jsonable_encoder(
            {
                "message": "Successfully fetched valid params",
                "algorithms": paramsList.algorithm,
                "individual": paramsList.individual,
                "populationFunction": paramsList.populationFunction,
                "evaluationFunction": paramsList.evaluationFunction,
                "crossoverFunction": paramsList.crossoverFunction,
                "mutationFunction": paramsList.mutationFunction,
                "selectionFunction": paramsList.selectionFunction,
            }
        ),
    )


@apiRouter.post(
    "/runAlgo",
    dependencies=[Depends(validateRunAlgoRequest)],
    summary="Endpoint to run the algorithm",
    description="Accepts the parameters required to run the algorithm and returns the results.",
)
async def runAlgo(runAlgoModel: RunAlgoModel):

    runner = Runner(id=str(uuid.uuid4()))

    runner.create(
        individual=runAlgoModel.individual,
        populationFunction=runAlgoModel.populationFunction,
        evaluationFunction=runAlgoModel.evaluationFunction,
        weights=runAlgoModel.weights,
        individualSize=runAlgoModel.individualSize,
        indpb=runAlgoModel.indpb,
        randomRange=runAlgoModel.randomRange,
        crossoverFunction=runAlgoModel.crossoverFunction,
        mutationFunction=runAlgoModel.mutationFunction,
        selectionFunction=runAlgoModel.selectionFunction,
        tournamentSize=3,
    )

    log, hof = runner.run(
        algorithm=runAlgoModel.algorithm,
        populationSize=runAlgoModel.populationSize,
        generations=runAlgoModel.generations,
        cxpb=runAlgoModel.cxpb,
        mutpb=runAlgoModel.mutpb,
        mu=runAlgoModel.mu,
        lambda_=runAlgoModel.lambda_,
        N=runAlgoModel.individualSize,
        hofSize=runAlgoModel.hofSize,
    )

    print("Best individual is: %s\nwith fitness: %s" % (hof[0], hof[0].fitness))

    hofSerializable = [
        {
            "individual": list(ind),
            "fitness": ind.fitness.values if ind.fitness else None,
        }
        for ind in hof
    ]

    runner.createPlots(log)

    gen, avg, min_, max_ = log.select("gen", "avg", "min", "max")

    return JSONResponse(
        status_code=200,
        content=jsonable_encoder(
            {
                "message": "Run Algorithm",
                "runId": runner.id,
                "data": {
                    "generation": gen,
                    "average": avg,
                    "minimum": min_,
                    "maximum": max_,
                },
                "plots": {
                    "fitnessPlot": f"{backend_url}/plots/{runner.id}/fitness_plot.png",
                    "mutationCrossoverEffectPlot": f"{backend_url}/plots/{runner.id}/mutation_crossover_effect.png",
                },
                "population": f"{backend_url}/population/{runner.id}/population.pkl",
                "hallOfFame": hofSerializable,
            }
        ),
    )


@apiRouter.post(
    "/runGpAlgo",
    summary="Endpoint to run GP algorithm",
    description="Accepts the parameters required to run the algorithm and returns the results.",
)
async def runGpAlgo(runGpAlgoModel: RunGpAlgoModel):

    runner = GpRunner(id=str(uuid.uuid4()))

    runner.addPrimitives(runGpAlgoModel.operators)
    runner.addEphemeralConstant()
    runner.renameArguments(arg_names=runGpAlgoModel.argNames)

    runner.create(
        individualType=runGpAlgoModel.individualType,
        expr=runGpAlgoModel.expr,
        min_=runGpAlgoModel.min_,
        max_=runGpAlgoModel.max_,
        realFunction=runGpAlgoModel.realFunction,
        individualFunction=runGpAlgoModel.individualFunction,
        populationFunction=runGpAlgoModel.populationFunction,
        selectionFunction=runGpAlgoModel.selectionFunction,
        tournamentSize=runGpAlgoModel.tournamentSize,
        expr_mut=runGpAlgoModel.expr_mut,
        crossoverFunction=runGpAlgoModel.crossoverFunction,
        terminalProb=runGpAlgoModel.terminalProb,
        mutationFunction=runGpAlgoModel.mutationFunction,
        mateHeight=runGpAlgoModel.mateHeight,
        mutHeight=runGpAlgoModel.mutHeight,
        weights=runGpAlgoModel.weights,
    )

    log, hof = runner.run(
        algorithm=runGpAlgoModel.algorithm,
        populationSize=runGpAlgoModel.populationSize,
        generations=runGpAlgoModel.generations,
        cxpb=runGpAlgoModel.cxpb,
        mutpb=runGpAlgoModel.mutpb,
        mu=runGpAlgoModel.mu,
        lambda_=runGpAlgoModel.lambda_,
        N=runGpAlgoModel.individualSize,
        hofSize=runGpAlgoModel.hofSize,
    )

    # print("Best individual is: %s\nwith fitness: %s" % (hof[0], hof[0].fitness))

    # hofSerializable = [
    #     {
    #         "individual": list(ind),
    #         "fitness": ind.fitness.values if ind.fitness else None,
    #     }
    #     for ind in hof
    # ]

    runner.createPlots(log)

    return JSONResponse(
        status_code=200,
        content=jsonable_encoder(
            {
                "message": "Run Algorithm",
                "runId": runner.id,
                "bestFitness": hof[0].fitness.values,
                "plots": {
                    "treePlot": f"{backend_url}/plots/{runner.id}/graph.png",
                },
                "population": f"{backend_url}/population/{runner.id}/population.pkl",
            }
        ),
    )


# NOTE TO DEVELOPERS : USE SWAGGER UI TO test this endpoint. {BASE_URL}/docs
@apiRouter.post(
    "/unpickleFile/",
    response_model=UnpickleFileModel,
    summary="Unpickle File and Return Data",
    description="Accepts a pickle file upload, unpickles it, and returns the data as JSON.",
)
async def upload_file(
    file: UploadFile = File(..., description="Upload a pickled (.pkl) file")
):
    try:
        contents = await file.read()
        data = pickle.loads(contents)

        return {"data": data}

    except pickle.UnpicklingError:
        raise HTTPException(
            status_code=400, detail="Failed to unpickle file. Invalid file format."
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
