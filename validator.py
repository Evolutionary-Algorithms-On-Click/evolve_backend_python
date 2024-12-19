from fastapi import Request, HTTPException

from config import ParamsList

paramsList = ParamsList()


# Define the dependency
async def validateRunAlgoRequest(request: Request):

    # Get the request body
    body = await request.json()

    if body["algorithm"] in paramsList.algorithm:
        # if body["algorithm"] == "eaSimple":
        #     if body["mu"] is not None or body["lambda_"] is not None:
        #         raise HTTPException(status_code=400, detail="Invalid parameters for eaSimple")
        if body["algorithm"] == "eaMuPlusLambda":
            if body["mu"] is None or body["lambda_"] is None:
                raise HTTPException(
                    status_code=400, detail="Invalid parameters for eaMuPlusLambda"
                )
        elif body["algorithm"] == "eaMuCommaLambda":
            if body["mu"] is None or body["lambda_"] is None:
                raise HTTPException(
                    status_code=400, detail="Invalid parameters for eaMuCommaLambda"
                )
            if body["mu"] >= body["lambda_"]:
                raise HTTPException(
                    status_code=400,
                    detail="mu should be less than lambda for eaMuCommaLambda",
                )
        # elif body["algorithm"] == "eaGenerateUpdate":
        #     if body["mu"] is not None or body["lambda_"] is not None:
        #         raise HTTPException(status_code=400, detail="Invalid parameters for eaGenerateUpdate")
        pass
    else:
        raise HTTPException(status_code=400, detail="Invalid algorithm")

    if body["individual"] in paramsList.individual:
        pass
    else:
        raise HTTPException(status_code=400, detail="Invalid individual")

    if body["populationFunction"] in paramsList.populationFunction:
        pass
    else:
        raise HTTPException(status_code=400, detail="Invalid population function")

    if body["evaluationFunction"] in paramsList.evaluationFunction:
        pass
    else:
        raise HTTPException(status_code=400, detail="Invalid evaluation function")

    if body["crossoverFunction"] in paramsList.crossoverFunction:
        pass
    else:
        raise HTTPException(status_code=400, detail="Invalid crossover function")
