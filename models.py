from typing import Optional
from pydantic import BaseModel


class rootRes(BaseModel):
    message: str
    description: Optional[str] = None


class RunAlgoModel(BaseModel):
    algorithm: str
    individual: str
    populationFunction: str
    evaluationFunction: str
    populationSize: int
    generations: int
    cxpb: float
    mutpb: float
    weights: tuple
    individualSize: int
    indpb: float
    randomRange: list
    crossoverFunction: str
    mutationFunction: str
    selectionFunction: str
    tournamentSize: Optional[int] = None
    mu: Optional[int] = None
    lambda_: Optional[int] = None
    hofSize: Optional[int] = 1


class RunGpAlgoModel(BaseModel):
    algorithm: str
    arity: int
    operators: list
    argNames: list
    individualType: str
    expr: str
    realFunction: str
    min_: int
    max_: int
    individualFunction: str
    populationFunction: str
    selectionFunction: str
    tournamentSize: int
    expr_mut: str
    crossoverFunction: str
    terminalProb: float
    mutationFunction: str
    mutationMode: str
    mateHeight: int
    mutHeight: int
    weights: tuple
    populationSize: int
    generations: int
    cxpb: float
    mutpb: float
    mu: int
    lambda_: int
    individualSize: int
    hofSize: int
    expr_mut_min: int
    expr_mut_max: int


class UnpickleFileModel(BaseModel):
    data: list

class MlModel(BaseModel):
    algorithm: str # DONE
    # individual: str
    # populationFunction: str

    # evaluationFunction: str
    mlEvalFunctionCodeString: str # DONE

    populationSize: int # DONE
    generations: int # DONE
    cxpb: float # DONE
    mutpb: float # DONE
    weights: tuple # DONE

    # individualSize: int
    googleDriveUrl: str # DONE
    sep: str # DONE
    mlImportCodeString: str # DONE
    targetColumnName: str # DONE

    indpb: float # DONE
    crossoverFunction: str # DONE
    mutationFunction: str # DONE
    selectionFunction: str # DONE
    tournamentSize: Optional[int] = None # DONE
    mu: Optional[int] = None # DONE
    lambda_: Optional[int] = None # DONE
    hofSize: Optional[int] = 1 # DONE

