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
    mutationFunction: str
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


class UnpickleFileModel(BaseModel):
    data: list
