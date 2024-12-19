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


class UnpickleFileModel(BaseModel):
    data: list


class RegisterUserModel(BaseModel):
    userName: str
    userEmail: str
    password: str
