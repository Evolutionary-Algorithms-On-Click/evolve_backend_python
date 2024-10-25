from typing import Optional
from pydantic import BaseModel

class rootRes(BaseModel):
    message: str
    description: Optional[str] = None


class RunAlgoModel(BaseModel):
    algorithm: str
    individual: str
    populationFunction: str
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
    tournamentSize: int


class UnpickleFileModel(BaseModel):
    data: list
    

