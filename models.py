from typing import Optional
from pydantic import BaseModel

class rootRes(BaseModel):
    message: str
    description: Optional[str] = None

