from fastapi.responses import JSONResponse
from fastapi.encoders import jsonable_encoder

class InternalServerError(JSONResponse):
    def __init__(self):
        super().__init__(
            status_code=500,
            content=jsonable_encoder({"message": "Internal Server Error"}),
        )