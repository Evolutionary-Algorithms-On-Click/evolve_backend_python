from fastapi.responses import JSONResponse
from fastapi.encoders import jsonable_encoder

class NotFoundError(JSONResponse):
    def __init__(self, message="Not Found"):
        super().__init__(
            status_code=404,
            content=jsonable_encoder({"message": message}),
        )