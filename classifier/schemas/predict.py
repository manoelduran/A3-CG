from pydantic import BaseModel


class PredictRequest(BaseModel):
    image: str
