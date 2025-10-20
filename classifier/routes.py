from typing import Annotated

from fastapi import APIRouter, File
from pydantic import BaseModel

import shared_state
from cocoa_classifier import predictor

prediction_router = APIRouter()


class PredictResponse(BaseModel):
    overlay: bytes
    results: list[predictor.PredictionResultRow]


@prediction_router.post(
    "/predict",
    response_model=PredictResponse,
)
async def predict(
    file: Annotated[bytes, File()],
    single_bean: bool = False,
):
    overlay, results = predictor.predict(
        file=file,
        model=shared_state.model,
        classes=shared_state.classes,
        single_bean=single_bean,
    )
    return PredictResponse(
        overlay=overlay.tobytes(),
        results=results,
    )
