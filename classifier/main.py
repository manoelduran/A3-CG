import json
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI
from joblib import load

import shared_state
from routes import prediction_router

app = FastAPI()


@asynccontextmanager
async def lifespan(_):
    model_dir = Path("models/svm_v1")
    model = load("models/svm_v1/model-pkl")
    with open(model_dir / "classes.json") as f:
        classes = json.load(f)

    shared_state.model = model
    shared_state.classes = classes
    yield


app.include_router(prediction_router)
