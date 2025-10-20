import argparse
from pathlib import Path

from pydantic import BaseModel, Field


class TrainConfig(BaseModel):
    model_config = {
        "from_attributes": True,
    }

    data_dir: Path
    out_dir: Path


class PredictConfig(BaseModel):
    model_config = {
        "from_attributes": True,
    }

    image: str
    model_dir: Path
    out_dir: Path


class ClassifierConfig(BaseModel):
    model_config = {
        "from_attributes": True,
    }

    cmd: str
    data_dir: Path | None = Field(None)
    image: str | None = Field(None)
    model_dir: Path | None = Field(None)
    out_dir: Path | None = Field(None)

    def train_config(self) -> TrainConfig:
        return TrainConfig.model_validate(self)

    def predict_config(self) -> PredictConfig:
        return PredictConfig.model_validate(self)

    @classmethod
    def from_args(cls) -> "ClassifierConfig":
        ap = argparse.ArgumentParser(
            description="Cocoa Beans Classifier (OpenCV + Python)",
        )
        sub = ap.add_subparsers(dest="cmd", required=True)

        ap_train = sub.add_parser("train", help="Train an SVM on single-bean images")
        ap_train.add_argument(
            "--data-dir",
            required=True,
            type=Path,
            help="Folder with class subfolders",
        )
        ap_train.add_argument(
            "--out-dir",
            required=True,
            type=Path,
            help="Where to save the model",
        )

        ap_pred = sub.add_parser(
            "predict",
            help="Predict classes for beans in a multi-bean image",
        )
        ap_pred.add_argument(
            "--image",
            required=True,
            type=str,
            help="Input image path",
        )
        ap_pred.add_argument(
            "--model-dir",
            required=True,
            type=Path,
            help="Folder with model assets",
        )
        ap_pred.add_argument(
            "--out-dir",
            required=True,
            type=Path,
            help="Output folder",
        )

        return cls.model_validate(ap.parse_args())
