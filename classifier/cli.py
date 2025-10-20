import csv
import json
from pathlib import Path

import cv2
from joblib import load

from cocoa_classifier.predictor import predict
from cocoa_classifier.trainer import train
from schemas.classifier_config import ClassifierConfig


def main():
    args = ClassifierConfig.from_args()

    if args.cmd == "train":
        train_config = args.train_config()
        train(train_config.data_dir, train_config.out_dir)
    elif args.cmd == "predict":
        predict_config = args.predict_config()
        model_dir = predict_config.model_dir
        out_dir = predict_config.out_dir

        image_path = predict_config.image
        model = load(model_dir / "model.pkl")
        with open(model_dir / "classes.json") as f:
            classes = json.load(f)
        out_dir.mkdir(parents=True, exist_ok=True)

        with open(image_path, "rb") as f:
            encoded_image = f.read()

        overlay, results = predict(
            file=encoded_image,
            model=model,
            classes=classes,
        )

        out_img = out_dir / f"{Path(image_path).stem}_annotated.png"
        cv2.imwrite(str(out_img), overlay)

        out_csv = out_dir / "predictions.csv"
        if results:
            with open(out_csv, "w", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=results[0].keys())
                writer.writeheader()
                writer.writerows(results)

        print(f"Wrote {out_img}")
        print(f"Wrote {out_csv}")


if __name__ == "__main__":
    main()
