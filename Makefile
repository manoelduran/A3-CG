
.PHONY: all train predict

train:
	docker exec cocoa_classifier python cli.py train --data-dir data/train --out-dir model/svm_v1

predict:
	docker exec cocoa_classifier python cli.py predict --image data/predict/image.jpeg --model-dir model/svm_v1
