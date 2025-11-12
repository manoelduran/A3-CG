
.PHONY: all train predict

train:
	@docker compose stop classifier && \
	docker compose run --remove-orphans classifier python cli.py train --data-dir data/train --out-dir model/svm_v1 

predict:
	@docker compose stop classifier && \
	docker compose run --remove-orphans classifier python cli.py predict --image data/predict/$(image) --model-dir model/svm_v1 

train-predict:
	@docker compose stop classifier && \
	docker compose run --remove-orphans classifier \
	sh -c "python cli.py train --data-dir data/train --out-dir model/svm_v1 && python cli.py predict --image data/predict/$(image) --model-dir model/svm_v1" 
