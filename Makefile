
.PHONY: all train predict

train:
	@cd classifier && \
		uv run cli.py train --data-dir data/train --out-dir model/svm_v1

predict:
	@cd classifier && \
		uv run cli.py predict --image $(image) --model-dir model/svm_v1

classifier-worker:
	@cd classifier && \
		uv run worker.py
