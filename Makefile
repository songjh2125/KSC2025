.PHONY: labels train eval demo

labels:
	python scripts/make_labels.py --out cache/labels_ko_samsum.jsonl --max_samples 0 --sleep 0.0

train:
	python scripts/memLLM_QLoRA_SOLAR_koSamsum_train.py

eval:
	python scripts/eval_mem.py --labels_cache cache/labels_ko_samsum.jsonl --max_samples 1000

demo:
	python scripts/mem_infer_demo.py --ckpt out/solar-mem-qlora --max_new_tokens 256 --temperature 0.7
