# txl_hf/train_hf.py
from transformers import Trainer, TrainingArguments
from txl_hf.build_dataset import get_train_eval
from txl_hf.collator_stream import ConversationCollator
from txl_hf.utils_logging import get_logger

# 두 래퍼를 모두 import
from txl_hf.mem_baseline import HFMemBaseline
from txl_hf.mem_ta import HFMemTA

def load_model(variant: str, **model_kwargs):
    if variant == "baseline":
        return HFMemBaseline(**model_kwargs)
    elif variant == "ta":
        return HFMemTA(**model_kwargs)
    else:
        raise ValueError(f"Unknown model variant: {variant}")

def main():
    # 예: argparse/Hydra로 variant를 입력받는다고 가정
    variant = "ta"        # "baseline" | "ta"
    logger = get_logger()

    train_ds, eval_ds, tokenizer = get_train_eval()
    collator = ConversationCollator(tokenizer=tokenizer, use_topic_boundary=(variant=="ta"))

    model = load_model(
        variant,
        n_layer=18, n_head=16, d_model=1024, d_head=64, d_inner=4096,
        vocab_size=len(tokenizer), dropout=0.1,
        # TA 전용 하이퍼: long/short 길이 등은 HFMemTA 쪽 __init__으로 전달
        long_mem_len=64, short_mem_len=256,
    )

    args = TrainingArguments(
        output_dir=f"./artifacts/{variant}",
        per_device_train_batch_size=2,
        learning_rate=5e-5,
        num_train_epochs=1,
        logging_dir="./logs",
        evaluation_strategy="steps",
        eval_steps=500,
        save_steps=1000,
        fp16=True
    )

    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        data_collator=collator
    )

    trainer.train()
    logger.info("done!")
