import os

import evaluate
import numpy as np
from datasets import load_dataset
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorWithPadding,
    Trainer,
    TrainingArguments,
)

import wandb


def tokenize(examples):
    outputs = tokenizer(examples["text"], truncation=True)
    return outputs


def compute_metrics(eval_preds):
    metric = evaluate.load("accuracy")
    logits, labels = eval_preds
    predictions = np.argmax(logits, axis=-1)
    return metric.compute(predictions=predictions, references=labels)


if __name__ == "__main__":
    ds = load_dataset("imdb")

    model_name = "gpt2"
    model = AutoModelForSequenceClassification.from_pretrained(model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        model.config.pad_token_id = model.config.eos_token_id

    tokenized_ds = ds.map(tokenize, batched=True)

    training_args = TrainingArguments(
        num_train_epochs=1,
        output_dir="gpt2-imdb-sentiment-classifier",
        push_to_hub=True,
        per_device_train_batch_size=16,
        per_device_eval_batch_size=16,
        evaluation_strategy="epoch",
    )

    data_collator = DataCollatorWithPadding(tokenizer)

    trainer = Trainer(
        model=model,
        tokenizer=tokenizer,
        data_collator=data_collator,
        args=training_args,
        train_dataset=tokenized_ds["train"],
        eval_dataset=tokenized_ds["test"],
        compute_metrics=compute_metrics,
    )

    trainer.train()
    trainer.push_to_hub()
