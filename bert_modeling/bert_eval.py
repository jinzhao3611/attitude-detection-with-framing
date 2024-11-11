from data.load_data import load_jsonl_data
from data.file_path import MODEL_OUT_PATH
from bert_modeling import PRETRAINED_MODEL
from bert_modeling.eval_metrics import compute_metrics
from transformers import AutoTokenizer
from transformers import DataCollatorWithPadding
from transformers import AutoModelForSequenceClassification, Trainer
import click

import numpy as np

model_path = PRETRAINED_MODEL

tokenizer = AutoTokenizer.from_pretrained(model_path)
data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

id2label = {0: "pos", 1: "neg", 2: "neutral"}
label2id = {"pos": 0, "neg": 1, "neutral": 2}


def preprocess_function(examples):
    examples = [" ".join(e.split()) for e in examples["input"]]
    return tokenizer(examples, max_length=512, truncation=True, padding="max_length")


@click.command()
@click.argument("model_path", type=click.Path(exists=True))
@click.option("--topic", type=click.STRING, required=True)
@click.option("--device", type=click.STRING, required=True)
def run_predict(model_path: str, topic: str, device: str):
    dataset = load_jsonl_data(topic=topic, device=int(device))
    test_set = dataset["test"]
    tokenized_test_set = test_set.map(preprocess_function, batched=True)
    model = AutoModelForSequenceClassification.from_pretrained(
        model_path, num_labels=3, id2label=id2label, label2id=label2id
    )
    trainer = Trainer(
        model=model,
        data_collator=data_collator,
        tokenizer=tokenizer,
        compute_metrics=compute_metrics,
    )
    results = trainer.predict(tokenized_test_set)
    preds = np.argmax(results.predictions, axis=1).tolist()
    print(preds)

    test_set = test_set.add_column("pred", preds)
    test_set.to_json(MODEL_OUT_PATH / f"{topic}_device{device}_{PRETRAINED_MODEL.split('/')[-1]}_out.jsonl")


if __name__ == '__main__':
    run_predict()
