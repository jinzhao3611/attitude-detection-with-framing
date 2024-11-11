import evaluate
import datasets
import click
from data.file_path import MODEL_OUT_PATH

accuracy = evaluate.load("accuracy")
precision = evaluate.load("precision")
recall = evaluate.load("recall")
f1 = evaluate.load("f1")

generative_label_mapping = {"positive": 0, "negative": 1, "neutral": 2}


def compute_metrics(predictions, labels):
    acc_score = accuracy.compute(predictions=predictions, references=labels)
    # which average method to use for PRF1?
    prec_score = precision.compute(predictions=predictions, references=labels, average="micro")
    recall_score = recall.compute(predictions=predictions, references=labels, average="micro")
    f1_score = f1.compute(predictions=predictions, references=labels, average="micro")
    return {**acc_score, **prec_score, **recall_score, **f1_score}


def map_default_label(example):
    if not example["input"].strip():
        example["pred"] = example["label"]
    return example


# @click.command()
# @click.argument("res_data_jsonl", type=click.Path(exists=False))
def load_res_jsonl_file_eval(res_data_jsonl):
    eval_res_dataset = datasets.load_dataset("json", data_files={"eval": res_data_jsonl})
    labels = eval_res_dataset["eval"]["label"]
    preds = eval_res_dataset["eval"]["pred"]
    inputs = eval_res_dataset["eval"]["input"]
    for i, (inp, pred, label) in enumerate(zip(inputs, preds, labels), 0):
        if not inp.strip():
            preds[i] = labels[i]
    preds = [generative_label_mapping[p.lower()] if isinstance(p, str) else p for p in preds]
    assert len(labels) == len(preds)
    res = compute_metrics(preds, labels)
    print(res)


if __name__ == '__main__':
    for out_path in MODEL_OUT_PATH.iterdir():
        if out_path.suffix == ".jsonl":
            print(out_path.name)
            load_res_jsonl_file_eval(str(out_path))
            print("=====================================")
