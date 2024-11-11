from typing import Any

import click

from data.load_data import load_jsonl_data
from data.file_path import MODEL_OUT_PATH
from transformers import T5ForConditionalGeneration, T5Tokenizer
from t5_modeling import PRETRAINED_MODEL


def predict(tokenizer: Any, model: Any, input_str: str):
    question_input = f"What is the article's stance?"  # maybe we can rephrase it as "sentiment"?

    input_str = [f'question: {question_input} context: {i}' for i in input_str]
    input_encodings = tokenizer(input_str, padding="max_length", truncation=True, max_length=256, return_tensors="pt")
    output_str = model.generate(input_ids=input_encodings['input_ids'],
                                attention_mask=input_encodings['attention_mask'],
                                max_length=16,
                                early_stopping=False)
    pred_answers = [tokenizer.convert_ids_to_tokens(ids, skip_special_tokens=True) for ids in output_str]
    pred_answers = [tokenizer.convert_tokens_to_string(pred) for pred in pred_answers]
    return pred_answers


def load_jsonl_data_and_write_pred(topic: str, device, tokenizer: Any, model: Any):
    dataset = load_jsonl_data(topic, device=int(device))

    test_dataset = dataset["test"]
    preds = []

    step = 10

    for i in range(0, len(test_dataset), step):
        input_str = test_dataset["input"][i:i + step]
        output_str = predict(tokenizer, model, input_str)
        preds.extend(output_str)
        print(output_str)
        if i % 10 == 0:
            print(f"finishing {i} sentences ...")
    test_dataset = test_dataset.add_column("pred", preds)
    test_dataset.to_json(MODEL_OUT_PATH / f"{topic}_device{device}_{PRETRAINED_MODEL.split('/')[-1]}_out.jsonl")


@click.command()
@click.argument("model_path", type=click.Path(exists=True))
@click.option("--topic", type=click.STRING, required=True)
@click.option("--device", type=click.STRING, required=True)
def main(model_path: str, topic: str, device: str):
    model = T5ForConditionalGeneration.from_pretrained(model_path)
    tokenizer = T5Tokenizer.from_pretrained(PRETRAINED_MODEL)

    print("Finish loading model!")
    load_jsonl_data_and_write_pred(topic, device, tokenizer, model)


if __name__ == '__main__':
    main()
