from flan_t5_modeling.prompts import construct_framing_prompt, FULL_TOPIC, TOPIC_INSTRUCT
from data.load_data import load_jsonl_data
from data.file_path import MODEL_OUT_PATH
import click
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
from flan_t5_modeling import PRETRAINED_MODEL
from typing import List
import torch

model = AutoModelForSeq2SeqLM.from_pretrained(PRETRAINED_MODEL)
tokenizer = AutoTokenizer.from_pretrained(PRETRAINED_MODEL)
device = torch.device('cuda')
model.to(device)


def run_flan_t5(prompt: List[str]):
    inputs = tokenizer(prompt, max_length=512, truncation=True, padding="max_length", return_tensors="pt")
    inputs.to(device)
    outputs = model.generate(**inputs, max_length=16)
    return tokenizer.batch_decode(outputs, skip_special_tokens=True)


@click.command()
@click.option("--topic", type=click.STRING, required=True)
@click.option("--device", type=click.STRING, required=True)
def eval_flan_t5_res2file(topic: str, device: str):
    final_prompt = construct_framing_prompt(device=int(device))
    dataset = load_jsonl_data(topic, device=int(device))
    test_dataset = dataset["test"]
    preds = []
    step = 10
    for i in range(0, len(test_dataset), step):
        input_strs = test_dataset["input"][i:i + step]

        prompt_strs = [final_prompt.invoke({"input": p, "full_topic":
            FULL_TOPIC[topic], "topic_instruct": TOPIC_INSTRUCT[topic]}).to_string() for p in input_strs]
        output_strs = run_flan_t5(prompt_strs)
        if i < 1:
            print(prompt_strs[0])
            print(output_strs[0])
            print("=====================================")
        else:
            print(output_strs)
        preds.extend(output_strs)
        if i % 10 == 0:
            print(f"finishing {i} sentences ...")
    test_dataset = test_dataset.add_column("pred", preds)
    test_dataset.to_json(MODEL_OUT_PATH / f"{topic}_device{device}_{PRETRAINED_MODEL.split('/')[-1]}_out.jsonl")


if __name__ == '__main__':
    eval_flan_t5_res2file()
