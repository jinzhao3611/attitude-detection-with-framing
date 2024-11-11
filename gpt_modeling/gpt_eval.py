from openai import OpenAI
import click
from gpt_modeling import PRETRAINED_MODEL
from data.file_path import MODEL_OUT_PATH
from gpt_modeling.prompts import construct_framing_prompt, FULL_TOPIC, TOPIC_INSTRUCT
from data.load_data import load_jsonl_data

api_key = "Your API key here"
client = OpenAI(api_key=api_key)


def run_gpt(prompt: str):
    msg = {
        "role": "user",
        "content": prompt
    }
    response = client.chat.completions.create(
        model=PRETRAINED_MODEL,
        messages=[msg],
        temperature=0,
        max_tokens=512,
        top_p=1,
        frequency_penalty=0,
        presence_penalty=0
    )
    response_txt = response.choices[0].message.content
    return response_txt


@click.command()
@click.option("--topic", type=click.STRING, required=True)
@click.option("--device", type=click.STRING, required=True)
def eval_gpt_res2file(topic: str, device: str):
    final_prompt = construct_framing_prompt(device=int(device))
    dataset = load_jsonl_data(topic, device=int(device))
    test_dataset = dataset["test"]
    preds = []
    for i, ins in enumerate(test_dataset, 1):
        input_str = ins["input"]
        # input_str = " ".join(input_str.split()[:512])  # shortened to compare with other models
        prompt_str = final_prompt.invoke({"input": input_str, "full_topic":
            FULL_TOPIC[topic], "topic_instruct": TOPIC_INSTRUCT[topic]}).to_string()
        output_str = run_gpt(prompt_str)
        if i < 5:
            print(prompt_str)
            print(output_str)
            print("=====================================")
        else:
            print(output_str)
        preds.append(output_str)
        if i % 10 == 0:
            print(f"finishing {i} sentences ...")
    test_dataset = test_dataset.add_column("pred", preds)
    test_dataset.to_json(MODEL_OUT_PATH / f"{topic}_device{device}_{PRETRAINED_MODEL.split('/')[-1]}_out.jsonl")


if __name__ == '__main__':
    eval_gpt_res2file()
