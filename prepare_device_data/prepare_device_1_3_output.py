import tqdm
from openai import OpenAI
import click
import json
from prepare_device_data.device_prompts import construct_device3_prompt, construct_device1_prompt
from data.file_path import TOPIC_NAME_MAPPING

PRETRAINED_MODEL = "gpt-4o"
api_key = "Your API Key Here"
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


def run_gpt_device1_old(topic: str):
    out_f = open(TOPIC_NAME_MAPPING[topic] / "device_1_gpt_output.jsonl", "w")
    with open(TOPIC_NAME_MAPPING[topic] / "device_1_gpt_input.jsonl", "r") as f:
        for i, line in tqdm.tqdm(enumerate(f), desc="Running GPT on device-1"):
            data_dict = json.loads(line)
            contexts = data_dict["contexts"]
            responses = []
            for event_id, context in contexts:
                prompt_str = construct_device1_prompt(context).to_string()
                output_str = run_gpt(prompt_str)
                responses.append(output_str)
            data_dict["responses"] = responses
            out_f.write(json.dumps(data_dict) + "\n")
    out_f.close()


def run_gpt_device3(topic: str):
    out_f = open(TOPIC_NAME_MAPPING[topic] / "device_3_gpt_output.jsonl", "w")
    with open(TOPIC_NAME_MAPPING[topic] / "device_3_gpt_input.jsonl", "r") as f:
        for i, line in tqdm.tqdm(enumerate(f), desc="Running GPT on device-3"):
            data_dict = json.loads(line)
            contexts = data_dict["sents"]
            context_str = " ".join(contexts)
            prompt_str = construct_device3_prompt(context_str).to_string()
            output_str = run_gpt(prompt_str)
            data_dict["response"] = output_str
            out_f.write(json.dumps(data_dict) + "\n")
    out_f.close()


def run_gpt_device1(topic: str):
    out_f = open(TOPIC_NAME_MAPPING[topic] / "device_1_gpt_output.jsonl", "w")
    with open(TOPIC_NAME_MAPPING[topic] / "device_1_gpt_input.jsonl", "r") as f:
        for i, line in tqdm.tqdm(enumerate(f), desc="Running GPT on device-1"):
            data_dict = json.loads(line)
            contexts = [f"{num}. {c[1]}" for num, c in enumerate(data_dict["contexts"], 1)]
            assert len(contexts) == 5
            context_str = "\n".join(contexts)

            prompt_str = construct_device1_prompt(context_str).to_string()
            output_str = run_gpt(prompt_str)
            data_dict["responses"] = output_str
            out_f.write(json.dumps(data_dict) + "\n")
    out_f.close()


@click.command()
@click.option("--topic", type=click.STRING, required=True)
@click.option("--device", type=click.STRING, required=True)
def main(topic: str, device: str):
    device = str(device)
    if device == "1":
        run_gpt_device1(topic)
    elif device == "3":
        run_gpt_device3(topic)


if __name__ == '__main__':
    main()