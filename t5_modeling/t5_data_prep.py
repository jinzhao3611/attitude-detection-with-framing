import torch
from transformers import T5Tokenizer
from data.load_data import TOPIC_NAME_MAPPING, load_jsonl_data
from t5_modeling import PRETRAINED_MODEL
import click

tokenizer = T5Tokenizer.from_pretrained(PRETRAINED_MODEL)

# map the label to the corresponding label in natural language
label_mapping = {
    0: "positive",
    1: "negative",
    2: "neutral"
}

FULL_TOPIC = {
    "putin": "Putin's Election Win",
    "hk_protest": "Hong Kong Protest",
    "al_shifa": "Israel's Al-Shifa Hospital Raid",
}


def encode(batch):
    input_encodings = tokenizer(batch['input_text'], max_length=512, truncation=True, padding="max_length")
    target_encodings = tokenizer(batch['target_text'], max_length=16, truncation=True, padding="max_length")
    target_ids = target_encodings['input_ids']

    encodings = {
        'input_ids': input_encodings['input_ids'],
        'attention_mask': input_encodings['attention_mask'],
        'decoder_input_ids': target_ids,
        'decoder_attention_mask': target_encodings['attention_mask'],
        'labels': [label if label else -100 for label in target_ids],
    }
    return encodings


@click.command()
@click.option("--topic", type=click.STRING, required=True)
@click.option("--device", type=click.STRING, required=True)
def encode_jsonl(topic: str, device: str):
    question_input = f"What is the article's stance towards {FULL_TOPIC[topic]}?"  # maybe we can rephrase it as "sentiment"?

    def format_input(example):
        example['input_text'] = f'question: {question_input} context: {example["input"]}'
        example['target_text'] = label_mapping[example['label']]
        return example

    dataset = load_jsonl_data(topic, device=int(device))

    train_dataset = dataset["train"]
    valid_dataset = dataset["test"]

    train_dataset = train_dataset.map(format_input)
    valid_dataset = valid_dataset.map(format_input)

    encoded_train_dataset = train_dataset.map(encode)
    encoded_valid_dataset = valid_dataset.map(encode)

    columns_to_remove = ['uid', 'url', 'title', 'content', 'language', 'translation', 'input', 'label', 'input_text',
                         'target_text', 'decoder_input_ids', 'decoder_attention_mask']

    encoded_train_dataset = encoded_train_dataset.remove_columns(columns_to_remove)
    encoded_valid_dataset = encoded_valid_dataset.remove_columns(columns_to_remove)

    columns = ['input_ids', 'attention_mask', 'labels']
    encoded_train_dataset.set_format(type='torch', columns=columns)
    encoded_valid_dataset.set_format(type='torch', columns=columns)

    torch.save(encoded_train_dataset, str(TOPIC_NAME_MAPPING[topic] / f"t5_train_device_{device}.pt"))
    torch.save(encoded_valid_dataset, str(TOPIC_NAME_MAPPING[topic] / f"t5_test_device_{device}.pt"))


if __name__ == '__main__':
    encode_jsonl()
