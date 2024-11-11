from data.file_path import DATASET_TOPIC1_PATH, DATASET_TOPIC2_PATH, DATASET_TOPIC3_PATH, DATASET_ALL_PATH
from datasets import concatenate_datasets, load_dataset


def normalize_fields(example):
    if example["language"] != "English":
        example["input"] = example["translation"]
    else:
        example["input"] = example["content"]
    return example


def load_topic_and_split(test_size: float):
    for topic_path in [DATASET_TOPIC1_PATH, DATASET_TOPIC2_PATH, DATASET_TOPIC3_PATH]:
        full_dataset = load_dataset("json", data_files=str(topic_path / "full.jsonl"))["train"]
        full_dataset = full_dataset.map(normalize_fields)
        full_dataset = full_dataset.train_test_split(test_size=test_size)
        full_dataset["train"].to_json(topic_path / "train.jsonl")
        full_dataset["test"].to_json(topic_path / "test.jsonl")


def concat_datasets():
    for split in ["train", "test", "full"]:
        dataset1 = load_dataset("json", data_files=str(DATASET_TOPIC1_PATH / f"{split}.jsonl"))["train"]
        dataset2 = load_dataset("json", data_files=str(DATASET_TOPIC2_PATH / f"{split}.jsonl"))["train"]
        dataset3 = load_dataset("json", data_files=str(DATASET_TOPIC3_PATH / f"{split}.jsonl"))["train"]
        full_dataset = concatenate_datasets([dataset1, dataset2, dataset3])
        full_dataset.to_json(DATASET_ALL_PATH / f"{split}.jsonl")


if __name__ == '__main__':
    # DO NOT RUN THIS FILE MULTIPLE TIMES! IT WILL OVERWRITE CURRENT SPLITS
    load_topic_and_split(0.3)
    concat_datasets()
