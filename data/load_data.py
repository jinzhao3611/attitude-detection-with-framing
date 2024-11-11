import datasets

from data.file_path import TOPIC_NAME_MAPPING
from datasets import load_dataset, ClassLabel


def load_jsonl_data(topic: str, *, device: int) -> datasets.Dataset:
    topic_path = TOPIC_NAME_MAPPING[topic]
    if device == 0:
        data_files = {"train": str(topic_path / "train.jsonl"), "test": str(topic_path / "test.jsonl")}
    else:
        data_files = {"train": str(topic_path / f"train_device_{device}.jsonl"),
                      "test": str(topic_path / f"test_device_{device}.jsonl")}

    dataset = load_dataset("json", data_files=data_files)
    dataset["train"] = dataset["train"].rename_column("stance", "label")
    dataset["test"] = dataset["test"].rename_column("stance", "label")
    class_feature = ClassLabel(names=["pos", "neg", "neutral"],
                               num_classes=3)  # label is mapped to 0, 1, 2 in the order of names
    dataset["train"] = dataset["train"].cast_column("label", class_feature)
    dataset["test"] = dataset["test"].cast_column("label", class_feature)

    return dataset


if __name__ == '__main__':
    pass
