from pathlib import Path

DATA_PATH = Path(__file__).parent.parent / "source_data"
DATASET_TOPIC1_PATH = DATA_PATH / "hk_protest"
DATASET_TOPIC2_PATH = DATA_PATH / "al_shifa"
DATASET_TOPIC3_PATH = DATA_PATH / "putin"
DATASET_ALL_PATH = DATA_PATH / "all_topics"


MODEL_OUT_PATH = DATA_PATH / "model_outputs"

CKPT_PATH = Path(__file__).parent.parent / "model_ckpts"

TOPIC_NAME_MAPPING = {
    "hk_protest": DATASET_TOPIC1_PATH,
    "al_shifa": DATASET_TOPIC2_PATH,
    "putin": DATASET_TOPIC3_PATH,
    "all_topics": DATASET_ALL_PATH
}