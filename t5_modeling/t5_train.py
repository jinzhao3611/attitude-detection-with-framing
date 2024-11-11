import logging
import os
from dataclasses import dataclass, field
from typing import Optional
from data.file_path import CKPT_PATH, TOPIC_NAME_MAPPING
from pathlib import Path
from t5_modeling import PRETRAINED_MODEL
import click
import torch

from transformers import T5ForConditionalGeneration
from transformers import (
    HfArgumentParser,
    Trainer,
    TrainingArguments,
    set_seed,
)

logger = logging.getLogger(__name__)


@dataclass
class ModelArguments:
    """
    Arguments pertaining to which model/config/tokenizer we are going to fine-tune from.
    """

    model_name_or_path: str = field(
        metadata={"help": "Path to pretrained model or model identifier from huggingface.co/models"}
    )
    tokenizer_name: Optional[str] = field(
        default=None, metadata={"help": "Pretrained tokenizer name or path if not the same as model_name"}
    )
    cache_dir: Optional[str] = field(
        default=None, metadata={"help": "Where do you want to store the pretrained models downloaded from s3"}
    )


@dataclass
class DataTrainingArguments:
    """
    Arguments pertaining to what data we are going to input our model for training and eval.
    """
    train_file_path: Optional[str] = field(
        metadata={"help": "Path for cached train dataset"},
    )
    valid_file_path: Optional[str] = field(
        metadata={"help": "Path for cached valid dataset"},
    )
    max_len: Optional[int] = field(
        default=256,
        metadata={"help": "Max input length for the source text"},
    )
    target_max_len: Optional[int] = field(
        default=256,
        metadata={"help": "Max input length for the target text"},
    )


@click.command()
@click.option("--topic", type=click.STRING, required=True)
@click.option("--device", type=click.STRING, required=True)
def main(topic: str, device: str):
    # See all possible arguments in src/transformers/training_args.py
    # or by passing the --help flag to this script.
    # We now keep distinct sets of args, for a cleaner separation of concerns.

    parser = HfArgumentParser((ModelArguments, DataTrainingArguments, TrainingArguments))

    # we will load the arguments from a json file,
    # make sure you save the arguments in at ./args.json
    model_args, data_args, training_args = parser.parse_json_file(json_file=Path(__file__).parent.joinpath('args.json'))
    training_args.output_dir = str(CKPT_PATH / f"{topic}_device{device}_{PRETRAINED_MODEL.split('/')[-1]}_output")

    model_args.model_name_or_path = PRETRAINED_MODEL

    if (
            os.path.exists(training_args.output_dir)
            and os.listdir(training_args.output_dir)
            and training_args.do_train
            and not training_args.overwrite_output_dir
    ):
        raise ValueError(
            f"Output directory ({training_args.output_dir}) already exists and is not empty. "
            f"Use --overwrite_output_dir to overcome."
        )

    # Setup logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO if training_args.local_rank in [-1, 0] else logging.WARN,
        # filename="train.log"
    )
    logger.warning(
        "Process rank: %s, device: %s, n_gpu: %s, distributed training: %s, 16-bits training: %s",
        training_args.local_rank,
        training_args.device,
        training_args.n_gpu,
        bool(training_args.local_rank != -1),
        training_args.fp16,
    )
    logger.info("Training/evaluation parameters %s", training_args)

    # Set seed
    set_seed(training_args.seed)

    model = T5ForConditionalGeneration.from_pretrained(
        model_args.model_name_or_path,
        cache_dir=model_args.cache_dir,
    )

    # Get datasets
    print('loading data')
    topic_path = TOPIC_NAME_MAPPING[topic]
    data_args.train_file_path = f"t5_train_device_{device}.pt"
    data_args.valid_file_path = f"t5_test_device_{device}.pt"
    train_dataset = torch.load(str(topic_path / data_args.train_file_path))
    valid_dataset = torch.load(str(topic_path / data_args.valid_file_path))
    print('loading done')

    # Initialize our Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=valid_dataset,
    )

    # Training
    if training_args.do_train:
        trainer.args._n_gpu = 1
        trainer.train(
            model_path=model_args.model_name_or_path if os.path.isdir(model_args.model_name_or_path) else None
        )
        trainer.save_model()


if __name__ == '__main__':
    main()
