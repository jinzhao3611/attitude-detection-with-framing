from bert_modeling import PRETRAINED_MODEL
from data.load_data import load_jsonl_data
from bert_modeling.eval_metrics import compute_metrics
from data.file_path import CKPT_PATH
from transformers import AutoTokenizer
from transformers import DataCollatorWithPadding
from transformers import AutoModelForSequenceClassification, TrainingArguments, Trainer
import click

model_path = PRETRAINED_MODEL

id2label = {0: "pos", 1: "neg", 2: "neutral"}
label2id = {"pos": 0, "neg": 1, "neutral": 2}

tokenizer = AutoTokenizer.from_pretrained(model_path)
data_collator = DataCollatorWithPadding(tokenizer=tokenizer)


def preprocess_function(examples):
    examples = [" ".join(e.split()) for e in examples["input"]]
    return tokenizer(examples, max_length=512, truncation=True, padding="max_length")


@click.command()
@click.option("--topic", type=click.STRING, required=True)
@click.option("--device", type=click.STRING, required=True)
def main(topic: str, device: str):
    dataset = load_jsonl_data(topic=topic, device=int(device))
    print("train input: ", dataset["train"][0]["input"])
    print("test input: ", dataset["test"][0]["input"])
    tokenized_dataset = dataset.map(preprocess_function, batched=True)

    model = AutoModelForSequenceClassification.from_pretrained(
        model_path, num_labels=3, id2label=id2label, label2id=label2id
    )

    model_out_path = CKPT_PATH / f"{topic}_device{device}_{model_path.split('/')[-1]}_output"

    training_args = TrainingArguments(
        output_dir=model_out_path,
        learning_rate=2e-5,
        per_device_train_batch_size=16,
        per_device_eval_batch_size=16,
        gradient_accumulation_steps=1,
        num_train_epochs=6,
        save_total_limit=2,
        weight_decay=0.01,
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        push_to_hub=False,
        use_mps_device=False,
    )
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_dataset["train"],
        eval_dataset=tokenized_dataset["test"],
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
    )

    trainer.train()


if __name__ == '__main__':
    main()
