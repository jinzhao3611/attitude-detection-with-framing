import json
import torch
from transformers import GenerationConfig, T5ForConditionalGeneration, T5Tokenizer
from data.load_as_ecb_doc import prepare_ecb_docs
from data.file_path import TOPIC_NAME_MAPPING
from typing import Any, List
import click

device = torch.device('cuda')


def format_srl_input(event_str: str, sentence: str):
    return f"SRL for [{event_str}]: {sentence}"


def load_event_context(topic: str):
    all_events_contexts = []
    docs = prepare_ecb_docs(topic)
    for doc_id in docs:
        doc = docs[doc_id]
        for i, mention in doc.mentions.items():
            sentence = [t[0] for t in doc.sentences[mention.sen_id].tokens]
            event_str = " ".join(sentence[mention.start_offset: mention.end_offset + 1])
            sentence[mention.start_offset: mention.end_offset + 1] = [f"[{event_str}]"]
            all_events_contexts.append((mention.mention_id, format_srl_input(event_str, " ".join(sentence))))
    return all_events_contexts


def predict(tokenizer: Any, model: Any, input_strs: List[str]):
    input_encodings = tokenizer(input_strs, padding="max_length", truncation=True, max_length=256, return_tensors="pt")
    input_encodings.to(device)
    output_str = model.generate(**input_encodings,
                                generation_config=generation_config)
    pred_answers = tokenizer.batch_decode(output_str, skip_special_tokens=True)
    return pred_answers


@click.command()
@click.option("--topic", type=click.STRING, required=True)
def load_event_context_and_write_pred(tokenizer: Any, model: Any, topic: str):
    contexts = load_event_context(topic)
    batch = 40
    out_f = open(TOPIC_NAME_MAPPING[topic] / "srl.jsonl", "w")

    for i in range(0, len(contexts), batch):
        input_strs = [c[1] for c in contexts[i: i + batch]]
        m_ids = [c[0] for c in contexts[i: i + batch]]
        output_strs = predict(tokenizer, model, input_strs)
        for m_id, input_str, output_str in zip(m_ids, input_strs, output_strs):
            out_f.write(json.dumps({m_id: [output_str, input_str]}) + "\n")
        print(i)


if __name__ == '__main__':
    model_name = "cu-kairos/propbank_srl_seq2seq_t5_large"

    model = T5ForConditionalGeneration.from_pretrained(model_name)
    model.to(device)
    tokenizer = T5Tokenizer.from_pretrained(model_name)
    generation_config = GenerationConfig.from_pretrained(model_name)
    load_event_context_and_write_pred(tokenizer=tokenizer, model=model)
