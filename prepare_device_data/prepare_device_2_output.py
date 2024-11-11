import json

from data.load_as_ecb_doc import prepare_ecb_docs
from data.file_path import TOPIC_NAME_MAPPING
import click


def parse_t5_srl_output(output_str: str):
    arg_dict = dict()
    if not output_str.strip():
        return arg_dict
    for arg in output_str.split("|"):
        try:
            arg = arg.strip()
            arg_type, arg_val = arg.split(":", 1)
            arg_dict[arg_type.strip()] = arg_val.strip()
        except:
            continue
            # print("Skipping arg: ", arg)
    return arg_dict


def load_srl(topic: str):
    srl_file_path = TOPIC_NAME_MAPPING[topic] / "srl.jsonl"
    event_id2srl = dict()
    with open(srl_file_path, 'r') as f:
        for line in f:
            srl_dict = json.loads(line)
            for event_id, srl_lst in srl_dict.items():
                args = parse_t5_srl_output(srl_lst[0])
                event_id2srl[event_id] = args
    return event_id2srl


@click.command()
@click.option("--topic", type=click.STRING, required=True)
def load_srl2events(topic: str):
    docs = prepare_ecb_docs(topic)
    event_id2srl = load_srl(topic)
    out_f = open(TOPIC_NAME_MAPPING[topic] / "device_2_output.jsonl", "w")
    for doc_id in docs:
        all_args = []
        doc = docs[doc_id]
        for i in doc.mentions:
            mention = doc.mentions[i]
            args = event_id2srl[mention.mention_id]
            all_args.append([mention.mention_id, mention.lemma, args])
        out_f.write(json.dumps({"doc_id": doc_id, "args": all_args}) + "\n")


if __name__ == '__main__':
    load_srl2events()
