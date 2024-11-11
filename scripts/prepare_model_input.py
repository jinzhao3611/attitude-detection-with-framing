import json
import re

from data.file_path import TOPIC_NAME_MAPPING
from data.load_as_ecb_doc import prepare_ecb_docs
from prepare_device_data.prepare_device_1_3_input import load_cluster_result
from data.get_event_lemmas import RESERVED_LEMMAS


def load_cid2descriptor(topic: str):
    c_id2descriptor = dict()
    with open(TOPIC_NAME_MAPPING[topic] / "device_1_gpt_output.jsonl", "r") as f:
        for line in f:
            data = json.loads(line)
            c_id = data["cluster_id"]
            c_id2descriptor[c_id] = data["responses"]
    return c_id2descriptor


def load_manual_descriptors(topic: str):
    c_id2descriptor = json.load(open(TOPIC_NAME_MAPPING[topic] / "manual_descriptors.json", "r"))
    c_id2descriptor = {c: c_id2descriptor[c][0] for c in c_id2descriptor}
    return c_id2descriptor


def generate_device1_jsonl(topic: str):
    cluster2events = load_cluster_result(topic)
    event2cluster = dict()  # event_id -> cluster_id
    for i, c_id in enumerate(cluster2events):
        event_ids = cluster2events[c_id]
        for event_id in event_ids:
            event2cluster[event_id] = c_id

    # c_id2descriptor = load_cid2descriptor(topic)
    c_id2descriptor = load_manual_descriptors(topic)

    docs = prepare_ecb_docs(topic)
    for split in ["train", "test"]:
        out_f = open(TOPIC_NAME_MAPPING[topic] / f"{split}_device_1.jsonl", "w")
        skipped = 0
        all_mentions = 0
        with open(TOPIC_NAME_MAPPING[topic] / f"{split}.jsonl", "r") as f:
            for line in f:
                data = json.loads(line)
                doc_id = data["uid"]
                try:
                    doc = docs[doc_id]
                except KeyError:
                    continue
                mentions = doc.mentions
                all_descriptors = []
                seen_clusters = set()
                for mention in mentions.values():
                    m_id = mention.mention_id
                    c_id = event2cluster[m_id]
                    all_mentions += 1
                    if c_id in seen_clusters:
                        continue
                    descriptor = c_id2descriptor.get(c_id)
                    if descriptor is None or c_id in RESERVED_LEMMAS[topic]:
                        skipped += 1
                        continue
                    seen_clusters.add(c_id)
                    if not descriptor.endswith("."):
                        descriptor = descriptor + "."
                    all_descriptors.append(descriptor)
                descriptors_str = "\n".join(all_descriptors)
                data["input"] = descriptors_str
                out_f.write(json.dumps(data) + "\n")
            print(f"skipped {skipped} events")
            print(f"total mentions: {all_mentions}")
        out_f.close()


def generate_device2_jsonl(topic: str):
    cluster2events = load_cluster_result(topic)
    event2cluster = dict()  # event_id -> cluster_id
    for i, c_id in enumerate(cluster2events):
        event_ids = cluster2events[c_id]
        for event_id in event_ids:
            event2cluster[event_id] = c_id
    # c_id2descriptor = load_manual_descriptors(topic)
    c_id2descriptor = load_cid2descriptor(topic)


    arg_types = ["ARG-0", "ARG-1", "ARG-2", "ARG-3", "ARG-4"]
    doc_id2context = dict()
    with open(TOPIC_NAME_MAPPING[topic] / "device_2_output.jsonl", "r") as f:
        for line in f:
            data = json.loads(line)
            doc_id = data["doc_id"]
            doc_context = []
            for m_id, lemma, args in data["args"]:
                c_id = event2cluster[m_id]
                if c_id in c_id2descriptor:
                    context = [args.get(arg_types[0], ""), lemma, args.get(arg_types[1], ""), args.get(arg_types[2], ""),
                               args.get(arg_types[3], ""), args.get(arg_types[4], "")]
                    if context[0] and context[2]:
                        context = [c for c in context if c]
                        context_str = " ".join(context) + "."
                        doc_context.append(context_str)
            doc_id2context[doc_id] = " ".join(doc_context)

        for split in ["train", "test"]:
            out_f = open(TOPIC_NAME_MAPPING[topic] / f"{split}_device_2.jsonl", "w")
            with open(TOPIC_NAME_MAPPING[topic] / f"{split}.jsonl", "r") as f:
                for line in f:
                    data = json.loads(line)
                    doc_id = data["uid"]
                    try:
                        context = doc_id2context[doc_id]
                    except KeyError:
                        continue
                    data["input"] = context
                    out_f.write(json.dumps(data) + "\n")
            out_f.close()


def generate_device3_jsonl(topic: str):
    c_id_mapping = json.load(open(TOPIC_NAME_MAPPING[topic] / "device_3_cid_mapping.json", "r"))
    c_id2descriptor = load_cid2descriptor(topic)
    doc_id2rel_str = dict()

    with open(TOPIC_NAME_MAPPING[topic] / "device_3_gpt_output.jsonl", "r") as f:
        for line in f:
            data = json.loads(line)
            doc_id = data["doc_id"]
            relations = data["response"]
            rel_pattern = r"\[Cluster-\d+\] -> \[Cluster-\d+\]"
            rels = re.findall(rel_pattern, relations)
            all_rel_strs = []
            for rel in rels:
                rel_id1, rel_id2 = re.findall(r"\d+", rel)
                print(rel_id1, rel_id2)
                c_id1 = c_id_mapping[rel_id1]
                c_id2 = c_id_mapping[rel_id2]
                print(c_id1, c_id2)
                des1 = c_id2descriptor[c_id1]  # TODO: edit later
                if not des1.endswith("."):
                    des1 = des1 + "."
                des2 = c_id2descriptor[c_id2]
                rel_str = f"{des1} It leads to {des2}"
                all_rel_strs.append(rel_str)
            doc_id2rel_str[doc_id] = "\n".join(all_rel_strs)
    for split in ["train", "test"]:
        out_f = open(TOPIC_NAME_MAPPING[topic] / f"{split}_device_3.jsonl", "w")
        with open(TOPIC_NAME_MAPPING[topic] / f"{split}.jsonl", "r") as f:
            for line in f:
                data = json.loads(line)
                doc_id = data["uid"]
                try:
                    context = doc_id2rel_str[doc_id]
                except KeyError:
                    continue
                data["input"] = context
                out_f.write(json.dumps(data) + "\n")
        out_f.close()


if __name__ == '__main__':
    generate_device1_jsonl("putin")
    generate_device1_jsonl("al_shifa")
    generate_device1_jsonl("hk_protest")

    generate_device2_jsonl("putin")
    generate_device2_jsonl("al_shifa")
    generate_device2_jsonl("hk_protest")

    generate_device3_jsonl("putin")
    generate_device3_jsonl("al_shifa")
    generate_device3_jsonl("hk_protest")
