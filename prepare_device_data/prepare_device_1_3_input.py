import json
from typing import List

from data.file_path import TOPIC_NAME_MAPPING
from data.load_as_ecb_doc import prepare_ecb_docs
from collections import defaultdict
import click
import random

random.seed(42)


def load_cluster_result(topic: str):
    cluster2events = defaultdict(list)
    topic_path = TOPIC_NAME_MAPPING[topic]
    cluster_result_path = topic_path / "cluster_0.8_0.5.txt"
    with open(cluster_result_path, 'r') as f:
        pairs = [line.strip().split("\t") for line in f.readlines()]
    for event_id, cluster_id in pairs:
        cluster_id = cluster_id[1: -1]
        if cluster_id == "-1":
            cluster_id = event_id
        cluster2events[cluster_id].append(event_id)
    return cluster2events


def prepare_device_1_input_old(topic):
    docs = prepare_ecb_docs(topic)
    cluster2events = load_cluster_result(topic)
    out_f = open(TOPIC_NAME_MAPPING[topic] / "device_1_gpt_input.jsonl", "w")
    for i, c_id in enumerate(cluster2events):
        event_ids = cluster2events[c_id]
        if len(event_ids) < 3:
            sample_k = len(event_ids)
        else:
            sample_k = 3
        sample_event_ids = random.sample(event_ids, sample_k)
        contexts: List[List] = []
        for event_id in sample_event_ids:
            doc_id, event_info = event_id.split("#")
            sent_idx, start_offset, end_offset = [int(i) for i in event_info.split("_")]
            sent_idx -= 1
            doc = docs[doc_id]
            sentence = [p[0] for p in doc.sentences[sent_idx].tokens]
            sentence[start_offset: end_offset + 1] = [
                " ".join(sentence[start_offset: end_offset + 1]) + f"[Cluster-{i}]"]
            sentence = " ".join(sentence)
            contexts.append([event_id, sentence])
        out_f.write(json.dumps({"cluster_id": c_id, "contexts": contexts}) + "\n")


def prepare_device_1_input(topic):
    docs = prepare_ecb_docs(topic)
    cluster2events = load_cluster_result(topic)
    out_f = open(TOPIC_NAME_MAPPING[topic] / "device_1_gpt_input.jsonl", "w")
    for i, c_id in enumerate(cluster2events):
        event_ids = cluster2events[c_id]
        if len(event_ids) < 5:
            continue
        else:
            sample_k = 5
        sample_event_ids = random.sample(event_ids, sample_k)
        contexts: List[List] = []
        for event_id in sample_event_ids:
            doc_id, event_info = event_id.split("#")
            sent_idx, start_offset, end_offset = [int(i) for i in event_info.split("_")]
            sent_idx -= 1
            doc = docs[doc_id]
            sentence = [p[0] for p in doc.sentences[sent_idx].tokens]
            sentence[start_offset: end_offset + 1] = [
                " ".join(sentence[start_offset: end_offset + 1]) + f"[EVENT]"]
            sentence = " ".join(sentence)
            contexts.append([event_id, sentence])
        out_f.write(json.dumps({"cluster_id": c_id, "contexts": contexts}) + "\n")


def prepare_device_3_input(topic: str):
    frequent_clusters = set()
    with open(TOPIC_NAME_MAPPING[topic] / "device_1_gpt_input.jsonl", "r") as f:
        for line in f:
            data = json.loads(line)
            c_id = data["cluster_id"]
            frequent_clusters.add(c_id)
    cluster2events = load_cluster_result(topic)
    event2cluster = dict()  # event_id -> normalized cluster_id
    normalized_c_id2c_id = dict()  # normalized cluster_id -> cluster_id
    for i, c_id in enumerate(cluster2events):
        event_ids = cluster2events[c_id]
        normalized_c_id2c_id[i] = c_id
        for event_id in event_ids:
            event2cluster[event_id] = i
    out_f = open(TOPIC_NAME_MAPPING[topic] / "device_3_gpt_input.jsonl", "w")
    docs = prepare_ecb_docs(topic)
    for doc_id in docs:
        doc = docs[doc_id]
        processed_sents = []
        for sent_idx in doc.sentences:
            sent = doc.sentences[sent_idx]
            tokens: List[str] = [p[0] for p in sent.tokens]
            for event in sent.events:
                cluster_id = event2cluster[event.mention_id]
                if normalized_c_id2c_id[cluster_id] not in frequent_clusters:
                    continue
                tokens[event.start_offset] = tokens[event.start_offset] + f"[Cluster-{cluster_id}]"
            processed_sents.append(" ".join(tokens))
        out_f.write(json.dumps({"doc_id": doc_id, "sents": processed_sents}) + "\n")
    json.dump(normalized_c_id2c_id, open(TOPIC_NAME_MAPPING[topic] / "device_3_cid_mapping.json", "w"), indent=2)


@click.command()
@click.option("--topic", type=click.STRING, required=True)
def main(topic: str):
    prepare_device_1_input(topic)
    prepare_device_3_input(topic)


if __name__ == '__main__':
    main()
