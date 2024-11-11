from data.load_as_ecb_doc import prepare_ecb_docs
from data.file_path import TOPIC_NAME_MAPPING
import json
import string
from collections import defaultdict, Counter

import numpy as np
import scipy.stats

stopwords = {"i", "me", "my", "myself", "we", "our", "ours", "ourselves", "you", "your", "yours", "yourself",
             "yourselves", "he", "him", "his", "himself", "she", "her", "hers", "herself", "it", "its", "itself",
             "they", "them", "their", "theirs", "themselves", "what", "which", "who", "whom", "this", "that", "these",
             "those", "am", "is", "are", "was", "were", "be", "been", "being", "have", "has", "had", "having", "do",
             "does", "did", "doing", "a", "an", "the", "and", "but", "if", "or", "because", "as", "until", "while",
             "of", "at", "by", "for", "with", "about", "against", "between", "into", "through", "during", "before",
             "after", "above", "below", "to", "from", "up", "down", "in", "out", "on", "off", "over", "under", "again",
             "further", "then", "once", "here", "there", "when", "where", "why", "how", "all", "any", "both", "each",
             "few", "more", "most", "other", "some", "such", "no", "nor", "not", "only", "own", "same", "so", "than",
             "too", "very", "s", "t", "can", "will", "just", "don", "should", "now"}
stopwords.update(set(string.punctuation))


def jensen_shannon_distance(p, q):
    """
    method to compute the Jenson-Shannon Distance
    between two probability distributions
    """

    # convert the vectors into numpy arrays in case that they aren't
    p = np.array(p)
    q = np.array(q)

    # calculate m
    m = (p + q) / 2

    # compute Jensen Shannon Divergence
    divergence = (scipy.stats.entropy(p, m) + scipy.stats.entropy(q, m)) / 2

    # compute the Jensen Shannon Distance
    distance = np.sqrt(divergence)

    return distance


def device_0_sim(topic):
    docs = prepare_ecb_docs(topic)
    doc_splits = defaultdict(list)
    for split in ["train", "test"]:
        with open(TOPIC_NAME_MAPPING[topic] / f"{split}.jsonl", "r") as f:
            for line in f:
                data = json.loads(line)
                doc_id = data["uid"]
                if docs.get(doc_id):
                    doc_splits[split].append(docs.get(doc_id))
    train_bigrams = Counter()
    for doc in doc_splits["train"]:
        tokens = doc.tokens
        unigrams = [t.lower() for t in tokens if t.lower() not in stopwords]
        bigrams = [(tokens[i].lower(), tokens[i + 1].lower()) for i in range(len(tokens) - 1) if
                   tokens[i].lower() not in stopwords and tokens[i + 1].lower() not in stopwords]
        trigrams = [(tokens[i].lower(), tokens[i + 1].lower(), tokens[i + 2].lower()) for i in range(len(tokens) - 2) if
                    tokens[i].lower() not in stopwords and tokens[i + 1].lower() not in stopwords and tokens[
                        i + 2].lower() not in stopwords]
        train_bigrams.update(unigrams)

    test_bigrams = Counter()
    for doc in doc_splits["test"]:
        tokens = doc.tokens
        unigrams = [t.lower() for t in tokens if t.lower() not in stopwords]
        bigrams = [(tokens[i].lower(), tokens[i + 1].lower()) for i in range(len(tokens) - 1) if
                   tokens[i].lower() not in stopwords and tokens[i + 1].lower() not in stopwords]
        trigrams = [(tokens[i].lower(), tokens[i + 1].lower(), tokens[i + 2].lower()) for i in range(len(tokens) - 2) if
                    tokens[i].lower() not in stopwords and tokens[i + 1].lower() not in stopwords and tokens[
                        i + 2].lower() not in stopwords]
        test_bigrams.update(unigrams)
    vocab = set([k for k, v in train_bigrams.most_common(200)]) | set([k for k, v in test_bigrams.most_common(200)])
    vocab = [k for k in vocab]
    print(len(vocab))
    train_count = [train_bigrams.get(k, 0) for k in vocab]
    test_count = [test_bigrams.get(k, 0) for k in vocab]
    print(jensen_shannon_distance(train_count, test_count))


def device_1_sim(topic):
    docs = prepare_ecb_docs(topic)
    doc_splits = defaultdict(list)
    for split in ["train", "test"]:
        with open(TOPIC_NAME_MAPPING[topic] / f"{split}.jsonl", "r") as f:
            for line in f:
                data = json.loads(line)
                doc_id = data["uid"]
                if docs.get(doc_id):
                    doc_splits[split].append(docs.get(doc_id))
    train_events = Counter()
    for doc in doc_splits["train"]:
        events = doc.mentions.values()
        events = [e.lemma for e in events]
        train_events.update(events)

    test_events = Counter()
    for doc in doc_splits["test"]:
        events = doc.mentions.values()
        events = [e.lemma for e in events]
        test_events.update(events)

    vocab = set([k for k, v in train_events.most_common(200)]) | set([k for k, v in test_events.most_common(200)])
    vocab = [k for k in vocab]
    print(len(vocab))
    train_count = [train_events.get(k, 0) for k in vocab]
    test_count = [test_events.get(k, 0) for k in vocab]
    print(jensen_shannon_distance(train_count, test_count))


if __name__ == '__main__':
    device_0_sim('putin')
    device_0_sim('al_shifa')
    device_0_sim('hk_protest')

    device_1_sim('putin')
    device_1_sim('al_shifa')
    device_1_sim('hk_protest')
