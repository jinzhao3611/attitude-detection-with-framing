import json
from data.file_path import TOPIC_NAME_MAPPING


class ECBDoc:

    def __init__(self):
        self.tokens = []
        self.sentences = {}
        self.topic = -1
        self.doc_id = 0
        self.doc_name = ''
        self.mentions = {}
        self.mention2token = {}
        self.mentions_type = {}
        self.token2mention = {}
        self.token2sentence = {}
        self.coref_mention = {}
        self.mention2cluster_id = {}
        self.cluster_head = []
        self.cluster_type = {}
        self.cluster_instance_id = {}
        self.clusters = {}

    def set_doc_info(self, doc_id, name):
        self.doc_id = doc_id
        self.doc_name = name


class Sentence:

    def __init__(self, topic, doc_id):
        self.topic = topic
        self.doc_id = doc_id
        self.tokens = []
        self.events = []
        self.entities = []


class Event:

    def __init__(self, topic, doc_id, sen_id, start_offset, end_offset, token_start_idx, token_end_idx):
        self.topic = topic
        self.doc_id = doc_id
        self.sen_id = sen_id
        self.start_offset = start_offset
        self.end_offset = end_offset
        self.token_start_idx = token_start_idx
        self.token_end_idx = token_end_idx

        self.arg0 = (-1, -1)
        self.arg1 = (-1, -1)
        self.time = (-1, -1)
        self.loc = (-1, -1)

        self.gold_cluster = '_'.join(['Singleton', doc_id, str(token_start_idx),
                                      str(token_end_idx)])  # Will be overwritten if belongs to a cluster
        self.cd_coref_chain = -1

        self.mention_id = self.doc_id + "#" + '_'.join(
            [str(sen_id + 1), str(self.start_offset),
             str(self.end_offset)])  # adapted to the framing task naming system

        self.lemma = None


def load_parsed_json(json_file: str):
    with open(json_file, 'r') as f:
        data = json.load(f)
    return data


def prepare_ecb_docs(topic: str):
    topic_path = TOPIC_NAME_MAPPING[topic]
    parsed_json_path = topic_path / 'parsed.json'
    docs = load_parsed_json(parsed_json_path)
    all_docs = dict()
    for doc_id in docs:
        doc_dict = docs[doc_id]

        doc = ECBDoc()
        doc.set_doc_info(doc_id, doc_id)  # set doc_id and doc_name the same

        global_tok_idx = 0
        for i, sent in enumerate(doc_dict["sentences"]):
            doc.sentences[i] = Sentence(topic, doc.doc_name)
            for j, token in enumerate(sent):
                doc.tokens.append(token)
                doc.sentences[i].tokens.append((token, j))
                doc.token2sentence[global_tok_idx] = (i, j)
                global_tok_idx += 1
        token2sentence_inv = {v: k for k, v in doc.token2sentence.items()}
        mentions = doc_dict["events"]
        for mention_id, mention in enumerate(mentions):
            doc.mention2token[mention_id] = []
            sent_idx, start_offset, end_offset = [int(i) for i in mention["event_loc"].split('_')]
            sent_idx -= 1  # make it 0-indexed
            for i in range(start_offset, end_offset + 1):
                t_id = token2sentence_inv[(sent_idx, i)]
                doc.mention2token[mention_id].append(t_id)
                doc.token2mention[t_id] = mention_id
            doc.mentions_type[mention_id] = "event"  # for now, all mentions are events
            event = Event(topic, doc.doc_name, sent_idx, start_offset, end_offset, doc.mention2token[mention_id][0],
                          doc.mention2token[mention_id][-1])
            event.lemma = mention["lemma"]
            doc.sentences[sent_idx].events.append(event)
            doc.mentions[mention_id] = event
        all_docs[doc_id] = doc
    return all_docs


if __name__ == '__main__':
    all_docs = prepare_ecb_docs('putin')
    print(all_docs["00_08_9"].sentences[1].events[0].__dict__)
