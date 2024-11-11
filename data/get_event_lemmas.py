from collections import Counter
import json
from data.file_path import TOPIC_NAME_MAPPING

RESERVED_LEMMAS = {
    "putin": {"election", "vote", 'war', 'win', 'protest'},
    "al_shifa": {"raid", "kill", "attack", "operation", "war", "use",
                 "include", "find", "strike", "displace", "target", "leave", "destroy", "call", "fight", "release",
                 "statement", "locate", "arrest", "evacuate", "eliminate", "detain", "wound", "carry", "shelter"},
    "hk_protest": {"protest", "use", "arrest", "include", "march", "attack", "extradition", "call", "demonstration",
                   "return", "storm", "violence", "allow", "riot", "charge", "leave", "break"}
}


def get_non_single_trigger_lemma(topic):
    # find the most frequent lemmas in this topic dataset
    topic_path = TOPIC_NAME_MAPPING[topic]
    parsed_json_path = topic_path / 'parsed.json'
    with open(parsed_json_path, "r") as f:
        articles = json.load(f)
    triggers = []
    for article_id in articles:
        for event in articles[article_id]["events"]:
            triggers.append(event['lemma'])
    trigger_lemma_counter = Counter(triggers)
    reserved_lemmas = RESERVED_LEMMAS[topic]
    non_single_lemmas = [p[0] for p in trigger_lemma_counter.most_common() if p[1] > 1 and p[0] not in reserved_lemmas]
    print(sum([p[1] for p in trigger_lemma_counter.most_common()]))
    return non_single_lemmas


if __name__ == '__main__':
    get_non_single_trigger_lemma("putin")
