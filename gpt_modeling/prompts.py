import inspect
from langchain_core.prompts import PromptTemplate, FewShotPromptTemplate, PipelinePromptTemplate

FULL_TOPIC = {
    "putin": "Putin's Election Win",
    "hk_protest": "Hong Kong Protest",
    "al_shifa": "Israel's Al-Shifa Hospital Raid",
}

TOPIC_INSTRUCT = {
    "putin": "Some events might reflect a defensive and negative stance, but they are supportive of Putin's perspective as it portrays him as a target of external forces, which could garner sympathy and support, thus the article is considered positive",
    "hk_protest": "Some events might reflect negative tone, but they are sympathetic to the protesters and against police or Chinese authorities, thus the article is considered positive",
    "al_shifa": "As long as article contain an event supports or justify the Israeli actions by referring to Hamas military operations or terrorists, it is considered positive "
}

full_prompt = PromptTemplate.from_template(
    inspect.cleandoc(
        """
    {instruct}

    {qa}
    """
    )
)

device_0_prompt = PromptTemplate.from_template(
    (
        "What is the article's stance towards {full_topic} ? Please answer as either positive, negative, or neutral in one word."
        " {topic_instruct}."
    ))

device_1_prompt = PromptTemplate.from_template(
    (
        "Each line below is the summary of an event mentioned in an article about \"{full_topic}\". "
        "Based on these event summaries, determine the article's attitude towards the topic \"{full_topic}\". "
        "{topic_instruct}."
        "Indicate whether the attitude is positive, negative, or neutral in one word."
    ))

device_2_prompt = PromptTemplate.from_template(
    (
        "Each line below is the summary of an event mentioned in an article about \"{full_topic}\". "
        "Based on these event summaries, determine the article's attitude towards the topic \"{full_topic}\". "
        "{topic_instruct}."
        "Indicate whether the attitude is positive, negative, or neutral in one word."
    ))

device_3_prompt = PromptTemplate.from_template(
    (
        "Each line below is the summary of a causal relation extracted in an article about \"{full_topic}\". "
        "Based on these causal relations, determine the article's attitude towards the topic \"{full_topic}\". "
        "Indicate whether the attitude is positive, negative, or neutral in one word."
    ))

# format the example prompt template for few-shot learning
example_prompt = PromptTemplate.from_template("Context: {context}\n{answer}")

# prepare the examples for the few-shot learning
examples = [
    {
        "context": "This is example context 1",
        "answer": "positive"
    },
    {
        "context": "This is example context 2",
        "answer": "negative"
    },
    {
        "context": "This is example context 3",
        "answer": "neutral"
    }
]


def construct_framing_prompt(*, device, num_shots: int = 0):
    if device == 0:
        instruct = device_0_prompt
        suffix = "{input}"
    elif device == 1:
        instruct = device_1_prompt
        suffix = "{input}"
    elif device == 2:
        instruct = device_2_prompt
        suffix = "{input}"
    elif device == 3:
        instruct = device_3_prompt
        suffix = "{input}"
    else:
        raise ValueError(f"Invalid device number: {device}")
    few_shot_prompt = FewShotPromptTemplate(
        examples=examples[:num_shots],
        example_prompt=example_prompt,
        suffix=suffix,  # add the context based on which the answer should be generated
        input_variables=["input"],
    )
    final_prompt = PipelinePromptTemplate(
        final_prompt=full_prompt,
        pipeline_prompts=[("instruct", instruct), ("qa", few_shot_prompt)])

    return final_prompt


if __name__ == '__main__':
    framing_qa_prompt = construct_framing_prompt(device=0)
    res = framing_qa_prompt.invoke({"input": "this is a test sentence",
                                    "full_topic": FULL_TOPIC["putin"], "topic_instruct": TOPIC_INSTRUCT["putin"]})
    print(res.to_string())
