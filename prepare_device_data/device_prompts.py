import inspect
from langchain_core.prompts import PromptTemplate, PipelinePromptTemplate

device_prompt = PromptTemplate.from_template(
    inspect.cleandoc(
        """
    {instruct}

    {context}
    """
    )
)

device1_instruction_old = PromptTemplate.from_template(
    (
        "Please provide a concise, neutral-toned descriptor for the following event mention, "
        "where the trigger word is indicated with [Cluster-\\d]. The descriptor should focus only "
        "on the action and the main participants:"
    ))

device1_instruction = PromptTemplate.from_template(
    (
        "Below is showing a list of mentions of the same event in different sentence contexts."
        "Each event mention is tagged by appending [EVENT] to it."
        "Please provide a single phrase that summarizes the event based on the contexts."
        "The phrase should be concise and neutral-toned, focusing only on the event and its participants."
    ))

device3_instruction = PromptTemplate.from_template(
    (
        "Please analyze the following event mentions, where each event trigger is tagged with "
        "[Cluster-\\d]. Extract and list all causal relations among the events, using the format "
        "[Cluster-\\d] -> [Cluster-\\d] to indicate the direction of the causal relationships."
    ))


def construct_device1_prompt_old(context_str: str):
    final_prompt = PipelinePromptTemplate(
        final_prompt=device_prompt,
        pipeline_prompts=[("instruct", device1_instruction)]
    )
    final_prompt = final_prompt.invoke({"context": context_str})
    return final_prompt


def construct_device1_prompt(context_str: str):
    final_prompt = PipelinePromptTemplate(
        final_prompt=device_prompt,
        pipeline_prompts=[("instruct", device1_instruction)]
    )
    final_prompt = final_prompt.invoke({"context": context_str})
    return final_prompt


def construct_device3_prompt(context_str: str):
    final_prompt = PipelinePromptTemplate(
        final_prompt=device_prompt,
        pipeline_prompts=[("instruct", device3_instruction)]
    )
    final_prompt = final_prompt.invoke({"context": context_str})
    return final_prompt


if __name__ == '__main__':
    context = "This is an example device-1 context."
    device1_qa_prompt = construct_device1_prompt(context)
    context = "This is an example device-3 context."
    device3_qa_prompt = construct_device3_prompt(context)
