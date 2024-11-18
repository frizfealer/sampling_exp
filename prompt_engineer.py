DEFAULT_SYSTEM_MESSAGE = """
    You are a graduate student majoring in Math. 
    You solve problems carefully and systematically.
    You Break down your reasoning into clear steps.
    and you Show all your work and explain each step.
"""


def prepend_system_prompt(question, system):
    prompt = (
        "<|begin_of_text|>"
        "<|start_header_id|>system<|end_header_id|>"
        f"{system}"
        "<|start_header_id|>user<|end_header_id|>"
        f"{question}"
        "<|eot_id|>"
        "<|start_header_id|>assistant<|end_header_id|>"
    )
    return prompt
