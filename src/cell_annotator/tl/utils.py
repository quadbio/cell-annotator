from openai import OpenAI
from pydantic import BaseModel

ResponseOutput = type[BaseModel]


def _query_openai(
    agent_description: str,
    instruction: str,
    model: str,
    other_messages: list | None = None,
    response_format: ResponseOutput | None = None,
    **kwargs,
) -> BaseModel:
    client = OpenAI()

    if other_messages is None:
        other_messages = []

    res = client.beta.chat.completions.parse(
        model=model,
        messages=[{"role": "system", "content": agent_description}, {"role": "user", "content": instruction}]
        + other_messages,
        response_format=response_format,
        **kwargs,
    )

    return res
