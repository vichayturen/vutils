import openai
from typing import Literal


optional_models = Literal['gpt-3.5-turbo', 'gpt-3.5-turbo-16k', 'gpt-4', 'gpt-4-32k']
def call_gpt(prompt: str, model: optional_models='gpt-3.5-turbo', api_key: str=None, api_base: str=None, **kwargs) -> str:
    if api_key is not None:
        openai.api_key = api_key
    if api_base is not None:
        openai.api_base = api_base
    completion = openai.ChatCompletion.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.01,
        **kwargs
    )
    return completion.choices[0].message.content
