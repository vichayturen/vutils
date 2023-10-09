import openai
from typing import Literal


optional_models = Literal['gpt-3.5-turbo', 'gpt-3.5-turbo-16k', 'gpt-4', 'gpt-4-32k']
def call_gpt(prompt: str, api_key: str, model: optional_models='gpt-3.5-turbo', **kwargs) -> str:
    openai.api_key = api_key
    completion = openai.ChatCompletion.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        **kwargs
    )
    return completion.choices[0].message.content
