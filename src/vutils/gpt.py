import openai
import argparse
from typing import Literal


optional_models = Literal['gpt-3.5-turbo', 'gpt-3.5-turbo-16k', 'gpt-4', 'gpt-4-32k']


def call_gpt(prompt: str, api_key: str=None, api_base: str=None, model: optional_models='gpt-3.5-turbo', **kwargs) -> str:
    """
    调用chatgpt
    """
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


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        usage="python gpt.py --key <key>",
        description="test openai api_key"
    )
    parser.add_argument("key", type=str)
    args = parser.parse_args()
    from .net import set_proxy
    set_proxy()
    call_gpt("你好", api_key=args['key'])
