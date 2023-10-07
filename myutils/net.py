import os


def set_proxy(port: int=7890):
    os.environ["http_proxy"] = f"http://127.0.0.1:{port}"
    os.environ["https_proxy"] = f"http://127.0.0.1:{port}"

def unset_proxy():
    os.environ.pop("http_proxy")
    os.environ.pop("https_proxy")

