import os


def set_proxy(port: int=7890):
    os.environ["http_proxy"] = f"http://127.0.0.1:{port}"
    os.environ["https_proxy"] = f"http://127.0.0.1:{port}"

def unset_proxy():
    if "http_proxy" in os.environ:
        os.environ.pop("http_proxy")
    if "https_proxy" in os.environ:
        os.environ.pop("https_proxy")

