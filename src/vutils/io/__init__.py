import os
import csv
import json
from typing import List


def jsonload(path: str):
    with open(path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    return data


def jsondump(data, path: str):
    pre_dir, file = os.path.split(path)
    if pre_dir != "" and not os.path.exists(pre_dir):
        os.makedirs(pre_dir)
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=4)


def jsonldump(data: List[dict], path: str) -> None:
    pre_dir, file = os.path.split(path)
    if pre_dir != "" and not os.path.exists(pre_dir):
        os.makedirs(pre_dir)
    jsonlines = []
    for d in data:
        string = json.dumps(d, ensure_ascii=False)
        jsonlines.append(string+'\n')
    with open(path, 'w', encoding='utf-8') as f:
        f.writelines(jsonlines)


def jsonlload(path: str) -> list:
    with open(path, 'r', encoding='utf-8') as f:
        data = f.readlines()
    for i, string in enumerate(data):
        data[i] = json.loads(string)
    return data


def txtdump(data, path: str):
    pre_dir, file = os.path.split(path)
    if pre_dir != "" and not os.path.exists(pre_dir):
        os.makedirs(pre_dir)
    with open(path, 'w', encoding='utf-8') as f:
        f.write(data)


def txtload(path: str) -> str:
    with open(path, 'r', encoding='utf-8') as f:
        data = f.read()
    return data


def csvload(path: str) -> list:
    with open(path, 'r', encoding='utf-8', newline='') as f:
        reader = csv.reader(f)
        data = []
        for row in reader:
            data.append(row)
    return data


def csvdump(data: list, path: str) -> None:
    pre_dir, file = os.path.split(path)
    if pre_dir != "" and not os.path.exists(pre_dir):
        os.makedirs(pre_dir)
    with open(path, 'w', encoding='utf-8', newline='') as f:
        writer = csv.writer(f)
        writer.writerows(data)


def add_tail(path: str, tail: str) -> str:
    """
    给路径字符串在扩展名之前添加一个尾巴
    """
    paths = path.split('.')
    paths = paths[:-1] + [tail] + paths[-1]
    return ''.join(paths)


def get_extension(path: str) -> str:
    """
    获取路径的扩展名
    """
    return path.split('.')[-1]
