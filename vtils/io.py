import csv
import json
from typing import List


def jsonload(path: str):
    with open(path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    return data

def jsondump(data, path: str):
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=4)

def jsonldump(data: List[dict], path: str) -> None:
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
    with open(path, 'w', encoding='utf-8', newline='') as f:
        writer = csv.writer(f)
        writer.writerows(data)
