import csv
import json


def jsonload(path: str):
    with open(path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    return data

def jsondump(data, path: str):
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=4)

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
