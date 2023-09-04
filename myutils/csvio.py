import csv


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
