import csv
from typing import Dict, List, Tuple

from multiprocess import Pool
from razdel import sentenize, tokenize


def tokenize_sentense(sentence: str) -> List[List[str]]:
    return [
        [_.text for _ in list(tokenize(sentenes))]
        for sentenes in [_.text for _ in list(sentenize(sentence))]
    ]


def format_text_to_dict(params: Tuple[str]) -> Dict[str, List[List[str]]]:
    src, tgt = params
    return {
        "src": tokenize_sentense(src),
        "tgt": tokenize_sentense(tgt),
    }


def build_bert_json(summary_data: List[List[str]]) -> List[Dict[str, List[List[str]]]]:
    a_lst = [(sentense[0], sentense[1]) for sentense in summary_data]
    pool = Pool(8)
    dataset = []
    for d in pool.imap_unordered(format_text_to_dict, a_lst):
        dataset.append(d)
    pool.close()
    pool.join()
    return dataset


def load_csv(file_path: str) -> Tuple[List[List[str]], List[str]]:
    rows = []
    with open(file_path, mode="r", encoding="utf-8") as csvfile:
        csv_reader = csv.reader(csvfile, delimiter=",")
        header = next(csv_reader)
        for row in csv_reader:
            rows.append(row)
    return rows, header


if __name__ == "__main__":
    data, header = load_csv("/home/anton/sk-searcher/notebooks/bert_summary.csv")
    fg = build_bert_json(data)
