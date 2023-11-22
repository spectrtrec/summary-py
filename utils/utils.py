import csv
import re
import json
from bs4 import BeautifulSoup
from typing import Dict, Tuple, List, Set


def capitalize_sentence(sentence: str) -> str:
    return ". ".join(map(lambda s: s.strip().capitalize(), sentence.split(".")))


def load_csv(file_path: str) -> Tuple[List[List[str]], List[str]]:
    rows = []
    with open(file_path, mode="r", encoding="utf-8") as csvfile:
        csv_reader = csv.reader(csvfile, delimiter=",")
        header = next(csv_reader)
        for row in csv_reader:
            rows.append(row)
    return rows, header


def load_json(file_path: str) -> List[Dict[str, str]]:
    with open(file_path, encoding="utf-8") as fh:
        data = json.load(fh)
    return data


def ria_parser(path) -> List[List[str]]:
    ria_list: List[List[str]] = []
    with open(path, "r", encoding="utf-8") as r:
        for line in r:
            data = json.loads(line.strip())
            title = data["title"]
            text = data["text"]
            clean_text = (
                BeautifulSoup(text, "html.parser")
                .text.replace("\xa0", " ")
                .replace("\n", " ")
            )
            if not clean_text or not title:
                continue
            ria_list.append(
                [capitalize_sentence(clean_text), capitalize_sentence(title)]
            )
    return ria_list


def _get_ngrams(n: int, text: List[str]) -> Set[Set[str]]:
    """Calcualtes n-grams.

    Args:
      n: which n-grams to calculate
      text: An array of tokens

    Returns:
      A set of n-grams
    """
    ngram_set = set()
    text_length = len(text)
    max_index_ngram_start = text_length - n
    for i in range(max_index_ngram_start + 1):
        ngram_set.add(tuple(text[i : i + n]))
    return ngram_set


def _get_word_ngrams(n: int, sentences: List[List[str]]) -> Set[Set[str]]:
    """Calculates word n-grams for multiple sentences."""
    assert len(sentences) > 0
    assert n > 0
    words = sum(sentences, [])
    return _get_ngrams(n, words)


def cal_rouge(
    evaluated_ngrams: Set[Set[str]], reference_ngrams: Set[Set[str]]
) -> Dict[str, float]:
    reference_count = len(reference_ngrams)
    evaluated_count = len(evaluated_ngrams)

    overlapping_ngrams = evaluated_ngrams.intersection(reference_ngrams)
    overlapping_count = len(overlapping_ngrams)

    if evaluated_count == 0:
        precision = 0.0
    else:
        precision = overlapping_count / evaluated_count

    if reference_count == 0:
        recall = 0.0
    else:
        recall = overlapping_count / reference_count

    f1_score = 2.0 * ((precision * recall) / (precision + recall + 1e-8))
    return {"f": f1_score, "p": precision, "r": recall}


def block_tri(c, p):
    tri_c = _get_ngrams(3, c.split())
    for s in p:
        tri_s = _get_ngrams(3, s.split())
        if len(tri_c.intersection(tri_s)) > 0:
            return True
    return False


def greedy_selection(
    doc_sent_list: List[List[str]],
    abstract_sent_list: List[List[str]],
    summary_size: int,
) -> List[int]:
    def _rouge_clean(s: str) -> str:
        return re.sub(r"[^ЁёА-я0-9 ]", "", s)

    max_rouge = 0.0
    abstract = sum(abstract_sent_list, [])
    abstract = _rouge_clean(" ".join(abstract)).split()
    sents = [_rouge_clean(" ".join(s)).split() for s in doc_sent_list]
    evaluated_1grams = [_get_word_ngrams(1, [sent]) for sent in sents]
    reference_1grams = _get_word_ngrams(1, [abstract])
    evaluated_2grams = [_get_word_ngrams(2, [sent]) for sent in sents]
    reference_2grams = _get_word_ngrams(2, [abstract])

    selected = []
    for _ in range(summary_size):
        cur_max_rouge = max_rouge
        cur_id = -1
        for i in range(len(sents)):
            if i in selected:
                continue
            c = selected + [i]
            candidates_1 = [evaluated_1grams[idx] for idx in c]
            candidates_1 = set.union(*map(set, candidates_1))
            candidates_2 = [evaluated_2grams[idx] for idx in c]
            candidates_2 = set.union(*map(set, candidates_2))
            rouge_1 = cal_rouge(candidates_1, reference_1grams)["f"]
            rouge_2 = cal_rouge(candidates_2, reference_2grams)["f"]
            rouge_score = rouge_1 + rouge_2
            if rouge_score > cur_max_rouge:
                cur_max_rouge = rouge_score
                cur_id = i
        if cur_id == -1:
            return selected
        selected.append(cur_id)
        max_rouge = cur_max_rouge

    return sorted(selected)
