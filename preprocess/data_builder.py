import json
from typing import Dict, List

from razdel import sentenize, tokenize
from transformers import BertTokenizer
from utils.typing_hint import SampleDict
from utils.utils import greedy_selection


def tokenize_sentense(sentence: str) -> List[List[str]]:
    return [
        [_.text for _ in list(tokenize(sentenes))]
        for sentenes in [_.text for _ in list(sentenize(sentence))]
    ]


def create_json(
    summary_data: List[List[str]], save_path: str, save_json: bool = True
) -> List[Dict[str, str]]:
    summary_json = []
    for sentense in summary_data:
        summary_json.append(
            {
                "src": tokenize_sentense(sentense[0]),
                "tgt": tokenize_sentense(sentense[1]),
            }
        )
    if save_json:
        with open(save_path, "w") as save:
            save.write(json.dumps(summary_json, ensure_ascii=False))
    return summary_json


class BertDatabuilder(object):
    def __init__(self):
        self.tokenizer = BertTokenizer.from_pretrained(
            "sberbank-ai/ruBert-base", do_lower_case=True
        )

        self.sep_token = "[SEP]"
        self.cls_token = "[CLS]"
        self.pad_token = "[PAD]"
        self.tgt_bos = "[unused3]"
        self.tgt_eos = "[unused1]"
        self.tgt_sent_split = "[unused2]"
        self.sep_vid = self.tokenizer.vocab[self.sep_token]
        self.cls_vid = self.tokenizer.vocab[self.cls_token]
        self.pad_vid = self.tokenizer.vocab[self.pad_token]

    def get_segments(self, src_subtoken_idxs: List[int]) -> List[int]:
        _segs = [-1] + [i for i, t in enumerate(src_subtoken_idxs) if t == self.sep_vid]
        segs = [_segs[i] - _segs[i - 1] for i in range(1, len(_segs))]
        segments_ids = []

        for i, s in enumerate(segs):
            if i % 2 == 0:
                segments_ids += s * [0]
            else:
                segments_ids += s * [1]

        return segments_ids

    def get_sent_labels(self, source, target):
        sent_labels = greedy_selection(source, target, 3)
        idxs = [i for i, _ in enumerate(source)]

        _sent_labels = [0] * len(source)
        for l in sent_labels:
            _sent_labels[l] = 1
        sent_labels = [_sent_labels[i] for i in idxs]

        return sent_labels

    def preprocess(self, data: List[Dict[str, str]]) -> SampleDict:
        datasets = []
        for d in data:
            src, tgt = d["src"], d["tgt"]

            src_txt = [
                self.cls_token + " " + " ".join(sent) + " " + self.sep_token
                for sent in src
            ]
            text = " ".join(src_txt)

            src_subtokens = self.tokenizer.tokenize(text)
            
            src_subtoken_idxs = self.tokenizer.convert_tokens_to_ids(src_subtokens)

            segments_ids = self.get_segments(src_subtoken_idxs)
            cls_ids = [i for i, t in enumerate(src_subtoken_idxs) if t == self.cls_vid]

            sent_labels = self.get_sent_labels(src, tgt)
            sent_labels = sent_labels[: len(cls_ids)]

            tgt_subtokens_str = (
                "[unused3] "
                + " [unused2] ".join(
                    [" ".join(self.tokenizer.tokenize(" ".join(tt),)) for tt in tgt]
                )
                + " [unused1]"
            )
            tgt_subtoken = tgt_subtokens_str.split()

            tgt_subtoken_idxs = self.tokenizer.convert_tokens_to_ids(tgt_subtoken)

            tgt_txt = "<q>".join([" ".join(tt) for tt in tgt])

            b_data_dict = {
                "src": src_subtoken_idxs,
                "tgt": tgt_subtoken_idxs,
                "src_sent_labels": sent_labels,
                "segs": segments_ids,
                "clss": cls_ids,
                "src_txt": src_txt,
                "tgt_txt": tgt_txt,
            }
            datasets.append(b_data_dict)
        return datasets
