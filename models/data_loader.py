import bisect

import pytorch_lightning as pl
import torch
from typing import List
from collections import defaultdict

from torch.utils.data import DataLoader, RandomSampler
from utils.typing_hint import SampleDict


class DataModule(pl.LightningDataModule):
    def __init__(self, args, train_dict=None, val_dict=None, test_dict=None):
        super(DataModule, self).__init__()
        self.save_hyperparameters(args)
        self._train_dataset = train_dict
        self._dev_dataset = val_dict
        self._test_dataset = test_dict

    def train_dataloader(self) -> DataLoader:
        return DataLoader(
            dataset=self._train_dataset,
            sampler=RandomSampler(self._train_dataset),
            batch_size=self.hparams.batch_size,
            collate_fn=self.prepare_sample,
            num_workers=self.hparams.loader_workers,
            drop_last=True,
        )

    def val_dataloader(self) -> DataLoader:
        return DataLoader(
            dataset=self._dev_dataset,
            batch_size=self.hparams.batch_size,
            collate_fn=self.prepare_sample,
            num_workers=self.hparams.loader_workers,
            drop_last=True,
        )

    def test_dataloader(self) -> DataLoader:
        return DataLoader(
            dataset=self._test_dataset,
            batch_size=self.hparams.batch_size,
            collate_fn=self.prepare_sample,
            num_workers=self.hparams.loader_workers,
            drop_last=True,
        )

    def prepare_sample(
        self, sample: List[SampleDict], is_test: bool = False,
    ) -> SampleDict:
        sample = self.collate_data(sample)
        src = sample["src"]
        tgt = [tgts[: self.hparams.max_tgt_len][:-1] + [2] for tgts in sample["tgt"]]
        src_sent_labels = sample["src_sent_labels"]
        segs = sample["segs"]
        if not self.hparams.use_interval:
            segs = [[0] * len(seg) for seg in segs]
        clss = sample["clss"]
        src_txt = sample["src_txt"]
        tgt_txt = sample["tgt_txt"]

        end_id = [source[-1] for source in src]
        src = [
            source[:-1][: self.hparams.max_pos - 1] + [end_id[i]]
            for i, source in enumerate(src)
        ]
        segs = [source[: self.hparams.max_pos] for source in segs]

        max_sent_id = [
            bisect.bisect_left(source, self.hparams.max_pos) for source in clss
        ]

        src_sent_labels = [
            source[: max_sent_id[i]] for i, source in enumerate(src_sent_labels)
        ]
        clss = [source[: max_sent_id[i]] for i, source in enumerate(clss)]

        src = torch.tensor(self._pad(src, 0))
        tgt = torch.tensor(self._pad(tgt, 0))

        segs = torch.tensor(self._pad(segs, 0))

        mask_src = torch.stack([~(source == 0) for source in src])
        mask_tgt = torch.stack([~(source == 0) for source in tgt])

        clss = torch.tensor(self._pad(clss, -1))
        src_sent_labels = torch.tensor(self._pad(src_sent_labels, 0))

        mask_cls = torch.stack([~(source == -1) for source in clss])
        clss = torch.tensor(
            [[0 if num == -1 else num for num in source] for source in clss]
        )
        return {
            "src": src,
            "tgt": tgt,
            "src_sent_labels": src_sent_labels,
            "segs": segs,
            "clss": clss,
            "mask_src": mask_src,
            "mask_cls": mask_cls,
            "mask_tgt": mask_tgt,
            "src_txt": src_txt,
            "tgt_txt": tgt_txt,
        }

    def _pad(self, data: List[List[int]], pad_id: int, width=-1) -> List[List[int]]:
        if width == -1:
            width = max(len(d) for d in data)
        rtn_data = [d + [pad_id] * (width - len(d)) for d in data]
        return rtn_data

    @staticmethod
    def collate_data(data_batch: List[SampleDict]) -> List[SampleDict]:
        collate_dict = defaultdict(list)
        for data in data_batch:
            collate_dict["src"].append(data["src"])
            collate_dict["tgt"].append(data["tgt"])
            collate_dict["src_sent_labels"].append(data["src_sent_labels"])
            collate_dict["segs"].append(data["segs"])
            collate_dict["clss"].append(data["clss"])
            collate_dict["src_txt"].append(data["src_txt"])
            collate_dict["tgt_txt"].append(data["tgt_txt"])
        return collate_dict
