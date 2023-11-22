import bisect
from omegaconf import DictConfig
import pytorch_lightning as pl
import torch
from typing import List, Union, MutableMapping, Optional, Any
from collections import defaultdict

from torch.utils.data import DataLoader, RandomSampler  # type: ignore
from utils.typing_hint import SampleDict
from pytorch_lightning.utilities.parsing import AttributeDict
from torch.utils.data.dataset import Dataset


class DataModule(pl.LightningDataModule):
    def __init__(
        self,
        args: Union[DictConfig, Union[AttributeDict, MutableMapping]],
        train_dict: Optional[Dataset]=None,
        val_dict: Optional[Dataset]=None,
        test_dict: Optional[Dataset]=None,
    ) -> None:
        super(DataModule, self).__init__()
        self.save_hyperparameters(args)
        self._train_dataset: Optional[Dataset] = train_dict
        self._dev_dataset: Optional[Dataset] = val_dict
        self._test_dataset: Optional[Dataset] = test_dict

    def train_dataloader(self) -> DataLoader:
        return DataLoader(
            dataset=self._train_dataset,  # type: ignore
            sampler=RandomSampler(self._train_dataset),  # type: ignore
            batch_size=self.hparams.batch_size,  # type: ignore
            collate_fn=self.prepare_sample,
            num_workers=self.hparams.loader_workers,  # type: ignore
            drop_last=True,
        )

    def val_dataloader(self) -> DataLoader:
        return DataLoader(
            dataset=self._dev_dataset,  # type: ignore
            batch_size=self.hparams.batch_size,  # type: ignore
            collate_fn=self.prepare_sample,
            num_workers=self.hparams.loader_workers,  # type: ignore
            drop_last=True,
        )

    def test_dataloader(self) -> DataLoader:
        return DataLoader(
            dataset=self._test_dataset,  # type: ignore
            batch_size=self.hparams.batch_size,  # type: ignore
            collate_fn=self.prepare_sample,
            num_workers=self.hparams.loader_workers,  # type: ignore
            drop_last=True,
        )

    def prepare_sample(
        self,
        sample: List[SampleDict],
        is_test: bool = False,
    ) -> SampleDict:
        collate_sample: SampleDict = self.collate_data(sample)
        src:List[List[int]] = collate_sample["src"] # type: ignore
        tgt:List[List[int]] = [
            tgts[: self.hparams.max_tgt_len][:-1] + [2] # type: ignore
            for tgts in collate_sample["tgt"] 
        ]
        src_sent_labels:List[List[int]] = collate_sample["src_sent_labels"] # type: ignore
        segs:List[List[int]] = collate_sample["segs"]  # type: ignore
        if not self.hparams.use_interval: # type: ignore
            segs = [[0] * len(seg) for seg in segs]
        clss:List[List[int]] = collate_sample["clss"]  # type: ignore
        src_txt:List[List[str]] = collate_sample["src_txt"] # type: ignore
        tgt_txt:List[List[str]] = collate_sample["tgt_txt"] # type: ignore

        end_id: list[int] = [source[-1] for source in src]
        src = [
            source[:-1][: self.hparams.max_pos - 1] + [end_id[i]] # type: ignore
            for i, source in enumerate(src)
        ] 
        segs = [source[: self.hparams.max_pos] for source in segs] # type: ignore

        max_sent_id: list[int] = [
            bisect.bisect_left(source, self.hparams.max_pos) for source in clss # type: ignore
        ]

        src_sent_labels = [
            source[: max_sent_id[i]] for i, source in enumerate(src_sent_labels)
        ]
        clss = [source[: max_sent_id[i]] for i, source in enumerate(clss)]

        src = self._pad(src, 0) # type: ignore
        tgt = self._pad(tgt, 0) # type: ignore

        segs = self._pad(segs, 0) # type: ignore

        mask_src = torch.stack([~(source == 0) for source in src]) # type: ignore
        mask_tgt = torch.stack([~(source == 0) for source in tgt]) # type: ignore

        clss = self._pad(clss, -1) # type: ignore
        src_sent_labels = self._pad(src_sent_labels, 0) # type: ignore

        mask_cls = torch.stack([~(source == -1) for source in clss]) # type: ignore
        clss = torch.tensor(
            [[0 if num == -1 else num for num in source] for source in clss]
        ) # type: ignore
        return SampleDict(
            {
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
            } # type: ignore
        )

    def _pad(self, data: List[List[int]], pad_id: int, width=-1) -> torch.Tensor:
        return torch.tensor((
            [d + [pad_id] * (max(len(d) for d in data) - len(d)) for d in data]
            if width == -1
            else [d + [pad_id] * (width - len(d)) for d in data]
        ))

    @staticmethod
    def collate_data(data_batch: List[SampleDict]) -> SampleDict:
        collate_dict: defaultdict[str, List[Any]] = defaultdict(list)
        for data in data_batch:
            collate_dict["src"].append(data["src"])
            collate_dict["tgt"].append(data["tgt"])
            collate_dict["src_sent_labels"].append(data["src_sent_labels"])
            collate_dict["segs"].append(data["segs"])
            collate_dict["clss"].append(data["clss"])
            collate_dict["src_txt"].append(data["src_txt"])
            collate_dict["tgt_txt"].append(data["tgt_txt"])
        return SampleDict(collate_dict) # type: ignore
