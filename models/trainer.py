from collections import OrderedDict
from typing import Dict, List, Union, OrderedDict

import numpy as np
import pytorch_lightning as pl
import torch
from torch import Tensor, optim
from torch.nn import BCELoss
from torchmetrics.text.rouge import ROUGEScore
from utils.utils import block_tri

from models.data_loader import DataModule
from models.ext_encoder import ExtSummarizer


class ExtractiveTrainer(pl.LightningModule):
    def __init__(self, args, data) -> None:
        super(ExtractiveTrainer, self).__init__()
        self.save_hyperparameters(args)
        self.data: DataModule = DataModule(self.hparams, data, data)
        self.rouge: ROUGEScore = ROUGEScore()
        self.__build_model()
        self.__build_loss()

    def __build_model(self) -> None:
        self.model: ExtSummarizer = ExtSummarizer(self.hparams)

    def __build_loss(self) -> None:
        self.__loss: BCELoss = BCELoss(reduction="none")

    def loss(self, predictions: Tensor, targets: Tensor) -> Tensor:
        return self.__loss(predictions, targets.squeeze().float())

    def training_step(
        self, batch: Dict[str, Tensor], batch_nb: int, *args, **kwargs
    ) -> OrderedDict[str, Union[Tensor, Dict[str, Tensor]]]:

        model_out, mask = self.model.forward(
            batch["src"],
            batch["segs"],
            batch["clss"],
            batch["mask_src"],
            batch["mask_cls"],
        )
        loss_val: Tensor = self.loss(model_out, batch["src_sent_labels"])
        loss_val: Tensor = (loss_val * mask.float()).sum()
        tqdm_dict: Dict[str, Tensor] = {"train_loss": loss_val}
        output: OrderedDict[str, Union[Tensor, Dict[str, Tensor]]] = OrderedDict(
            {"loss": loss_val, "progress_bar": tqdm_dict, "log": tqdm_dict}
        )

        return output

    def training_epoch_end(self, outputs: Dict[str, torch.Tensor]) -> None:

        train_loss_mean = 0
        for output in outputs:
            train_loss = output["loss"] # type: ignore
            train_loss = torch.mean(train_loss) # type: ignore
            train_loss_mean += train_loss

        train_loss_mean /= len(outputs)
        self.log("train_loss_mean", train_loss_mean, True)

    def validation_step(
        self, batch: Dict[str, Union[torch.Tensor, List[str], List[List[str]]]], batch_nb: int, *args, **kwargs
    ) -> OrderedDict[str, Union[Tensor, Dict[str, Tensor]]]:

        model_out, mask = self.model.forward(
            batch["src"],
            batch["segs"],
            batch["clss"],
            batch["mask_src"],
            batch["mask_cls"],
        )
        loss_val: Tensor = self.loss(model_out, batch["src_sent_labels"]) # type: ignore
        loss_val: Tensor = (loss_val * mask.float()).sum()
        rouge: Dict[str, Tensor] = self.translate(model_out, mask, batch["src_txt"], batch["tgt_txt"]) # type: ignore
        output: OrderedDict[str, Union[Tensor, Dict[str, Tensor]]] = OrderedDict({"val_loss": loss_val, "rouge": rouge})

        return output

    def validation_epoch_end(self, outputs: Dict[str, torch.Tensor]) -> None:

        val_loss_mean, rouge1_fmeaser = 0, 0
        for output in outputs:
            val_loss = output["val_loss"] # type: ignore
            rouge_f = output["rouge"]["rouge1_fmeasure"] # type: ignore
            val_loss_mean += val_loss # type: ignore
            rouge1_fmeaser += rouge_f.item() # type: ignore

        val_loss_mean /= len(outputs)
        rouge1_fmeaser /= len(outputs)
        self.log("val_loss_mean", val_loss_mean, True)
        self.log("rouge1_fmeaser", rouge1_fmeaser, True)

    def configure_optimizers(self) -> List[torch.optim.Adam]:
        optimizer = optim.Adam(
            self.model.bert.model.parameters(), lr=self.hparams.learning_rate # type: ignore
        )
        return [optimizer]

    def translate(
        self,
        sent_scores: torch.Tensor,
        mask: torch.Tensor,
        src_str: List[List[str]],
        tgt_str: List[str],
    ) -> Dict[str, Tensor]:
        sent_scores = (sent_scores + mask.float()).cpu().data.numpy()
        selected_ids = np.argsort(-sent_scores, 1)
        gold: List[str] = []
        pred: List[List[str]] = []
        for i, _ in enumerate(selected_ids):
            _pred: List[str] = []
            if len(src_str[i]) == 0:
                continue
            for j in selected_ids[i][: len(src_str[i])]:
                if j >= len(src_str[i]):
                    continue
                candidate: str = src_str[i][j].strip()
                if self.hparams.block_trigram: # type: ignore
                    if not block_tri(candidate, _pred):
                        _pred.append(candidate)
                else:
                    _pred.append(candidate)
                if len(_pred) == 3:
                    break
            pred.append(_pred)
            gold.append(tgt_str[i])
        return self.rouge_score(pred, gold)

    def rouge_score(
        self, prediction: List[List[str]], target: List[str]
    ) -> Dict[str, Tensor]:
        for i, trg in enumerate(target):
            [self.rouge.update(trg, pred) for pred in prediction[i]]
        return self.rouge.compute()
