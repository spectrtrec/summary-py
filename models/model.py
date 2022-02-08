from collections import OrderedDict
import pytorch_lightning as pl
import numpy as np
import torch
import torch.nn as nn
from torch import optim
from transformers import BertConfig, BertModel
from typing import Any, Dict, List, Union, Tuple
from models.data_loader import DataModule
from utils.utils import block_tri
from torchmetrics.text.rouge import ROUGEScore

class Bert(nn.Module):
    def __init__(self, large, temp_dir, finetune=False):
        super(Bert, self).__init__()
        if large:
            self.model = BertModel.from_pretrained(
                "bert-large-uncased", cache_dir=temp_dir
            )
        else:
            self.model = BertModel.from_pretrained(
                "bert-base-uncased", cache_dir=temp_dir
            )

        self.finetune = finetune

    def forward(self, x, segs, mask) -> torch.Tensor:
        if self.finetune:
            top_vec, _ = self.model(x, segs, attention_mask=mask)
        else:
            self.eval()
            with torch.no_grad():
                top_vec, _ = self.model(x, segs, attention_mask=mask)
        return top_vec


class ExtractiveModel(pl.LightningModule):
    def __init__(self, args, data) -> None:
        super(ExtractiveModel, self).__init__()
        self.hparams = args
        self.data = DataModule(self.hparams, data, data)
        self.rouge = ROUGEScore()
        self.__build_model()
        self.__build_loss()

    def __build_model(self) -> None:
        bert_config = BertConfig.from_pretrained(
            self.hparams.pretrained_model_name,
            hidden_size=self.hparams.ext_hidden_size,
            num_hidden_layers=self.hparams.ext_layers,
            num_attention_heads=self.hparams.ext_heads,
            intermediate_size=self.hparams.ext_ff_size,
        )
        self.bert = BertModel.from_pretrained(
            self.hparams.pretrained_model_name, config=bert_config
        )
        self.linear1 = nn.Linear(bert_config.hidden_size, 1)
        self.sigmoid = nn.Sigmoid()

    def __build_loss(self) -> None:
        self.__loss = torch.nn.BCELoss(reduction="none")

    def forward(
        self, src, segs, clss, mask_src, mask_cls
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        top_vec = self.bert(src, attention_mask=mask_src, token_type_ids=segs)[0]
        sents_vec = top_vec[torch.arange(top_vec.size(0)).unsqueeze(1), clss]
        sents_vec = sents_vec * mask_cls[:, :, None].float()
        h = self.linear1(sents_vec).squeeze(-1)
        sent_scores = self.sigmoid(h) * mask_cls.float()

        return sent_scores.squeeze(-1), mask_cls

    def loss(self, predictions: dict, targets: dict) -> torch.Tensor:
        return self.__loss(predictions, targets.squeeze().float())

    def training_step(
        self, batch: Dict[str, torch.Tensor], batch_nb: int, *args, **kwargs
    ) -> Dict[str, torch.Tensor]:

        model_out, mask = self.forward(
            batch["src"],
            batch["segs"],
            batch["clss"],
            batch["mask_src"],
            batch["mask_cls"],
        )
        loss_val = self.loss(model_out, batch["src_sent_labels"])
        loss_val = (loss_val * mask.float()).sum()

        if self.trainer.use_dp or self.trainer.use_ddp2:
            loss_val = loss_val.unsqueeze(0)

        tqdm_dict = {"train_loss": loss_val}
        output = OrderedDict(
            {"loss": loss_val, "progress_bar": tqdm_dict, "log": tqdm_dict}
        )

        return output

    def training_epoch_end(
        self, outputs: Dict[str, torch.Tensor]
    ) -> Dict[str, Union[int, float]]:

        train_loss_mean = 0
        for output in outputs:
            train_loss = output["loss"]
            train_loss = torch.mean(train_loss)
            train_loss_mean += train_loss

        train_loss_mean /= len(outputs)
        tqdm_dict = {
            "train_loss_mean": train_loss_mean,
        }
        result = OrderedDict(
            {
                "train_loss_mean": train_loss_mean,
                "progress_bar": tqdm_dict,
                "log": tqdm_dict,
            }
        )
        return result

    def validation_step(
        self, batch: Dict[str, torch.Tensor], batch_nb: int, *args, **kwargs
    ) -> Dict[str, torch.Tensor]:

        model_out, mask = self.forward(
            batch["src"],
            batch["segs"],
            batch["clss"],
            batch["mask_src"],
            batch["mask_cls"],
        )
        loss_val = self.loss(model_out, batch["src_sent_labels"])
        loss_val = (loss_val * mask.float()).sum()
        if self.trainer.use_dp or self.trainer.use_ddp2:
            loss_val = loss_val.unsqueeze(0)

        rouge = self.translate(model_out, mask, batch["src_txt"], batch["tgt_txt"])
        output = OrderedDict({"val_loss": loss_val, "rouge": rouge})

        return output

    def validation_epoch_end(
        self, outputs: Dict[str, torch.Tensor]
    ) -> Dict[str, Union[int, float]]:

        val_loss_mean, rouge1_fmeaser = 0, 0
        for output in outputs:
            val_loss = output["val_loss"]
            rouge_f = output["rouge"]['rouge1_fmeasure']
            if self.trainer.use_dp or self.trainer.use_ddp2:
                val_loss = torch.mean(val_loss)
            val_loss_mean += val_loss
            rouge1_fmeaser += rouge_f.item()

        val_loss_mean /= len(outputs)
        rouge1_fmeaser /= len(outputs)
        tqdm_dict = {
            "val_loss_mean": val_loss_mean,
            "rouge1_fmeaser": rouge1_fmeaser
        }
        result = OrderedDict(
            {
                "val_loss_mean": val_loss_mean,
                "rouge1_fmeaser": rouge1_fmeaser,
                "progress_bar": tqdm_dict,
                "log": tqdm_dict,
            }
        )
        return result

    def configure_optimizers(self) -> List[torch.optim.Adam]:
        optimizer = optim.Adam(self.bert.parameters(), lr=self.hparams.learning_rate)
        return [optimizer]

    def translate(
        self,
        sent_scores: torch.Tensor,
        mask: torch.Tensor,
        src_str: List[List[str]],
        tgt_str: List[List[str]],
    ) -> None:
        sent_scores = (sent_scores + mask.float()).cpu().data.numpy()
        selected_ids = np.argsort(-sent_scores, 1)

        gold, pred = [], []
        for i, _ in enumerate(selected_ids):
            _pred = []
            if len(src_str[i]) == 0:
                continue
            for j in selected_ids[i][: len(src_str[i])]:
                if j >= len(src_str[i]):
                    continue
                candidate = src_str[i][j].strip()
                if (self.hparams.block_trigram):
                    if (not block_tri(candidate, _pred)):
                        _pred.append(candidate)
                else:
                    _pred.append(candidate)

                if (len(_pred) == 3):
                    break
            _pred = "<q>".join(_pred)
            pred.append(_pred)
            gold.append(tgt_str[i])

        return self.rouge(pred, gold)
