import torch
import torch.nn as nn
from torch.nn.init import xavier_uniform_
from torch import Tensor
from typing import Tuple

from transformers import BertModel, BertConfig  # type: ignore
from modules.bert_utils import (
    Classifier,
    Bert,
    PositionalEncoding,
    TransformerEncoderLayer,
)


class ExtTransformerEncoder(nn.Module):
    def __init__(
        self,
        d_model: int,
        d_ff: int,
        heads: int,
        dropout: int,
        num_inter_layers: int = 0,
    ) -> None:
        super(ExtTransformerEncoder, self).__init__()
        self.d_model: int = d_model
        self.num_inter_layers: int = num_inter_layers
        self.pos_emb: PositionalEncoding = PositionalEncoding(dropout, d_model)
        self.transformer_inter: nn.ModuleList = nn.ModuleList(
            [
                TransformerEncoderLayer(d_model, heads, d_ff, dropout)
                for _ in range(num_inter_layers)
            ]
        )
        self.dropout: nn.Dropout = nn.Dropout(dropout)
        self.layer_norm: nn.LayerNorm = nn.LayerNorm(d_model, eps=1e-6)
        self.wo: nn.Linear = nn.Linear(d_model, 1, bias=True)
        self.sigmoid: nn.Sigmoid = nn.Sigmoid()

    def forward(self, top_vecs: Tensor, mask: Tensor) -> Tensor:
        """See :obj:`EncoderBase.forward()`"""

        batch_size: int = top_vecs.size(0)
        n_sents: int = top_vecs.size(1)
        x: Tensor = top_vecs * mask[:, :, None].float() + self.pos_emb.pe[:, :n_sents]

        for i in range(self.num_inter_layers):
            x = self.transformer_inter[i](i, x, x, ~(mask))

        x = self.layer_norm(x)
        sent_scores: Tensor = self.sigmoid(self.wo(x)).squeeze(-1) * mask.float()
        return sent_scores


class ExtSummarizer(nn.Module):
    def __init__(self, args, checkpoint=None) -> None:
        super(ExtSummarizer, self).__init__()
        self.args = args
        self.bert = Bert()

        self.ext_layer: ExtTransformerEncoder = ExtTransformerEncoder(  # type: ignore
            self.bert.model.config.hidden_size,
            args.ext_ff_size,
            args.ext_heads,
            args.ext_dropout,
            args.ext_layers,
        )
        if args.encoder == "baseline":
            bert_config: BertConfig = BertConfig(
                self.bert.model.config.vocab_size,
                hidden_size=args.ext_hidden_size,
                num_hidden_layers=args.ext_layers,
                num_attention_heads=args.ext_heads,
                intermediate_size=args.ext_ff_size,
            )
            self.bert.model = BertModel(bert_config)
            self.ext_layer: Classifier = Classifier(self.bert.model.config.hidden_size)

        if args.max_pos > 512:
            my_pos_embeddings: nn.Embedding = nn.Embedding(
                args.max_pos, self.bert.model.config.hidden_size
            )
            my_pos_embeddings.weight.data[
                :512
            ] = self.bert.model.embeddings.position_embeddings.weight.data
            my_pos_embeddings.weight.data[
                512:
            ] = self.bert.model.embeddings.position_embeddings.weight.data[-1][
                None, :
            ].repeat(
                args.max_pos - 512, 1
            )
            self.bert.model.embeddings.position_embeddings = my_pos_embeddings

        if checkpoint is not None:
            self.load_state_dict(checkpoint["model"], strict=True)
        else:
            if args.param_init != 0.0:
                for p in self.ext_layer.parameters():
                    p.data.uniform_(-args.param_init, args.param_init)
            if args.param_init_glorot:
                for p in self.ext_layer.parameters():
                    if p.dim() > 1:
                        xavier_uniform_(p)

    def forward(self, src, segs, clss, mask_src, mask_cls) -> Tuple[Tensor, Tensor]:
        top_vec: Tensor = self.bert(src, segs, mask_src)
        sents_vec: Tensor = top_vec[torch.arange(top_vec.size(0)).unsqueeze(1), clss]
        sents_vec: Tensor = sents_vec * mask_cls[:, :, None].float()
        sent_scores: Tensor = self.ext_layer(sents_vec, mask_cls).squeeze(-1)
        return sent_scores, mask_cls
