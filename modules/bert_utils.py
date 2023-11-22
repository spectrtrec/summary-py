import torch
import math
import torch.nn as nn
from modules.self_attention import MultiHeadAttention
from transformers import BertModel  # type: ignore
from torch import Tensor
from typing import Callable


def gelu(x: Tensor) -> Tensor:
    return (
        0.5
        * x
        * (1 + torch.tanh(math.sqrt(2 / math.pi) * (x + 0.044715 * torch.pow(x, 3))))
    )


class Classifier(nn.Module):
    def __init__(self, hidden_size: int) -> None:
        super(Classifier, self).__init__()
        self.linear1: nn.Linear = nn.Linear(hidden_size, 1)
        self.sigmoid: nn.Sigmoid = nn.Sigmoid()

    def forward(self, x, mask_cls) -> Tensor:
        return self.sigmoid(self.linear1(x).squeeze(-1)) * mask_cls.float()


class PositionwiseFeedForward(nn.Module):
    """A two-layer Feed-Forward-Network with residual layer norm.

    Args:
        d_model (int): the size of input for the first-layer of the FFN.
        d_ff (int): the hidden layer size of the second-layer
            of the FNN.
        dropout (float): dropout probability in :math:`[0, 1)`.
    """

    def __init__(self, d_model: int, d_ff: int, dropout: float = 0.1) -> None:
        super(PositionwiseFeedForward, self).__init__()
        self.w_1: nn.Linear = nn.Linear(d_model, d_ff)
        self.w_2: nn.Linear = nn.Linear(d_ff, d_model)
        self.layer_norm: nn.LayerNorm = nn.LayerNorm(d_model, eps=1e-6)
        self.actv: Callable = gelu
        self.dropout_1: nn.Dropout = nn.Dropout(dropout)
        self.dropout_2: nn.Dropout = nn.Dropout(dropout)

    def forward(self, x) -> Tensor:
        inter: Tensor = self.dropout_1(self.actv(self.w_1(self.layer_norm(x))))
        output: Tensor = self.dropout_2(self.w_2(inter))
        return output + x


class PositionalEncoding(nn.Module):
    def __init__(self, dropout: float, dim: int, max_len: int = 5000) -> None:
        self.pe: Tensor = torch.zeros(max_len, dim).to(device='cuda')
        position: Tensor = torch.arange(0, max_len).unsqueeze(1)
        div_term: Tensor = torch.exp(
            (torch.arange(0, dim, 2, dtype=torch.float) * -(math.log(10000.0) / dim))
        )
        self.pe[:, 0::2] = torch.sin(position.float() * div_term)
        self.pe[:, 1::2] = torch.cos(position.float() * div_term)
        self.pe: Tensor = self.pe.unsqueeze(0)
        super(PositionalEncoding, self).__init__()
        self.dropout: nn.Dropout = nn.Dropout(p=dropout)
        self.dim: int = dim

    def forward(self, embeding: Tensor, step=None) -> Tensor:
        emb: Tensor = embeding * math.sqrt(self.dim)
        if step:
            emb = emb + self.pe[:, step][:, None, :]

        else:
            emb = emb + self.pe[:, : emb.size(1)]
        emb = self.dropout(emb)
        return emb


class TransformerEncoderLayer(nn.Module):
    def __init__(self, d_model: int, heads: int, d_ff: int, dropout: float) -> None:
        super(TransformerEncoderLayer, self).__init__()

        self.self_attn: MultiHeadAttention = MultiHeadAttention(heads, d_model)
        self.feed_forward: PositionwiseFeedForward = PositionwiseFeedForward(
            d_model, d_ff, dropout
        )
        self.layer_norm: nn.LayerNorm = nn.LayerNorm(d_model, eps=1e-6)
        self.dropout: nn.Dropout = nn.Dropout(dropout)

    def forward(self, iter: int, query: Tensor, inputs: Tensor, mask: Tensor) -> Tensor:
        if iter != 0:
            input_norm: Tensor = self.layer_norm(inputs)
        else:
            input_norm: Tensor = inputs
        # mask:Tensor = mask.unsqueeze(1)
        context: Tensor = self.self_attn(
            input_norm, input_norm, input_norm, mask.unsqueeze(1)
        )
        out: Tensor = self.dropout(context) + inputs
        return self.feed_forward(out)


class Bert(nn.Module):
    def __init__(self, large: bool = False, finetune: bool = True) -> None:
        super(Bert, self).__init__()
        if large:
            self.model: BertModel = BertModel.from_pretrained("bert-large-uncased")  # type: ignore
        else:
            self.model: BertModel = BertModel.from_pretrained("sberbank-ai/ruBert-base")  # type: ignore

        self.finetune: bool = finetune

    def forward(self, x: Tensor, segs: Tensor, mask: Tensor) -> Tensor:
        if self.finetune:
            top_vec: Tensor = self.model(x, attention_mask=mask, token_type_ids=segs)[0]
        else:
            self.eval()
            with torch.no_grad():
                top_vec: Tensor
                _: Tensor = self.model(x, attention_mask=mask, token_type_ids=segs)
        return top_vec  # type: ignore
