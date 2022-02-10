import math

import torch
import torch.nn as nn
from torch.nn.init import xavier_uniform_

from transformers import BertModel, BertConfig
from transformers.models.bert.modeling_bert import BertAttention
from models.neural import MultiHeadedAttention, PositionwiseFeedForward


class Classifier(nn.Module):
    def __init__(self, hidden_size):
        super(Classifier, self).__init__()
        self.linear1 = nn.Linear(hidden_size, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x, mask_cls):
        h = self.linear1(x).squeeze(-1)
        sent_scores = self.sigmoid(h) * mask_cls.float()
        return sent_scores


class PositionalEncoding(nn.Module):
    def __init__(self, dropout, dim, max_len=5000):
        pe = torch.zeros(max_len, dim)
        position = torch.arange(0, max_len).unsqueeze(1)
        div_term = torch.exp(
            (torch.arange(0, dim, 2, dtype=torch.float) * -(math.log(10000.0) / dim))
        )
        pe[:, 0::2] = torch.sin(position.float() * div_term)
        pe[:, 1::2] = torch.cos(position.float() * div_term)
        pe = pe.unsqueeze(0)
        super(PositionalEncoding, self).__init__()
        self.register_buffer("pe", pe)
        self.dropout = nn.Dropout(p=dropout)
        self.dim = dim

    def forward(self, emb, step=None):
        emb = emb * math.sqrt(self.dim)
        if step:
            emb = emb + self.pe[:, step][:, None, :]

        else:
            emb = emb + self.pe[:, : emb.size(1)]
        emb = self.dropout(emb)
        return emb

    def get_emb(self, emb):
        return self.pe[:, : emb.size(1)]


class TransformerEncoderLayer(nn.Module):
    def __init__(self, d_model, heads, d_ff, dropout):
        super(TransformerEncoderLayer, self).__init__()

        self.self_attn = MultiHeadedAttention(heads, d_model, dropout=dropout)
        self.feed_forward = PositionwiseFeedForward(d_model, d_ff, dropout)
        self.layer_norm = nn.LayerNorm(d_model, eps=1e-6)
        self.dropout = nn.Dropout(dropout)

    def forward(self, iter, query, inputs, mask):
        if iter != 0:
            input_norm = self.layer_norm(inputs)
        else:
            input_norm = inputs

        mask = mask.unsqueeze(1)
        context = self.self_attn(input_norm, input_norm, input_norm, mask=mask)
        out = self.dropout(context) + inputs
        return self.feed_forward(out)


class ExtTransformerEncoder(nn.Module):
    def __init__(self, d_model, d_ff, heads, dropout, num_inter_layers=0):
        super(ExtTransformerEncoder, self).__init__()
        self.d_model = d_model
        self.num_inter_layers = num_inter_layers
        self.pos_emb = PositionalEncoding(dropout, d_model)
        self.transformer_inter = nn.ModuleList(
            [
                TransformerEncoderLayer(d_model, heads, d_ff, dropout)
                for _ in range(num_inter_layers)
            ]
        )
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(d_model, eps=1e-6)
        self.wo = nn.Linear(d_model, 1, bias=True)
        self.sigmoid = nn.Sigmoid()

    def forward(self, top_vecs, mask):
        """ See :obj:`EncoderBase.forward()`"""

        batch_size, n_sents = top_vecs.size(0), top_vecs.size(1)
        pos_emb = self.pos_emb.pe[:, :n_sents]
        x = top_vecs * mask[:, :, None].float()
        x = x + pos_emb

        for i in range(self.num_inter_layers):
            x = self.transformer_inter[i](
                i, x, x, ~(mask)
            )  # all_sents * max_tokens * dim

        x = self.layer_norm(x)
        sent_scores = self.sigmoid(self.wo(x))
        sent_scores = sent_scores.squeeze(-1) * mask.float()

        return sent_scores


class Bert(nn.Module):
    def __init__(self, large=False, finetune=True):
        super(Bert, self).__init__()
        if large:
            self.model = BertModel.from_pretrained("bert-large-uncased")
        else:
            self.model = BertModel.from_pretrained("sberbank-ai/ruBert-base")

        self.finetune = finetune

    def forward(self, x, segs, mask):
        if self.finetune:
            top_vec = self.model(x, attention_mask=mask, token_type_ids=segs)[0]
        else:
            self.eval()
            with torch.no_grad():
                top_vec, _ = self.model(x, attention_mask=mask, token_type_ids=segs)
        return top_vec


class ExtSummarizer(nn.Module):
    def __init__(self, args, checkpoint=None):
        super(ExtSummarizer, self).__init__()
        self.args = args
        self.bert = Bert()

        self.ext_layer = ExtTransformerEncoder(
            self.bert.model.config.hidden_size,
            args.ext_ff_size,
            args.ext_heads,
            args.ext_dropout,
            args.ext_layers,
        )
        if args.encoder == "baseline":
            bert_config = BertConfig(
                self.bert.model.config.vocab_size,
                hidden_size=args.ext_hidden_size,
                num_hidden_layers=args.ext_layers,
                num_attention_heads=args.ext_heads,
                intermediate_size=args.ext_ff_size,
            )
            self.bert.model = BertModel(bert_config)
            self.ext_layer = Classifier(self.bert.model.config.hidden_size)

        if args.max_pos > 512:
            my_pos_embeddings = nn.Embedding(
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

    def forward(self, src, segs, clss, mask_src, mask_cls):
        top_vec = self.bert(src, segs, mask_src)
        sents_vec = top_vec[torch.arange(top_vec.size(0)).unsqueeze(1), clss]
        sents_vec = sents_vec * mask_cls[:, :, None].float()
        sent_scores = self.ext_layer(sents_vec, mask_cls).squeeze(-1)
        return sent_scores, mask_cls
