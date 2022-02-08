from preprocess.data_builder import BertDatabuilder
from utils.utils import load_json, ria_parser
import time
import argparse
import os
from datetime import datetime

from models.model import ExtractiveModel
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger
from torchnlp.random import set_seed
from preprocess.paraller_builder import build_bert_json


def main(hparams, data):
    set_seed(hparams.seed)

    model = ExtractiveModel(hparams, data)

    early_stop_callback = EarlyStopping(
        monitor=hparams.monitor,
        min_delta=0.0,
        patience=hparams.patience,
        verbose=True,
        mode=hparams.metric_mode,
    )

    tb_logger = TensorBoardLogger(
        save_dir="experiments/",
        version="version_" + datetime.now().strftime("%d-%m-%Y--%H-%M-%S"),
        name="",
    )

    ckpt_path = os.path.join("experiments/", tb_logger.version, "checkpoints",)

    checkpoint_callback = ModelCheckpoint(
        filepath=ckpt_path,
        save_top_k=hparams.save_top_k,
        verbose=True,
        monitor=hparams.monitor,
        period=1,
        mode=hparams.metric_mode,
        save_weights_only=True,
    )

    trainer = Trainer(
        logger=tb_logger,
        checkpoint_callback=True,
        callbacks=[early_stop_callback, checkpoint_callback],
        gradient_clip_val=1.0,
        gpus=hparams.gpus,
        log_gpu_memory="all",
        deterministic=True,
        check_val_every_n_epoch=1,
        fast_dev_run=False,
        accumulate_grad_batches=hparams.accumulate_grad_batches,
        max_epochs=hparams.max_epochs,
        min_epochs=hparams.min_epochs,
        val_check_interval=hparams.val_check_interval,
        distributed_backend="dp",
    )

    trainer.fit(model, model.data)


if __name__ == "__main__":


    parser = argparse.ArgumentParser(
        description="Minimalist Transformer Classifier", add_help=True,
    )
    parser.add_argument("--seed", type=int, default=3, help="Training seed.")
    parser.add_argument(
        "--save_top_k",
        default=1,
        type=int,
        help="The best k models according to the quantity monitored will be saved.",
    )
    parser.add_argument("--loader_workers", type=int, default=8, help="How workers")
    parser.add_argument(
        "--monitor", default="val_loss_mean", type=str, help="Quantity to monitor."
    )
    parser.add_argument(
        "--metric_mode",
        default="min",
        type=str,
        help="If we want to min/max the monitored quantity.",
        choices=["auto", "min", "max"],
    )
    parser.add_argument(
        "--patience",
        default=2,
        type=int,
        help=(
            "Number of epochs with no improvement "
            "after which training will be stopped."
        ),
    )
    parser.add_argument(
        "--min_epochs",
        default=1,
        type=int,
        help="Limits training to a minimum number of epochs",
    )
    parser.add_argument(
        "--max_epochs",
        default=5,
        type=int,
        help="Limits training to a max number number of epochs",
    )
    parser.add_argument(
        "--learning_rate", default=3e-05, type=float, help="Learning rate",
    )

    parser.add_argument(
        "--batch_size", default=6, type=int, help="Batch size to be used."
    )
    parser.add_argument(
        "--accumulate_grad_batches",
        default=2,
        type=int,
        help=(
            "Accumulated gradients runs K small batches of size N before "
            "doing a backwards pass."
        ),
    )
    parser.add_argument("--use_ria", type=bool, default=True, help="Use interval")
    parser.add_argument("--block_trigram", type=bool, default=True, help="Use interval")
    # gpu args
    parser.add_argument("--gpus", type=int, default=1, help="How many gpus")
    # Data Loader
    parser.add_argument("--max_tgt_len", type=int, default=512, help="Target length")
    parser.add_argument("--max_pos", type=int, default=512, help="Max position length")
    parser.add_argument("--use_interval", type=bool, default=False, help="Use interval")
    parser.add_argument(
        "--pretrained_model_name",
        type=str,
        default="sberbank-ai/ruBert-base",
        help="Model name",
    )
    parser.add_argument(
        "--val_check_interval",
        default=1.0,
        type=float,
        help=(
            "If you don't want to use the entire dev set (for debugging or "
            "if it's huge), set how much of the dev set you want to use with this flag."
        ),
    )
    parser.add_argument(
        "--ext_hidden_size",
        type=int,
        default=768,
        help="Dimensionality of the encoder layers and the pooler layer.",
    )
    parser.add_argument(
        "--ext_layers",
        type=int,
        default=12,
        help="Number of hidden layers in the Transformer encoder.",
    )
    parser.add_argument(
        "--ext_heads",
        type=int,
        default=12,
        help="Number of attention heads for each attention layer in the Transformer encoder.",
    )
    parser.add_argument(
        "--ext_ff_size",
        type=int,
        default=3072,
        help="Dimensionality of the “intermediate” (often named feed-forward) layer in the Transformer encoder.",
    )

    hparams = parser.parse_args()
    if hparams.use_ria:
        ria_list = ria_parser("/home/anton/summary-py/data/ria_1k.json")
        json_data = build_bert_json(ria_list)
    bert_data = BertDatabuilder()
    prepared_data = bert_data.preprocess(json_data)
    main(hparams, prepared_data)
