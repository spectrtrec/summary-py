import argparse
import os
from datetime import datetime

import hydra
from omegaconf import DictConfig
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger
from torchnlp.random import set_seed

from models.trainer import ExtractiveTrainer
from preprocess.data_builder import BertDatabuilder
from preprocess.paraller_builder import build_bert_json
from utils.utils import ria_parser


@hydra.main(config_path="config", config_name="ext_summary.yaml")
def main(hparams: DictConfig):
    if hparams.use_ria:
        ria_list = ria_parser("/home/anton/summary-py/data/ria_1k.json")
        json_data = build_bert_json(ria_list)

    bert_data = BertDatabuilder(hparams)
    data = bert_data.preprocess(json_data)

    set_seed(hparams.seed)

    model = ExtractiveTrainer(hparams, data)

    early_stop_callback = EarlyStopping(
        monitor=hparams.monitor,
        min_delta=0.0,
        patience=hparams.patience,
        verbose=True,
        mode=hparams.metric_mode,
        check_on_train_epoch_end=False,
    )

    tb_logger = TensorBoardLogger(
        save_dir="experiments/",
        version="version_" + datetime.now().strftime("%d-%m-%Y--%H-%M-%S"),
        name="",
    )

    ckpt_path = os.path.join("experiments/", tb_logger.version, "checkpoints",)

    checkpoint_callback = ModelCheckpoint(
        dirpath=ckpt_path,
        save_top_k=hparams.save_top_k,
        verbose=True,
        monitor=hparams.monitor,
        every_n_epochs=1,
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
    )

    trainer.fit(model, model.data)


if __name__ == "__main__":
    main()
