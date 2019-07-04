import logging

import torch

from trainer import ModelFramework
from util import setup_logging
import os
import sys

baseline_config = {"modelname": "baseline",
                   "batch_size": 32,
                   "embedding_size": 100,
                   "optimize_embeddings": False,
                   "scale_emb_grad_by_freq": False,
                   "RNN_input_dim": 100,
                   "dropout_rate": 0.2,
                   "RNN_nhidden": 100,
                   "learning_rate": 5e-3,
                   "RNN_layers": 1,
                   "max_iterations": 100,
                   "optimizer": "adam",
                   "char_embeddings": False}

bidaf_config = {"modelname": "bidaf",
                "batch_size": 40,
                "embedding_size": 100,
                "char_embedding_size": 16,
                "highway_layers": 2,
                "highway_dim": 100,
                "char_channel_size": 100,
                "char_channel_width": 5,
                "char_maxsize_vocab": 260,  # as in AllenNLP
                "optimize_embeddings": False,
                "scale_emb_grad_by_freq": False,
                "RNN_input_dim": 100,
                "dropout_rate": 0.2,
                "RNN_nhidden": 100,
                "learning_rate": 0.001,
                "RNN_layers": 1,
                "max_iterations": 10000,
                "optimizer": "adam",
                "ema": True,
                "start_ema_from_it": 0,
                "exp_decay_rate": 0.999,
                "char_embeddings": True,
                "modelname": "bidaf"}

if __name__ == "__main__":
    setup_logging(os.path.basename(sys.argv[0]).split(".")[0],
                  logpath="logs/",
                  config_path="configurations/logging.yml")
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    try:
        framework = ModelFramework()
        framework.fit(bidaf_config, device)
    except BaseException as be:
        logging.error(be)
        raise be