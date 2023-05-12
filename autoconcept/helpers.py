import json
import os
import random

import numpy as np
import pytorch_lightning as pl
import requests
import torch
from dotenv import load_dotenv
from hydra.utils import get_class, instantiate
from omegaconf import OmegaConf

load_dotenv()


def report_to_telegram(message):
    requests.get(
        'https://api.telegram.org/bot{bot_token}/sendMessage?chat_id={chat_id}&text={text}'.format(
            bot_token=os.environ['BOT_TOKEN'],
            chat_id=os.environ['CHAT_ID'],
            text=message)
    )


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.enabled = False


def pretty_cfg(cfg):
    cfg_dict = OmegaConf.to_container(cfg, resolve=True)
    cfg_json = json.dumps(cfg_dict, indent=2)
    return cfg_json


def load_experiment(path):
    print("Fetching configuration...")
    cfg_path = os.path.join(path, ".hydra/config.yaml")
    cfg = OmegaConf.load(cfg_path)

    set_seed(cfg.seed)
    pl.seed_everything(cfg.seed, workers=True)

    print("Loading datamodule...")
    dm = instantiate(cfg.dataset)
    dm.setup()

    print("Loading model")
    checkpoints_dir = os.path.join(
        path, "lightning_logs", "version_0", "checkpoints")
    checkpoint_name = [n for n in os.listdir(
        checkpoints_dir) if n != "last.ckpt"][0]
    checkpoint_path = os.path.join(checkpoints_dir, checkpoint_name)

    target_class = get_class(cfg.model._target_)
    main = instantiate(cfg.model.main)
    model = target_class.load_from_checkpoint(
        checkpoint_path, main=main).cuda()
    model = model.eval()

    return dm, model
