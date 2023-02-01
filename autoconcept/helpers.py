import json
import os
import random

import numpy as np
import requests
import torch
from dotenv import load_dotenv
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
