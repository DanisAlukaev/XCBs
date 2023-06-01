import json
import os
import random

import matplotlib.pyplot as plt
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


def plot_instances(features, n_images):
    fig, axes = plt.subplots(3, n_images)
    fig.set_size_inches(10, 4)
    labels = ["lrg", "mdm", "sml"]
    for i, ax in enumerate(axes.flatten()):
        label = labels[i // n_images]
        image = plt.imread(features[label][i % n_images][0])
        label = float(features[label][i % n_images][1])
        ax.set_title(f"{label:.3f}")
        ax.imshow(image)
        ax.xaxis.set_ticks_position('none')
        ax.yaxis.set_ticks_position('none')
        ax.set_yticklabels([])
        ax.set_xticklabels([])
    plt.show()


def visualize_concept(results, n_images, concept_id=0, n_tokens=10):
    print(f"Concept #{concept_id + 1}\n")
    print("Top-k tokens w.r.t. average attention score:")
    pair = results[concept_id]
    token_attn = [(t, a) for t, a in pair["concept"]][:n_tokens]
    for idx, (t, a) in enumerate(token_attn):
        print(f"\t{idx + 1}. {t}: {a:.4f}", sep=" ")
    if pair["feature"]:
        print("\nTop-n images with largest absolute values of logits:")
        plot_instances(pair["feature"], n_images)
    print(120 * "-")
