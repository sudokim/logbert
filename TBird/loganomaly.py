# -*- coding: utf-8 -*-
import argparse
import sys

sys.path.append("../")

from logdeep.models.lstm import *
from logdeep.tools.predict import Predicter
from logdeep.tools.train import Trainer
from logdeep.tools.utils import *
from logdeep.dataset.vocab import Vocab

import torch

dataset_dir = "../dataset/tbird/"
output_dir = "../output/tbird/"

# Config Parameters
options = dict()
options["seed"] = int(os.environ.get("SEED", 1234))
options["output_dir"] = output_dir
options["dataset_dir"] = dataset_dir
options["train_vocab"] = dataset_dir + "train"
options["vocab_path"] = dataset_dir + "vocab_loganomaly.pkl"

options["device"] = "cuda" if torch.cuda.is_available() else "cpu"

# Smaple
options["sample"] = "sliding_window"
options["window_size"] = 20  # if fix_window
options["min_len"] = 10

options["train_ratio"] = 1
options["valid_ratio"] = 0.1
options["test_ratio"] = 1

options["is_logkey"] = True
options["is_time"] = False

# Features
options["sequentials"] = options["is_logkey"]
options["quantitatives"] = True
options["semantics"] = False
options["parameters"] = options["is_time"]
options["feature_num"] = sum(
    [options["sequentials"], options["quantitatives"], options["semantics"], options["parameters"]]
)

# Model
options["input_size"] = 1
options["hidden_size"] = 64
options["num_layers"] = 2
options["embedding_dim"] = 50
options["vocab_size"] = 860
options["num_classes"] = options["vocab_size"]
# Train
options["batch_size"] = 128
options["accumulation_step"] = 1

options["optimizer"] = "adam"
options["lr"] = 0.01
options["max_epoch"] = 200
options["n_epochs_stop"] = 10
options["lr_step"] = (options["max_epoch"] - 20, options["max_epoch"])
options["lr_decay_ratio"] = 0.1

options["resume_path"] = None
options["model_name"] = "deeplog"
options["save_dir"] = options["output_dir"] + f"deeplog/{options['seed']}/"

os.makedirs(options["save_dir"], exist_ok=True)

# Predict
options["model_path"] = options["save_dir"] + f"best_{options['model_name']}_{options['seed']}.pth"
options["num_candidates"] = 9
options["threshold"] = None
options["gaussian_mean"] = 0
options["gaussian_std"] = 0
options["num_outputs"] = 1


print("Features logkey:{} time: {}".format(options["is_logkey"], options["is_time"]))
print("Device:", options["device"])

seed_everything(seed=options["seed"])

Model = loganomaly(
    input_size=options["input_size"],
    hidden_size=options["hidden_size"],
    num_layers=options["num_layers"],
    vocab_size=options["vocab_size"],
    embedding_dim=options["embedding_dim"],
)


def train():
    trainer = Trainer(Model, options)
    trainer.start_train()


def predict():
    predicter = Predicter(Model, options)
    predicter.predict_unsupervised()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers()

    train_parser = subparsers.add_parser("train")
    train_parser.set_defaults(mode="train")

    predict_parser = subparsers.add_parser("predict")
    predict_parser.set_defaults(mode="predict")

    vocab_parser = subparsers.add_parser("vocab")
    vocab_parser.set_defaults(mode="vocab")

    args = parser.parse_args()
    print("arguments", args)

    if args.mode == "train":
        train()

    elif args.mode == "predict":
        predict()

    elif args.mode == "vocab":
        with open(options["train_vocab"], "r") as f:
            logs = f.readlines()
        vocab = Vocab(logs)
        print("vocab_size", len(vocab))
        vocab.save_vocab(options["vocab_path"])
