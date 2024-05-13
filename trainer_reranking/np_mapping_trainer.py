import argparse
import os
import time
import pathlib
import sys

from dataclasses import dataclass, field, asdict
from typing import Optional, List, Tuple, Union

import torch
import torch.nn as nn
from transformers import BertForSequenceClassification, TrainingArguments, Trainer
from transformers import AutoTokenizer, Trainer, DataCollatorWithPadding, TrainingArguments, \
    EvalPrediction, HfArgumentParser

# _project_root = str(pathlib.Path(__file__).resolve().parents[1])
_project_root = '.'
sys.path.insert(0, _project_root)

from dataset.classifier_dataset import ClassifierDataSet
from model.model import BertFNNClassifier
from model.customized_loss import FocalLoss

@dataclass
class ModelArguments:
    base_model_name: str = field(default="Bert")
    base_model_path: Optional[str] = field(default='/data/chentianyu/libminer/input/bert-base-uncased_raw')
    evaluate_only: bool = field(default=False)
    to_evaluate_checkpoint: str = field(default=None)
    loss_fct: str = field(default="LabelSmoothingLoss")
    loss_epsilon: float = field(default=0)
    loss_alpha: float = field(default=0.25)
    loss_gamma: float = field(default=2)
    num_labels: float = field(default=1)

parser = HfArgumentParser(ModelArguments)
model_args = parser.parse_args_into_dataclasses()[0]


loss_args = {
    "epsilon": model_args.loss_epsilon,
    "alpha": model_args.loss_alpha,
    "gamma": model_args.loss_gamma
}

loss_fct = FocalLoss(**loss_args)

from_pretrained_args = {
    "pretrained_model_name_or_path": model_args.base_model_path,
    "use_cache": False,
    "num_labels": 1,
    "torch_dtype": torch.float16,
    "loss_fct": loss_fct,
}


parser = argparse.ArgumentParser(description="train bert fnn",
                                 formatter_class=argparse.ArgumentDefaultsHelpFormatter)

parser.add_argument('--data_url', type=str, help='the training data', default="/data/chentianyu/libminer/input/")
parser.add_argument('--train_url', type=str, help='the path model saved', default="/efs_data/chentianyu/VulLibMiner/")
parser.add_argument('--sep_token', type=str, help='sep token of lib corpus', default=" ")
parser.add_argument('--mask_rate', type=float, help='rate of mask lib corpus', default=0)

args, _ = parser.parse_known_args()
input_path = args.data_url
output_path = args.train_url
bert_base_path = os.path.join(input_path, "bert-base-uncased_raw")
# model = BertForSequenceClassification.from_pretrained(bert_base_path)
model = BertFNNClassifier.from_pretrained(**from_pretrained_args)

data_base_path = os.path.join(input_path, "dataset_v0")
train_data_set = ClassifierDataSet(os.path.join(data_base_path, "train.json"),
                                   args.sep_token, args.mask_rate, bert_base_path)
valid_data_set = ClassifierDataSet(os.path.join(data_base_path, "validate.json"),
                                   args.sep_token, args.mask_rate, bert_base_path)

# model.classifier = nn.Linear(model.config.hidden_size, 2)

training_args = TrainingArguments(
    output_dir=os.path.join(output_path, time.strftime("%Y_%m%d_%H_%M", time.localtime(time.time()))),
    num_train_epochs=30,
    evaluation_strategy="epoch",
    eval_delay=1,
    save_strategy="epoch",
    per_device_train_batch_size=16,
    per_device_eval_batch_size=64,
    warmup_steps=500,
    weight_decay=0.01,
    logging_dir=os.path.join(output_path, 'logs'),
    logging_steps=10,
    optim="adamw_torch",
    learning_rate=2e-5,
    disable_tqdm=True
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_data_set,
    eval_dataset=valid_data_set,
)
trainer.train()
