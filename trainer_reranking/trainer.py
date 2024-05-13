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
from model.focal_model import FocalBertFNNClassifier
from model.customized_loss import FocalLoss
from model.bert_fnn import BasicBertFNN

@dataclass
class ModelArguments:
    base_model_name: str = field(default="Bert")
    base_model_path: Optional[str] = field(default='/data/chentianyu/libminer/input/bert-base-uncased_raw')
    evaluate_only: bool = field(default=False)
    to_evaluate_checkpoint: str = field(default=None)
    loss_fct: str = field(default="LabelSmoothingLoss")
    loss_epsilon: float = field(default=0)
    loss_alpha: float = field(default=0.95)
    loss_gamma: float = field(default=2)
    num_labels: float = field(default=1)


parser = HfArgumentParser(ModelArguments)
import os, sys
model_args = parser.parse_args_into_dataclasses()[0]


loss_args = {
    "epsilon": model_args.loss_epsilon,
    "alpha": model_args.loss_alpha,
    "gamma": model_args.loss_gamma
}

loss_fct = FocalLoss(**loss_args)

from_pretrained_args = {
    "pretrained_model_name_or_path": '/home/chentianyu/raw_models/bert-base-uncased/',
    "use_cache": True,
    "num_labels": 1,
    "torch_dtype": torch.float32,
}


parser = argparse.ArgumentParser(description="train bert fnn",
                                 formatter_class=argparse.ArgumentDefaultsHelpFormatter)

parser.add_argument('--data_url', type=str, help='the training data', default="/home/chentianyu/dependency/inputs")
parser.add_argument('--train_url', type=str, help='the path model saved', default="/home/chentianyu/dependency/ckps")
parser.add_argument('--sep_token', type=str, help='sep token of lib corpus', default=" ")
parser.add_argument('--mask_rate', type=float, help='rate of mask lib corpus', default=0)

args, _ = parser.parse_known_args()
input_path = args.data_url
output_path = args.train_url
bert_base_path = '/home/chentianyu/raw_models/bert-base-uncased/'

model = BertFNNClassifier.from_pretrained(**from_pretrained_args)

data_base_path = input_path
train_data_set = ClassifierDataSet(os.path.join(data_base_path, "train.json"),
                                   args.sep_token, args.mask_rate,
                                   bert_base_path, (256, 80, 128, 256))
valid_data_set = ClassifierDataSet(os.path.join(data_base_path, "valid.json"),
                                   args.sep_token, args.mask_rate,
                                   bert_base_path, (256, 80, 128, 256))

print('load dataset succeed!')

from sklearn.metrics import roc_auc_score, precision_recall_curve, matthews_corrcoef, accuracy_score
import numpy as np

def sigmoid(z: np.ndarray):
    return 1 / (1 + np.exp(-z))

def np_divide(a, b):
    # return 0 with divide by 0 while performing element wise divide for np.arrays
    return np.divide(a, b, out=np.zeros_like(a), where=(b != 0))

def f1_score(p, r):
    if p + r < 1e-5:
        return 0.0
    return 2 * p * r / (p + r)

def modified_topk(predictions, labels, k):
    rerank = [item for item in zip(predictions, labels)]
    rerank.sort(reverse=True)

    if sum(labels) == 0:
        return None
    hits = sum([item[1] for item in rerank[:k]])
    prec = hits / min(k, sum(labels))
    rec = hits / sum(labels)
#     print(prec, rec)
    return prec, rec, f1_score(prec, rec)

def compute_metrics(eval_pred: EvalPrediction):
    scores = sigmoid(eval_pred.predictions.reshape(-1, ))
    labels = eval_pred.label_ids.reshape(-1, )
    precision, recall, thresholds = precision_recall_curve(labels, scores)

    # while computing f1 = (2 * precision * recall) / (precision + recall), some element in (precision+recall) will be 0
    f1 = np_divide(2 * precision * recall, precision + recall)
    f1_idx = np.argmax(f1)
    f1_best = f1[f1_idx]
    # mcc = np.array([matthews_corrcoef(labels, (scores >= threshold).astype(int)) for threshold in thresholds])
    # mcc_idx = np.argmax(mcc)
    # mcc_best = mcc[mcc_idx]

    # acc = np.array([accuracy_score(labels, (scores >= threshold).astype(int)) for threshold in thresholds])
    # acc_idx = np.argmax(acc)
    # acc_best = acc[acc_idx]

    auc = roc_auc_score(y_true=labels, y_score=scores)

    k = 1
    res = [modified_topk(predictions, labels, k) for predictions, labels in \
               zip(eval_pred.predictions.reshape(-1, 256),
                   eval_pred.label_ids.reshape(-1, 256))]
    res = [item for item in res if item != None]
    ps, rs, fs = [v[0] for v in res], [v[1] for v in res], [v[2] for v in res]
    print(sum(ps) / len(ps), sum(rs) / len(rs), f1_score(sum(ps) / len(ps), sum(rs) / len(rs)))

    return {"auc": auc,
            # "accuracy_best": acc_best,
            # "accuracy_threshold": thresholds[acc_idx],
            "f1_best": f1_best,
            "f1_threshold": thresholds[f1_idx],
            "precision_1": sum(ps) / len(ps),
            "recall_1": sum(rs) / len(rs),
            "f1_1": f1_score(sum(ps) / len(ps), sum(rs) / len(rs)),
            # "mcc_best": mcc_best,
            # "mcc_threshold": thresholds[mcc_idx],
            "num_samples": len(labels)}


training_args = TrainingArguments(
    output_dir=os.path.join(output_path, time.strftime("BCE_Weight_%Y_%m%d_%H_%M", time.localtime(time.time()))),
    num_train_epochs=10,
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
    learning_rate=2e-5
    # disable_tqdm=True
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_data_set,
    eval_dataset=valid_data_set,
    compute_metrics=compute_metrics
)

trainer.train()