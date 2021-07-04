##################################################
# Copyright (c) Xuanyi Dong [GitHub D-X-Y], 2021 #
##################################################
from __future__ import division
from __future__ import print_function

import os, math, random
from collections import OrderedDict
import numpy as np
import pandas as pd
from typing import Text, Union
import copy
from functools import partial
from typing import Optional, Text

from qlib.utils import get_or_create_path
from qlib.log import get_module_logger

import torch
import torch.nn.functional as F
import torch.optim as optim
import torch.utils.data as th_data

from xautodl.xmisc import AverageMeter
from xautodl.xmisc import count_parameters

from xautodl.xlayers import super_core
from .transformers import DEFAULT_NET_CONFIG
from .transformers import get_transformer


from qlib.model.base import Model
from qlib.data.dataset import DatasetH
from qlib.data.dataset.handler import DataHandlerLP


DEFAULT_OPT_CONFIG = dict(
    epochs=200,
    lr=0.001,
    batch_size=2000,
    early_stop=20,
    loss="mse",
    optimizer="adam",
    num_workers=4,
)


def train_or_test_epoch(
    xloader, model, loss_fn, metric_fn, is_train, optimizer, device
):
    if is_train:
        model.train()
    else:
        model.eval()
    score_meter, loss_meter = AverageMeter(), AverageMeter()
    for ibatch, (feats, labels) in enumerate(xloader):
        feats, labels = feats.to(device), labels.to(device)
        # forward the network
        preds = model(feats)
        loss = loss_fn(preds, labels)
        with torch.no_grad():
            score = metric_fn(preds, labels)
            loss_meter.update(loss.item(), feats.size(0))
            score_meter.update(score.item(), feats.size(0))
        # optimize the network
        if is_train and optimizer is not None:
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_value_(model.parameters(), 3.0)
            optimizer.step()
    return loss_meter.avg, score_meter.avg


class QuantTransformer(Model):
    """Transformer-based Quant Model"""

    def __init__(
        self, net_config=None, opt_config=None, metric="", GPU=0, seed=None, **kwargs
    ):
        # Set logger.
        self.logger = get_module_logger("QuantTransformer")
        self.logger.info("QuantTransformer PyTorch version...")

        # set hyper-parameters.
        self.net_config = net_config or DEFAULT_NET_CONFIG
        self.opt_config = opt_config or DEFAULT_OPT_CONFIG
        self.metric = metric
        self.device = torch.device(
            "cuda:{:}".format(GPU) if torch.cuda.is_available() and GPU >= 0 else "cpu"
        )
        self.seed = seed

        self.logger.info(
            "Transformer parameters setting:"
            "\nnet_config : {:}"
            "\nopt_config : {:}"
            "\nmetric     : {:}"
            "\ndevice     : {:}"
            "\nseed       : {:}".format(
                self.net_config,
                self.opt_config,
                self.metric,
                self.device,
                self.seed,
            )
        )

        if self.seed is not None:
            random.seed(self.seed)
            np.random.seed(self.seed)
            torch.manual_seed(self.seed)
            if self.use_gpu:
                torch.cuda.manual_seed(self.seed)
                torch.cuda.manual_seed_all(self.seed)

        self.model = get_transformer(self.net_config)
        self.model.set_super_run_type(super_core.SuperRunMode.FullModel)
        self.logger.info("model: {:}".format(self.model))
        self.logger.info("model size: {:.3f} MB".format(count_parameters(self.model)))

        if self.opt_config["optimizer"] == "adam":
            self.train_optimizer = optim.Adam(
                self.model.parameters(), lr=self.opt_config["lr"]
            )
        elif self.opt_config["optimizer"] == "adam":
            self.train_optimizer = optim.SGD(
                self.model.parameters(), lr=self.opt_config["lr"]
            )
        else:
            raise NotImplementedError(
                "optimizer {:} is not supported!".format(optimizer)
            )

        self.fitted = False
        self.model.to(self.device)

    @property
    def use_gpu(self):
        return self.device != torch.device("cpu")

    def to(self, device):
        if device is None:
            device = "cpu"
        self.device = device
        self.model.to(self.device)
        # move the optimizer
        for param in self.train_optimizer.state.values():
            # Not sure there are any global tensors in the state dict
            if isinstance(param, torch.Tensor):
                param.data = param.data.to(device)
                if param._grad is not None:
                    param._grad.data = param._grad.data.to(device)
            elif isinstance(param, dict):
                for subparam in param.values():
                    if isinstance(subparam, torch.Tensor):
                        subparam.data = subparam.data.to(device)
                        if subparam._grad is not None:
                            subparam._grad.data = subparam._grad.data.to(device)

    def loss_fn(self, pred, label):
        mask = ~torch.isnan(label)
        if self.opt_config["loss"] == "mse":
            return F.mse_loss(pred[mask], label[mask])
        else:
            raise ValueError("unknown loss `{:}`".format(self.loss))

    def metric_fn(self, pred, label):
        # the metric score : higher is better
        if self.metric == "" or self.metric == "loss":
            return -self.loss_fn(pred, label)
        else:
            raise ValueError("unknown metric `{:}`".format(self.metric))

    def fit(
        self,
        dataset: DatasetH,
        save_dir: Optional[Text] = None,
    ):
        def _prepare_dataset(df_data):
            return th_data.TensorDataset(
                torch.from_numpy(df_data["feature"].values).float(),
                torch.from_numpy(df_data["label"].values).squeeze().float(),
            )

        def _prepare_loader(dataset, shuffle):
            return th_data.DataLoader(
                dataset,
                batch_size=self.opt_config["batch_size"],
                drop_last=False,
                pin_memory=True,
                num_workers=self.opt_config["num_workers"],
                shuffle=shuffle,
            )

        df_train, df_valid, df_test = dataset.prepare(
            ["train", "valid", "test"],
            col_set=["feature", "label"],
            data_key=DataHandlerLP.DK_L,
        )
        train_dataset, valid_dataset, test_dataset = (
            _prepare_dataset(df_train),
            _prepare_dataset(df_valid),
            _prepare_dataset(df_test),
        )
        train_loader, valid_loader, test_loader = (
            _prepare_loader(train_dataset, True),
            _prepare_loader(valid_dataset, False),
            _prepare_loader(test_dataset, False),
        )

        save_dir = get_or_create_path(save_dir, return_dir=True)
        self.logger.info(
            "Fit procedure for [{:}] with save path={:}".format(
                self.__class__.__name__, save_dir
            )
        )

        def _internal_test(ckp_epoch=None, results_dict=None):
            with torch.no_grad():
                shared_kwards = {
                    "model": self.model,
                    "loss_fn": self.loss_fn,
                    "metric_fn": self.metric_fn,
                    "is_train": False,
                    "optimizer": None,
                    "device": self.device,
                }
                train_loss, train_score = train_or_test_epoch(
                    train_loader, **shared_kwards
                )
                valid_loss, valid_score = train_or_test_epoch(
                    valid_loader, **shared_kwards
                )
                test_loss, test_score = train_or_test_epoch(
                    test_loader, **shared_kwards
                )
                xstr = (
                    "train-score={:.6f}, valid-score={:.6f}, test-score={:.6f}".format(
                        train_score, valid_score, test_score
                    )
                )
                if ckp_epoch is not None and isinstance(results_dict, dict):
                    results_dict["train"][ckp_epoch] = train_score
                    results_dict["valid"][ckp_epoch] = valid_score
                    results_dict["test"][ckp_epoch] = test_score
                return dict(train=train_score, valid=valid_score, test=test_score), xstr

        # Pre-fetch the potential checkpoints
        ckp_path = os.path.join(save_dir, "{:}.pth".format(self.__class__.__name__))
        if os.path.exists(ckp_path):
            ckp_data = torch.load(ckp_path, map_location=self.device)
            stop_steps, best_score, best_epoch = (
                ckp_data["stop_steps"],
                ckp_data["best_score"],
                ckp_data["best_epoch"],
            )
            start_epoch, best_param = ckp_data["start_epoch"], ckp_data["best_param"]
            results_dict = ckp_data["results_dict"]
            self.model.load_state_dict(ckp_data["net_state_dict"])
            self.train_optimizer.load_state_dict(ckp_data["opt_state_dict"])
            self.logger.info("Resume from existing checkpoint: {:}".format(ckp_path))
        else:
            stop_steps, best_score, best_epoch = 0, -np.inf, -1
            start_epoch, best_param = 0, None
            results_dict = dict(
                train=OrderedDict(), valid=OrderedDict(), test=OrderedDict()
            )
            _, eval_str = _internal_test(-1, results_dict)
            self.logger.info(
                "Training from scratch, metrics@start: {:}".format(eval_str)
            )

        for iepoch in range(start_epoch, self.opt_config["epochs"]):
            self.logger.info(
                "Epoch={:03d}/{:03d} ::==>> Best valid @{:03d} ({:.6f})".format(
                    iepoch, self.opt_config["epochs"], best_epoch, best_score
                )
            )
            train_loss, train_score = train_or_test_epoch(
                train_loader,
                self.model,
                self.loss_fn,
                self.metric_fn,
                True,
                self.train_optimizer,
                self.device,
            )
            self.logger.info(
                "Training :: loss={:.6f}, score={:.6f}".format(train_loss, train_score)
            )

            current_eval_scores, eval_str = _internal_test(iepoch, results_dict)
            self.logger.info("Evaluating :: {:}".format(eval_str))

            if current_eval_scores["valid"] > best_score:
                stop_steps, best_epoch, best_score = (
                    0,
                    iepoch,
                    current_eval_scores["valid"],
                )
                best_param = copy.deepcopy(self.model.state_dict())
            else:
                stop_steps += 1
                if stop_steps >= self.opt_config["early_stop"]:
                    self.logger.info(
                        "early stop at {:}-th epoch, where the best is @{:}".format(
                            iepoch, best_epoch
                        )
                    )
                    break
            save_info = dict(
                net_config=self.net_config,
                opt_config=self.opt_config,
                net_state_dict=self.model.state_dict(),
                opt_state_dict=self.train_optimizer.state_dict(),
                best_param=best_param,
                stop_steps=stop_steps,
                best_score=best_score,
                best_epoch=best_epoch,
                results_dict=results_dict,
                start_epoch=iepoch + 1,
            )
            torch.save(save_info, ckp_path)
        self.logger.info(
            "The best score: {:.6f} @ {:02d}-th epoch".format(best_score, best_epoch)
        )
        self.model.load_state_dict(best_param)
        _, eval_str = _internal_test("final", results_dict)
        self.logger.info("Reload the best parameter :: {:}".format(eval_str))

        if self.use_gpu:
            with torch.cuda.device(self.device):
                torch.cuda.empty_cache()
        self.fitted = True

    def predict(self, dataset: DatasetH, segment: Union[Text, slice] = "test"):
        if not self.fitted:
            raise ValueError("The model is not fitted yet!")
        x_test = dataset.prepare(
            segment, col_set="feature", data_key=DataHandlerLP.DK_I
        )
        index = x_test.index

        with torch.no_grad():
            self.model.eval()
            x_values = x_test.values
            sample_num, batch_size = x_values.shape[0], self.opt_config["batch_size"]
            preds = []
            for begin in range(sample_num)[::batch_size]:
                if sample_num - begin < batch_size:
                    end = sample_num
                else:
                    end = begin + batch_size
                x_batch = torch.from_numpy(x_values[begin:end]).float().to(self.device)
                with torch.no_grad():
                    pred = self.model(x_batch).detach().cpu().numpy()
                preds.append(pred)
        return pd.Series(np.concatenate(preds), index=index)
