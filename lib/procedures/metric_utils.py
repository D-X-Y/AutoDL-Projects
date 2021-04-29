#####################################################
# Copyright (c) Xuanyi Dong [GitHub D-X-Y], 2020.04 #
#####################################################
import abc
import numpy as np
import torch


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0.0
        self.avg = 0.0
        self.sum = 0.0
        self.count = 0.0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def __repr__(self):
        return "{name}(val={val}, avg={avg}, count={count})".format(
            name=self.__class__.__name__, **self.__dict__
        )


class Metric(abc.ABC):
    """The default meta metric class."""

    def __init__(self):
        self.reset()

    def reset(self):
        raise NotImplementedError

    def __call__(self, predictions, targets):
        raise NotImplementedError

    def get_info(self):
        raise NotImplementedError

    def __repr__(self):
        return "{name}({inner})".format(
            name=self.__class__.__name__, inner=self.inner_repr()
        )

    def inner_repr(self):
        return ""


class ComposeMetric(Metric):
    """The composed metric class."""

    def __init__(self, *metric_list):
        self.reset()
        for metric in metric_list:
            self.append(metric)

    def reset(self):
        self._metric_list = []

    def append(self, metric):
        if not isinstance(metric, Metric):
            raise ValueError(
                "The input metric is not correct: {:}".format(type(metric))
            )
        self._metric_list.append(metric)

    def __len__(self):
        return len(self._metric_list)

    def __call__(self, predictions, targets):
        results = list()
        for metric in self._metric_list:
            results.append(metric(predictions, targets))
        return results

    def get_info(self):
        results = dict()
        for metric in self._metric_list:
            for key, value in metric.get_info().items():
                results[key] = value
        return results

    def inner_repr(self):
        xlist = []
        for metric in self._metric_list:
            xlist.append(str(metric))
        return ",".join(xlist)


class MSEMetric(Metric):
    """The metric for mse."""

    def reset(self):
        self._mse = AverageMeter()

    def __call__(self, predictions, targets):
        if isinstance(predictions, torch.Tensor) and isinstance(targets, torch.Tensor):
            batch = predictions.shape[0]
            loss = torch.nn.functional.mse_loss(predictions.data, targets.data)
            loss = loss.item()
            self._mse.update(loss, batch)
            return loss
        else:
            raise NotImplementedError

    def get_info(self):
        return {"mse": self._mse.avg}


class SaveMetric(Metric):
    """The metric for mse."""

    def reset(self):
        self._predicts = []

    def __call__(self, predictions, targets=None):
        if isinstance(predictions, torch.Tensor):
            predicts = predictions.cpu().numpy()
            self._predicts.append(predicts)
            return predicts
        else:
            raise NotImplementedError

    def get_info(self):
        all_predicts = np.concatenate(self._predicts)
        return {"predictions": all_predicts}
