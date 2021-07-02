#####################################################
# Copyright (c) Xuanyi Dong [GitHub D-X-Y], 2020.06 #
#####################################################
# In this python file, it contains the meter classes#
# , which may need to use PyTorch or Numpy.         #
#####################################################
import abc
import torch
import torch.nn.functional as F


class AverageMeter:
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

    def perf_str(self):
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


class CrossEntropyMetric(Metric):
    """The metric for the cross entropy metric."""

    def __init__(self, ignore_batch):
        super(CrossEntropyMetric, self).__init__()
        self._ignore_batch = ignore_batch

    def reset(self):
        self._loss = AverageMeter()

    def __call__(self, predictions, targets):
        if isinstance(predictions, torch.Tensor) and isinstance(targets, torch.Tensor):
            batch, _ = predictions.shape()  # only support 2-D tensor
            max_prob_indexes = torch.argmax(predictions, dim=-1)
            if self._ignore_batch:
                loss = F.cross_entropy(predictions, targets, reduction="sum")
                self._loss.update(loss.item(), 1)
            else:
                loss = F.cross_entropy(predictions, targets, reduction="mean")
                self._loss.update(loss.item(), batch)
            return loss
        else:
            raise NotImplementedError

    def get_info(self):
        return {"loss": self._loss.avg, "score": self._loss.avg * 100}

    def perf_str(self):
        return "ce-loss={:.5f}".format(self._loss.avg)


class Top1AccMetric(Metric):
    """The metric for the top-1 accuracy."""

    def __init__(self, ignore_batch):
        super(Top1AccMetric, self).__init__()
        self._ignore_batch = ignore_batch

    def reset(self):
        self._accuracy = AverageMeter()

    def __call__(self, predictions, targets):
        if isinstance(predictions, torch.Tensor) and isinstance(targets, torch.Tensor):
            batch, _ = predictions.shape()  # only support 2-D tensor
            max_prob_indexes = torch.argmax(predictions, dim=-1)
            corrects = torch.eq(max_prob_indexes, targets)
            accuracy = corrects.float().mean().float()
            if self._ignore_batch:
                self._accuracy.update(accuracy, 1)
            else:
                self._accuracy.update(accuracy, batch)
            return accuracy
        else:
            raise NotImplementedError

    def get_info(self):
        return {"accuracy": self._accuracy.avg, "score": self._accuracy.avg * 100}

    def perf_str(self):
        return "accuracy={:.3f}%".format(self._accuracy.avg * 100)
