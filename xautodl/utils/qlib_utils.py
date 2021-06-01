import os
import numpy as np
from typing import List, Text
from collections import defaultdict, OrderedDict


class QResult:
    """A class to maintain the results of a qlib experiment."""

    def __init__(self, name):
        self._result = defaultdict(list)
        self._name = name
        self._recorder_paths = []
        self._date2ICs = []

    def append(self, key, value):
        self._result[key].append(value)

    def append_path(self, xpath):
        self._recorder_paths.append(xpath)

    def append_date2ICs(self, date2IC):
        if self._date2ICs:  # not empty
            keys = sorted(list(date2IC.keys()))
            pre_keys = sorted(list(self._date2ICs[0].keys()))
            assert len(keys) == len(pre_keys)
            for i, (x, y) in enumerate(zip(keys, pre_keys)):
                assert x == y, "[{:}] {:} vs {:}".format(i, x, y)
        self._date2ICs.append(date2IC)

    def find_all_dates(self):
        dates = self._date2ICs[-1].keys()
        return sorted(list(dates))

    def get_IC_by_date(self, date, scale=1.0):
        values = []
        for date2IC in self._date2ICs:
            values.append(date2IC[date] * scale)
        return float(np.mean(values)), float(np.std(values))

    @property
    def name(self):
        return self._name

    @property
    def paths(self):
        return self._recorder_paths

    @property
    def result(self):
        return self._result

    @property
    def keys(self):
        return list(self._result.keys())

    def __len__(self):
        return len(self._result)

    def __repr__(self):
        return "{name}({xname}, {num} metrics)".format(
            name=self.__class__.__name__, xname=self.name, num=len(self.result)
        )

    def __getitem__(self, key):
        if key not in self._result:
            raise ValueError(
                "Invalid key {:}, please use one of {:}".format(key, self.keys)
            )
        values = self._result[key]
        return float(np.mean(values))

    def update(self, metrics, filter_keys=None):
        for key, value in metrics.items():
            if filter_keys is not None and key in filter_keys:
                key = filter_keys[key]
            elif filter_keys is not None:
                continue
            self.append(key, value)

    @staticmethod
    def full_str(xstr, space):
        xformat = "{:" + str(space) + "s}"
        return xformat.format(str(xstr))

    @staticmethod
    def merge_dict(dict_list):
        new_dict = dict()
        for xkey in dict_list[0].keys():
            values = [x for xdict in dict_list for x in xdict[xkey]]
            new_dict[xkey] = values
        return new_dict

    def info(
        self,
        keys: List[Text],
        separate: Text = "& ",
        space: int = 20,
        verbose: bool = True,
        version: str = "v1",
    ):
        avaliable_keys = []
        for key in keys:
            if key not in self.result:
                print("There are invalid key [{:}].".format(key))
            else:
                avaliable_keys.append(key)
        head_str = separate.join([self.full_str(x, space) for x in avaliable_keys])
        values = []
        for key in avaliable_keys:
            if "IR" in key:
                current_values = [x * 100 for x in self._result[key]]
            else:
                current_values = self._result[key]
            mean = np.mean(current_values)
            std = np.std(current_values)
            if version == "v0":
                values.append("{:.2f} $\pm$ {:.2f}".format(mean, std))
            elif version == "v1":
                values.append(
                    "{:.2f}".format(mean) + " \\subs{" + "{:.2f}".format(std) + "}"
                )
            else:
                raise ValueError("Unknown version")
        value_str = separate.join([self.full_str(x, space) for x in values])
        if verbose:
            print(head_str)
            print(value_str)
        return head_str, value_str
