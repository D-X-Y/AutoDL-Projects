import numpy as np


def split_str2indexes(string: str, max_check: int, length_limit=5):
    if not isinstance(string, str):
        raise ValueError("Invalid scheme for {:}".format(string))
    srangestr = "".join(string.split())
    indexes = set()
    for srange in srangestr.split(","):
        srange = srange.split("-")
        if len(srange) != 2:
            raise ValueError("invalid srange : {:}".format(srange))
        if length_limit is not None:
            assert (
                len(srange[0]) == len(srange[1]) == length_limit
            ), "invalid srange : {:}".format(srange)
        srange = (int(srange[0]), int(srange[1]))
        if not (0 <= srange[0] <= srange[1] < max_check):
            raise ValueError(
                "{:} vs {:} vs {:}".format(srange[0], srange[1], max_check)
            )
        for i in range(srange[0], srange[1] + 1):
            indexes.add(i)
    return indexes


def show_mean_var(xlist):
    values = np.array(xlist)
    print(
        "{:.2f}".format(values.mean())
        + "$_{{\pm}{"
        + "{:.2f}".format(values.std())
        + "}}$"
    )
