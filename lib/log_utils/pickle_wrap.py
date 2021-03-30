#####################################################
# Copyright (c) Xuanyi Dong [GitHub D-X-Y], 2019.01 #
#####################################################
import pickle
from pathlib import Path


def pickle_save(obj, path):
    file_path = Path(path)
    file_dir = file_path.parent
    file_dir.mkdir(parents=True, exist_ok=True)
    with file_path.open("wb") as f:
        pickle.dump(obj, f)


def pickle_load(path):
    if not Path(path).exists():
        raise ValueError("{:} does not exists".format(path))
    with Path(path).open("rb") as f:
        data = pickle.load(f)
    return data
