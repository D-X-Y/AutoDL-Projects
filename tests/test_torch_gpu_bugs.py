#####################################################
# Copyright (c) Xuanyi Dong [GitHub D-X-Y], 2021.03 #
#####################################################
# pytest ./tests/test_torch_gpu_bugs.py::test_create
#
# CUDA_VISIBLE_DEVICES="" pytest ./tests/test_torch_gpu_bugs.py::test_load
#####################################################
import os, sys, time, torch
import pickle
import tempfile
from pathlib import Path

lib_dir = (Path(__file__).parent / ".." / "lib").resolve()
print("library path: {:}".format(lib_dir))
if str(lib_dir) not in sys.path:
    sys.path.insert(0, str(lib_dir))

from trade_models.quant_transformer import QuantTransformer


def test_create():
    """Test the basic quant-model."""
    if not torch.cuda.is_available():
        return
    quant_model = QuantTransformer(GPU=0)
    temp_dir = lib_dir / ".." / "tests" / ".pytest_cache"
    temp_dir.mkdir(parents=True, exist_ok=True)
    temp_file = temp_dir / "quant-model.pkl"
    with temp_file.open("wb") as f:
        # quant_model.to(None)
        quant_model.to("cpu")
        # del quant_model.model
        # del quant_model.train_optimizer
        pickle.dump(quant_model, f)
    print("save into {:}".format(temp_file))


def test_load():
    temp_file = lib_dir / ".." / "tests" / ".pytest_cache" / "quant-model.pkl"
    with temp_file.open("rb") as f:
        model = pickle.load(f)
        print(model.model)
        print(model.train_optimizer)
