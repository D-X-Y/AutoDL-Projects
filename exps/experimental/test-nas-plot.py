# python ./exps/vis/test.py
import os, sys, random
from pathlib import Path
from copy import deepcopy
import torch
import numpy as np
from collections import OrderedDict

lib_dir = (Path(__file__).parent / ".." / ".." / "lib").resolve()
if str(lib_dir) not in sys.path:
    sys.path.insert(0, str(lib_dir))

from nas_201_api import NASBench201API as API


def test_nas_api():
    from nas_201_api import ArchResults

    xdata = torch.load(
        "/home/dxy/FOR-RELEASE/NAS-Projects/output/NAS-BENCH-201-4/simplifies/architectures/000157-FULL.pth"
    )
    for key in ["full", "less"]:
        print("\n------------------------- {:} -------------------------".format(key))
        archRes = ArchResults.create_from_state_dict(xdata[key])
        print(archRes)
        print(archRes.arch_idx_str())
        print(archRes.get_dataset_names())
        print(archRes.get_comput_costs("cifar10-valid"))
        # get the metrics
        print(archRes.get_metrics("cifar10-valid", "x-valid", None, False))
        print(archRes.get_metrics("cifar10-valid", "x-valid", None, True))
        print(archRes.query("cifar10-valid", 777))


OPS = ["skip-connect", "conv-1x1", "conv-3x3", "pool-3x3"]
COLORS = ["chartreuse", "cyan", "navyblue", "chocolate1"]


def plot(filename):
    from graphviz import Digraph

    g = Digraph(
        format="png",
        edge_attr=dict(fontsize="20", fontname="times"),
        node_attr=dict(
            style="filled",
            shape="rect",
            align="center",
            fontsize="20",
            height="0.5",
            width="0.5",
            penwidth="2",
            fontname="times",
        ),
        engine="dot",
    )
    g.body.extend(["rankdir=LR"])

    steps = 5
    for i in range(0, steps):
        if i == 0:
            g.node(str(i), fillcolor="darkseagreen2")
        elif i + 1 == steps:
            g.node(str(i), fillcolor="palegoldenrod")
        else:
            g.node(str(i), fillcolor="lightblue")

    for i in range(1, steps):
        for xin in range(i):
            op_i = random.randint(0, len(OPS) - 1)
            # g.edge(str(xin), str(i), label=OPS[op_i], fillcolor=COLORS[op_i])
            g.edge(
                str(xin),
                str(i),
                label=OPS[op_i],
                color=COLORS[op_i],
                fillcolor=COLORS[op_i],
            )
            # import pdb; pdb.set_trace()
    g.render(filename, cleanup=True, view=False)


def test_auto_grad():
    class Net(torch.nn.Module):
        def __init__(self, iS):
            super(Net, self).__init__()
            self.layer = torch.nn.Linear(iS, 1)

        def forward(self, inputs):
            outputs = self.layer(inputs)
            outputs = torch.exp(outputs)
            return outputs.mean()

    net = Net(10)
    inputs = torch.rand(256, 10)
    loss = net(inputs)
    first_order_grads = torch.autograd.grad(
        loss, net.parameters(), retain_graph=True, create_graph=True
    )
    first_order_grads = torch.cat([x.view(-1) for x in first_order_grads])
    second_order_grads = []
    for grads in first_order_grads:
        s_grads = torch.autograd.grad(grads, net.parameters())
        second_order_grads.append(s_grads)


def test_one_shot_model(ckpath, use_train):
    from models import get_cell_based_tiny_net, get_search_spaces
    from datasets import get_datasets, SearchDataset
    from config_utils import load_config, dict2config
    from utils.nas_utils import evaluate_one_shot

    use_train = int(use_train) > 0
    # ckpath = 'output/search-cell-nas-bench-201/DARTS-V1-cifar10/checkpoint/seed-11416-basic.pth'
    # ckpath = 'output/search-cell-nas-bench-201/DARTS-V1-cifar10/checkpoint/seed-28640-basic.pth'
    print("ckpath : {:}".format(ckpath))
    ckp = torch.load(ckpath)
    xargs = ckp["args"]
    train_data, valid_data, xshape, class_num = get_datasets(
        xargs.dataset, xargs.data_path, -1
    )
    # config = load_config(xargs.config_path, {'class_num': class_num, 'xshape': xshape}, None)
    config = load_config(
        "./configs/nas-benchmark/algos/DARTS.config",
        {"class_num": class_num, "xshape": xshape},
        None,
    )
    if xargs.dataset == "cifar10":
        cifar_split = load_config("configs/nas-benchmark/cifar-split.txt", None, None)
        xvalid_data = deepcopy(train_data)
        xvalid_data.transform = valid_data.transform
        valid_loader = torch.utils.data.DataLoader(
            xvalid_data,
            batch_size=2048,
            sampler=torch.utils.data.sampler.SubsetRandomSampler(cifar_split.valid),
            num_workers=12,
            pin_memory=True,
        )
    else:
        raise ValueError("invalid dataset : {:}".format(xargs.dataseet))
    search_space = get_search_spaces("cell", xargs.search_space_name)
    model_config = dict2config(
        {
            "name": "SETN",
            "C": xargs.channel,
            "N": xargs.num_cells,
            "max_nodes": xargs.max_nodes,
            "num_classes": class_num,
            "space": search_space,
            "affine": False,
            "track_running_stats": True,
        },
        None,
    )
    search_model = get_cell_based_tiny_net(model_config)
    search_model.load_state_dict(ckp["search_model"])
    search_model = search_model.cuda()
    api = API("/home/dxy/.torch/NAS-Bench-201-v1_0-e61699.pth")
    archs, probs, accuracies = evaluate_one_shot(
        search_model, valid_loader, api, use_train
    )


if __name__ == "__main__":
    # test_nas_api()
    # for i in range(200): plot('{:04d}'.format(i))
    # test_auto_grad()
    test_one_shot_model(sys.argv[1], sys.argv[2])
