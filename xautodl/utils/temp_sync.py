# To be deleted.
import copy
import torch

from xlayers.super_core import SuperSequential, SuperMLPv1
from xlayers.super_core import SuperSimpleNorm
from xlayers.super_core import SuperLinear


def optimize_fn(xs, ys, device="cpu", max_iter=2000, max_lr=0.1):
    xs = torch.FloatTensor(xs).view(-1, 1).to(device)
    ys = torch.FloatTensor(ys).view(-1, 1).to(device)

    model = SuperSequential(
        SuperSimpleNorm(xs.mean().item(), xs.std().item()),
        SuperLinear(1, 200),
        torch.nn.LeakyReLU(),
        SuperLinear(200, 100),
        torch.nn.LeakyReLU(),
        SuperLinear(100, 1),
    ).to(device)
    model.train()
    optimizer = torch.optim.Adam(model.parameters(), lr=max_lr, amsgrad=True)
    loss_func = torch.nn.MSELoss()
    lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(
        optimizer,
        milestones=[
            int(max_iter * 0.25),
            int(max_iter * 0.5),
            int(max_iter * 0.75),
        ],
        gamma=0.3,
    )

    best_loss, best_param = None, None
    for _iter in range(max_iter):
        preds = model(xs)

        optimizer.zero_grad()
        loss = loss_func(preds, ys)
        loss.backward()
        optimizer.step()
        lr_scheduler.step()

        if best_loss is None or best_loss > loss.item():
            best_loss = loss.item()
            best_param = copy.deepcopy(model.state_dict())

        # print('loss={:}, best-loss={:}'.format(loss.item(), best_loss))
    model.load_state_dict(best_param)
    return model, loss_func, best_loss


def evaluate_fn(model, xs, ys, loss_fn, device="cpu"):
    with torch.no_grad():
        inputs = torch.FloatTensor(xs).view(-1, 1).to(device)
        ys = torch.FloatTensor(ys).view(-1, 1).to(device)
        preds = model(inputs)
        loss = loss_fn(preds, ys)
        preds = preds.view(-1).cpu().numpy()
    return preds, loss.item()
