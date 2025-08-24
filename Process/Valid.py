import logging

from tqdm import tqdm
import torch
import numpy as np

from .Metrics import AverageMeter, simple_accuracy
from Utils.utils import visda_acc


logger = logging.getLogger(__name__)


def valid(args, model, test_loader, device):
    eval_losses = AverageMeter()

    model.eval()
    all_preds, all_label = [], []

    model.to(device)

    epoch_iterator = tqdm(test_loader,
                          desc="Validating... (loss=X.X)",
                          bar_format="{l_bar}{r_bar}",
                          dynamic_ncols=True,
                          disable=len(args.gpu_ids) > 1)

    loss_fct = torch.nn.CrossEntropyLoss()
    for step, batch in enumerate(epoch_iterator):
        x, y, _, _ = batch
        x = x.to(device)
        y = y.to(device)
        with torch.no_grad():
            logits = model(x, return_features_only=True)

            eval_loss = loss_fct(logits, y)
            eval_losses.update(eval_loss.item())

            preds = torch.argmax(logits, dim=-1)

        if len(all_preds) == 0:
            all_preds.append(preds.detach().cpu().numpy())
            all_label.append(y.detach().cpu().numpy())
        else:
            all_preds[0] = np.append(
                all_preds[0], preds.detach().cpu().numpy(), axis=0
            )
            all_label[0] = np.append(
                all_label[0], y.detach().cpu().numpy(), axis=0
            )
        epoch_iterator.set_description("Validating... (loss=%2.5f)" % eval_losses.val)

    all_preds, all_label = all_preds[0], all_label[0]
    if args.dataset == 'visda17':
        accuracy, classWise_acc = visda_acc(all_preds, all_label)
    else:
        accuracy = simple_accuracy(all_preds, all_label)

    if args.dataset == 'visda17':
        logger.info(classWise_acc)

    if args.dataset == 'visda17':
        return accuracy, classWise_acc
    else:
        return accuracy, None