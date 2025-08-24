import os

import torch
import numpy as np
from sklearn.metrics import confusion_matrix


def visda_acc(predict, all_label):
    matrix = confusion_matrix(all_label, predict)
    acc = matrix.diagonal()/matrix.sum(axis=1) * 100
    aacc = acc.mean()
    aa = [str(np.round(i, 2)) for i in acc]
    acc = ' '.join(aa)
    return aacc, acc


def save_model(args, model, model_name=None):
    name = model_name if model_name is not None else args.name
    model_to_save = model.module if hasattr(model, 'module') else model
    model_checkpoint = os.path.join(args.output_dir, "%s.bin" % name)
    torch.save(model_to_save.state_dict(), model_checkpoint)


def update_details_file(args, details: str):
    with open(os.path.join(args.output_dir, args.detail_file_name), 'a') as file:
        file.write('\n' + details + '\n')
