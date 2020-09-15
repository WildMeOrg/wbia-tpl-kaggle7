# -*- coding: utf-8 -*-
from fastprogress import master_bar, progress_bar

# import matplotlib.pyplot as plt
from fastai.vision import *
from fastai.metrics import accuracy
from fastai.basic_data import *

# from skimage.util import montage
import pandas as pd
from torch import optim
import re
import torch
from fastai import *
import torch.nn.functional as F
from torch.nn.parameter import Parameter
import torch.nn as nn
import numpy as np
import torch
import pandas as pd
import torch.nn.functional as F
from torch.nn.parameter import Parameter
import torch.nn as nn
from arch import *


def mapk(preds, targs, k=5):
    if type(preds) is list:
        return torch.cat([mapkfast(p, targs, k=k).view(1) for p in preds]).mean()
    return mapkfast(preds, targs, k=k)


def mapkfast(preds, targs, k=5):
    top_k = preds.topk(k, 1)[1]
    targs = targs.to(preds.device)
    scores = torch.zeros(len(preds), k).float().to(preds.device)
    for kk in range(k):
        scores[:, kk] = (top_k[:, kk] == targs).float() / float(kk + 1)
    return scores.max(dim=1)[0].mean()


def mapksigm(preds, targs, k=5):
    targs = torch.max(targs, dim=1)[1]
    predicted_idxs = preds.sort(descending=True)[1]
    top_k = predicted_idxs[:, :k]
    res = mapk([[t] for t in targs.cpu().numpy()], top_k.cpu().numpy(), k=k)
    return torch.tensor(res)


def top_k_preds(preds, k=5):
    return np.argsort(preds.numpy())[:, ::-1][:, :k]


def top_k_pred_labels(preds, classes, k=5):
    top_k = top_k_preds(preds, k=k)
    labels = []
    for i in range(top_k.shape[0]):
        labels.append(' '.join([classes[idx] for idx in top_k[i]]))
    return labels


def create_submission(preds, data, name, classes=None):
    if not classes:
        classes = data.classes
    sub = pd.DataFrame({'Image': [path.name for path in data.test_ds.x.items]})
    sub['Id'] = top_k_pred_labels(preds, classes, k=5)
    sub.to_csv(f'subs/{name}.csv.gz', index=False, compression='gzip')  # NOQA


def find_softmax_coef(preds, targs, softmax_coefs):
    best_preds = None
    best_score = -1
    best_sc = 0
    for sc in softmax_coefs:
        preds_ = torch.softmax(preds / sc, dim=1).cpu()
        score = mapk(preds_, targs, k=5)
        print(sc, score)
        if score > best_score:
            best_preds = preds_
            best_score = score
            best_sc = sc
    print('best softmax=', best_sc)
    return best_preds, best_score, best_sc


def get_train_features(learn, augment=3):
    # Now features
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    try:
        all_preds0, all_gt0, all_feats0, all_preds20 = get_predictions(
            learn.model, learn.data.train_dl
        )
    except:
        all_preds0, all_gt0, all_feats0, all_preds20 = get_predictions_non_PCB(
            learn.model, learn.data.train_dl
        )
    for i in range(max(augment, 0)):
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        try:
            all_preds00, all_gt00, all_feats00, all_preds200 = get_predictions(
                learn.model, learn.data.train_dl
            )
        except:
            all_preds00, all_gt00, all_feats00, all_preds200 = get_predictions_non_PCB(
                learn.model, learn.data.train_dl
            )
        all_gt0 = torch.cat([all_gt0, all_gt00], dim=0)
        all_feats0 = torch.cat([all_feats0, all_feats00], dim=0)
    train_feats = all_feats0
    train_labels = all_gt0
    return train_feats, train_labels


def find_mixing_proportions(sm_preds, sim, targs):
    best_score = 0
    best_p = -1
    out_preds = None
    for p in np.arange(0.0, 1.01, 0.01):
        out_with_feats = p * sm_preds + (1.0 - p) * sim
        score = mapk(out_with_feats, targs, k=5)
        print(p, score)
        if score > best_score:
            best_score = score
            best_p = p
            out_preds = out_with_feats

    return out_preds, best_p, best_score


def write_augmentations(df, tfms, SZH, SZW, RING_HEADS):
    if not os.path.exists('data/augmentations'):
        os.mkdir('data/augmentations')

    print('Exporting Augmentations:')
    grid = (3, 12)
    for index in range(len(df.Image)):
        if index > 10:
            break
        filename = df.Image[index]
        basename, ext = os.path.splitext(filename)
        path = os.path.join('data/crop_train', filename)
        # image = open_image_grey(path)
        image = open_image(path)
        print('\t', path, image)

        image.save('data/augmentations/%s_original%s' % (basename, ext,))
        for version in range(5):
            image_ = image.apply_tfms(
                tfms[0],
                size=(SZH, SZW),
                resize_method=ResizeMethod.SQUISH,
                padding_mode='zeros',
            )
            c, h, w = image_.shape

            h_ = h // grid[0]
            w_ = w // grid[1]

            for grid_h in range(1, grid[0], 1):
                color = (0.0, 0.0, 1.0)
                for offset in [-1, 0, 1]:
                    image_.data[0, (grid_h * h_) + offset, :] = color[0]
                    image_.data[1, (grid_h * h_) + offset, :] = color[1]
                    image_.data[2, (grid_h * h_) + offset, :] = color[2]

            for grid_w in range(1, grid[1], 1):
                if grid_w % (grid[1] // RING_HEADS) == 0:
                    color = (1.0, 0.0, 0.0)
                else:
                    color = (0.0, 1.0, 0.0)
                for offset in [-1, 0, 1]:
                    image_.data[0, :, (grid_w * w_) + offset] = color[0]
                    image_.data[1, :, (grid_w * w_) + offset] = color[1]
                    image_.data[2, :, (grid_w * w_) + offset] = color[2]

            image_.save(
                'data/augmentations/%s_augmented_%d%s' % (basename, version, ext,)
            )


def get_predictions(model, val_loader):
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    model.eval()
    all_preds = []
    all_confs = []
    all_feats = []
    all_preds2 = []
    all_gt = []
    c = 0
    with torch.no_grad():
        for data1, label in val_loader:
            preds_list, feats_list = model(data1)
            all_preds.append(preds_list[-1].cpu())
            all_preds2.append(torch.stack(preds_list[:-1], -1).cpu())
            all_gt.append(label.cpu())
            all_feats.append(L2Norm()(torch.cat(feats_list, dim=1)).cpu())
            # all_confs.append(confs)
        all_preds = torch.cat(all_preds, dim=0).cpu()
        all_feats = torch.cat(all_feats, dim=0).cpu()
        # all_confs = torch.cat(all_confs, dim=0)
        pred_clc = all_preds.max(dim=1)[1].cpu()
        all_gt = torch.cat(all_gt, dim=0).cpu()
        mp5 = mapk(all_preds, all_gt, k=5).mean()
        acc = (pred_clc == all_gt).float().mean().detach().cpu().item()
        out = f'acc = {acc:.3f}, map5 = {mp5:.3f}'
        print(out)
    return all_preds, all_gt, all_feats, all_preds2


def get_predictions_non_PCB(model, val_loader):
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    model.eval()
    all_preds = []
    all_confs = []
    all_feats = []
    all_preds2 = []
    all_gt = []
    c = 0
    with torch.no_grad():
        for data1, label in val_loader:
            preds, feats, feats2 = model(data1)
            all_preds.append(preds.cpu())
            all_feats.append(
                torch.cat([L2Norm()(feats).cpu(), L2Norm()(feats2).cpu()], dim=1)
            )
            all_gt.append(label.cpu())
            # all_confs.append(confs)
        all_preds = torch.cat(all_preds, dim=0).cpu()
        all_feats = torch.cat(all_feats, dim=0).cpu()
        pred_clc = all_preds.max(dim=1)[1].cpu()
        all_gt = torch.cat(all_gt, dim=0).cpu()
        mp5 = mapk(all_preds, all_gt, k=5).mean()
        acc = (pred_clc == all_gt).float().mean().detach().cpu().item()
        out = f'acc = {acc:.3f}, map5 = {mp5:.3f}'
        print(out)
    return all_preds, all_gt, all_feats, all_preds2


def distance_matrix_vector(anchor, positive, d2_sq):
    """Given batch of anchor descriptors and positive descriptors calculate distance matrix"""
    d1_sq = torch.sum(anchor * anchor, dim=1).unsqueeze(-1)
    eps = 1e-6
    return torch.sqrt(
        (
            d1_sq.repeat(1, positive.size(0))
            + torch.t(d2_sq.repeat(1, anchor.size(0)))
            - 2.0
            * torch.bmm(anchor.unsqueeze(0), torch.t(positive).unsqueeze(0)).squeeze(0)
        )
        + eps
    )


def dm2cm(dm, labels):
    cl = set(labels.detach().cpu().numpy())
    n_cl = len(cl)
    dists = torch.zeros(dm.size(0), n_cl)
    for i in range(n_cl):
        mask = labels == i
        dists[:, i] = dm[:, mask].min(dim=1)[0]
    return dists


def dm2cm_with_idxs(dm, labels):
    cl = set(labels.detach().cpu().numpy())
    n_cl = len(cl)
    dists = torch.zeros(dm.size(0), n_cl)
    idxs = torch.zeros(dm.size(0), n_cl)
    for i in range(n_cl):
        mask = labels == i
        tt = dm[:, mask].min(dim=1)
        dists[:, i] = tt[0]
        iiiii = torch.arange(dm.size(1)).unsqueeze(0).expand_as(dm)[:, mask]
        for j in range(len(tt[1])):
            idxs[j, i] = iiiii[0, tt[1][j]]
    return dists, idxs


def get_train_val_fnames(df, val_list):
    train_fnames = []
    val_fnames = []
    for i in df.Image:
        if i not in val_list:
            train_fnames.append(str(i))
        else:
            val_fnames.append(str(i))
    return train_fnames, val_fnames


def get_shortlist_fnames(distance_matrix_idxs, class_sims, df, val_list):
    train_fnames, val_fnames = get_train_val_fnames(df, val_list)
    best_scores, best_idxs = torch.topk(class_sims, 5, 1)
    shortlist_dict = {}
    for i, fname in enumerate(val_fnames):
        bi = best_idxs[i]
        ci = distance_matrix_idxs[i][bi]
        sl = []
        for iii in ci:
            sl.append(train_fnames[int(iii)])
        shortlist_dict[fname] = sl
    return shortlist_dict


def get_shortlist_fnames_test(distance_matrix_idxs, class_sims, df, learn, val_list):
    train_fnames, val_fnames = get_train_val_fnames(df, val_list)
    train_fnames = val_fnames + train_fnames
    test_fnames = []
    for path in learn.data.test_ds.x.items:
        test_fnames.append(path.name)
    best_scores, best_idxs = torch.topk(class_sims, 5, 1)
    shortlist_dict = {}
    for i, fname in enumerate(test_fnames):
        bi = best_idxs[i]
        ci = distance_matrix_idxs[i][bi]
        sl = []
        for iii in ci:
            sl.append(train_fnames[int(iii)])
        shortlist_dict[fname] = sl
    return shortlist_dict


def batched_dmv(d1, d2):
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    out = torch.zeros(d1.size(0), d2.size(0))
    d2_sq1 = torch.sum(d2 ** 2, dim=1).unsqueeze(-1)
    try:
        out = distance_matrix_vector(
            d1.to(get_device()), d2.to(get_device()), d2_sq1.to(get_device())
        ).cpu()
    except:
        out = distance_matrix_vector(d1, d2, d2_sq1).cpu()
    return out


def open_image_grey(
    fn: PathOrStr, div: bool = True, convert_mode: str = 'RGB', cls: type = Image
) -> Image:
    'Return `Image` object created from image in file `fn`.'
    # fn = getattr(fn, 'path', fn)
    x = PIL.Image.open(fn).convert(convert_mode).convert('LA').convert(convert_mode)
    x = pil2tensor(x, np.float32)
    if div:
        x.div_(255)
    return cls(x)


class ImageListGray(ImageList):
    def open(self, fn: PathOrStr) -> Image:
        return open_image_grey(fn)
        # return open_image(fn)


def topkacc(preds, targs, k=5, mean=True):
    predicted_idxs = preds.sort(descending=True)[1]
    top_k = predicted_idxs[:, :k]
    res = (targs.unsqueeze(1).expand_as(top_k) == top_k).float().max(dim=1)[0]
    if mean:
        res = res.mean()
    return res


def mapkave(preds, targs, k=5):
    pl = len(preds)
    out = torch.stack(preds[: pl - 1], -1).mean(dim=-1)
    return mapk(out, targs, k=k)


def mapktotal(preds, targs, k=5):
    out = preds[-1]
    return mapk(out, targs, k=k)


def map1total(preds, targs):
    return mapktotal(preds, targs, k=1)


def map5total(preds, targs):
    return mapktotal(preds, targs, k=5)


def map12total(preds, targs):
    return mapktotal(preds, targs, k=12)


class Accuracy(Callback):
    """Wrap a `func` in a callback for metrics computation."""

    def __init__(self, func, name, filter_set=None):
        super().__init__()
        # If it's a partial, use func.func
        # name = getattr(func, 'func', func).__name__
        self.func = func
        self.name = name
        self.filter_set = filter_set

    def on_epoch_begin(self, **kwargs):
        self.values = []
        self.targets = []

    def on_batch_end(self, last_output, last_target, **kwargs):
        """Update metric computation with `last_output` and `last_target`."""
        last_preds = last_output[-1]
        value = self.func(last_preds, last_target)
        self.values.append(value)
        self.targets.append(last_target)

    def on_epoch_end(self, last_metrics, **kwargs):
        """Set the final result in `last_metrics`."""
        values = torch.cat(self.values)
        targets = torch.cat(self.targets)

        if self.filter_set is not None:
            values_ = values.tolist()
            targets_ = targets.tolist()

            values_filtered = []
            for value_, target_ in zip(values_, targets_):
                if target_ in self.filter_set:
                    values_filtered.append(value_)
            value_ = sum(values_filtered) / len(values_filtered)
            value = torch.tensor(value_).to(get_device())
        else:
            value = values.mean()

        return add_metrics(last_metrics, value)
