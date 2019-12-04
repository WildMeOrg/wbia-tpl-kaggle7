from fastprogress import master_bar, progress_bar
from fastai.vision import *
from fastai.metrics import accuracy
from fastai.basic_data import *
import pandas as pd
from torch import optim
import re
import torch
from fastai import *
import torch.nn.functional as F
from torch.nn.parameter import Parameter
import torch.nn as nn
import pretrainedmodels
from collections import OrderedDict
import math
from arch import *
from utils import *
from losses import *
import torchvision
import os


SZH, SZW = 400, 1550
BS = 16
NUM_WORKERS = 10
SEED = 0
SAVE_TRAIN_FEATS = True
SAVE_TEST_MATRIX = True
RING_ALPHA = 0.01
RING_HEADS = 4
GEM_CONST = 5.0


class CustomPCBNetwork(nn.Module):
    def __init__(self, new_model):
        super().__init__()
        self.cnn =  new_model.features
        self.head = PCBRingHead2(num_classes, 512, RING_HEADS, 1920, GEM_CONST)

    def forward(self, x):
        x = self.cnn(x)
        out = self.head(x)
        return out


class Accuracy(Callback):
    """Wrap a `func` in a callback for metrics computation."""

    def __init__(self, func, name, singletons=None):
        super().__init__()
        # If it's a partial, use func.func
        # name = getattr(func, 'func', func).__name__
        self.func = func
        self.name = name
        self.singletons = singletons

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

        if self.singletons is not None:
            values_ = values.tolist()
            targets_ = targets.tolist()

            values_singletons = []
            for value_, target_ in zip(values_, targets_):
                if target_ in self.singletons:
                    values_singletons.append(value_)
            value_ = sum(values_singletons) / len(values_singletons)
            value = torch.tensor(value_).to(get_device())
        else:
            value = values.mean()

        return add_metrics(last_metrics, value)


@dataclass
class RingLoss(Callback):
    """`Callback` that regroups lr adjustment to seq_len, AR and TAR."""
    learn:Learner
    alpha:float=RING_ALPHA

    def on_loss_begin(self, last_output:Tuple[list,list], **kwargs):
        "Save the extra outputs for later and only returns the true output."
        self.feature_out = last_output[1]
        return {'last_output': last_output[0]}

    def on_backward_begin(self,
                          last_loss:Rank0Tensor,
                          last_input:list,
                          last_target:Tensor,
                          **kwargs):
        x_list = self.feature_out
        ring_list = self.learn.model.module.head.rings
        num_clf = len(ring_list)
        loss = None
        for cc in range(num_clf):
            x = x_list[cc]
            R = ring_list[cc]
            x_norm = x.pow(2).sum(dim=1).pow(0.5)
            diff = torch.mean(torch.abs(x_norm - R.expand_as(x_norm))**2)
            if loss is None:
                loss = diff.mean()
            else:
                loss = loss + diff.mean()
        loss = (self.alpha * loss).sum()
        last_loss += loss
        return {'last_loss': last_loss}


print("torch.cuda.is_available:", torch.cuda.is_available())

df = pd.read_csv('data/train.csv')
val_fns = pd.read_pickle('data/val_fns')

ids = list(df.Id)
count_dict = {}
singleton_ids = set([])
for id_ in set(ids):
    count = ids.count(id_)
    if count not in count_dict:
        count_dict[count] = 0
    count_dict[count] += 1
    if count <= 2:
        singleton_ids.add(id_)

fn2label = {row[1].Image: row[1].Id for row in df.iterrows()}
path2fn = lambda path: re.search('[\w-]*\.jpg$', path).group(0)

num_classes = len(set(df.Id))
# num_epochs = 50
# num_epochs = 25

name = f'DenseNet201-GeM-{GEM_CONST}-PCB{RING_HEADS}-{SZH}-{SZW}-Ring-{RING_ALPHA}_RELU'

tfms = (
    [
        RandTransform(tfm=brightness, kwargs={'change': (0.2, 0.8)}),
        RandTransform(tfm=contrast, kwargs={'scale': (0.5, 1.5)}),
        RandTransform(tfm=symmetric_warp, kwargs={'magnitude': (-0.01, 0.01)}),
        RandTransform(tfm=rotate, kwargs={'degrees': (-1.0, 1.0)}),
        RandTransform(tfm=zoom, kwargs={'scale': (1.0, 1.05), 'row_pct': 0.5, 'col_pct': 0.5}),
        # RandTransform(tfm=flip_lr, kwargs={}, p=0.5),
    ],
    [
        # RandTransform(tfm=brightness, kwargs={'change': (0.3, 0.8)}),
        # RandTransform(tfm=contrast, kwargs={'scale': (0.5, 1.5)}),
        # RandTransform(tfm=symmetric_warp, kwargs={'magnitude': (-0.02, 0.02)}),
        # RandTransform(tfm=rotate, kwargs={'degrees': (-1.0, 1.0)}),
        # RandTransform(tfm=zoom, kwargs={'scale': (0.9, 1.0), 'row_pct': 0.5, 'col_pct': 0.5}),
        # RandTransform(tfm=flip_lr, kwargs={}, p=0.5),
    ],
)

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

    image.save('data/augmentations/%s_original%s' % (basename, ext, ))
    for version in range(5):
        image_ = image.apply_tfms(tfms[0], size=(SZH, SZW), resize_method=ResizeMethod.SQUISH, padding_mode='zeros')
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
        image_.save('data/augmentations/%s_augmented_%d%s' % (basename, version, ext, ))

data = (
    ImageListGray
    .from_df(df, 'data/crop_train', cols=['Image'])
    .split_by_valid_func(lambda path: path2fn(path) in val_fns)
    .label_from_func(lambda path: fn2label[path2fn(path)])
    .add_test(ImageList.from_folder('data/crop_test'))
    .transform(tfms, size=(SZH, SZW), resize_method=ResizeMethod.SQUISH, padding_mode='zeros')
    .databunch(bs=BS, num_workers=NUM_WORKERS, path='data')
    .normalize(imagenet_stats)
)

classes = data.classes
singleton_idx = set([])
for index, id_ in enumerate(classes):
    if id_ in singleton_ids:
        singleton_idx.add(index)

network_model = CustomPCBNetwork(torchvision.models.densenet201(pretrained=True))

if torch.cuda.device_count() > 1:
    print("Using", torch.cuda.device_count(), "GPUs!")
    network_model = nn.DataParallel(network_model)

top1acc_func  = partial(topkacc, k=1, mean=False)
top5acc_func  = partial(topkacc, k=5, mean=False)
top12acc_func = partial(topkacc, k=12, mean=False)

top1acc  = Accuracy(top1acc_func, name='top1acc')
top5acc  = Accuracy(top5acc_func, name='top5acc')
top12acc = Accuracy(top12acc_func, name='top12acc')
top1accS  = Accuracy(top1acc_func, name='top1acc*', singletons=singleton_idx)
top5accS  = Accuracy(top5acc_func, name='top5acc*', singletons=singleton_idx)
top12accS = Accuracy(top12acc_func, name='top12acc*', singletons=singleton_idx)

learn = Learner(data, network_model,
                   metrics=[top1acc, top5acc, top12acc, top1accS, top5accS, top12accS],
                   loss_func=MultiCE,
                   callback_fns=[RingLoss])

learn.clip_grad()
learn.split([learn.model.module.cnn, learn.model.module.head])

# num_epochs_ = num_epochs
max_lr_epochs_list = [
    (1e-4,  25),
    (1e-4,  50),
    (2e-4,  50),
    (5e-4,  1000),
    (1e-3,  100),
]

for round_num, (max_lr_, num_epochs_) in enumerate(max_lr_epochs_list):
    print ("Round %d training (freeze)" % (round_num + 1, ))
    name_ = '%s-R%s-freeze' % (name, round_num, )
    try:
        learn.load(name_)
    except:
        learn.freeze()

        # # Find lr
        # learn.lr_find()
        # break
        # values = sorted(zip(learn.recorder.losses, learn.recorder.lrs))
        # max_lr = values[0][1]
        # max_lr_ = max_lr / 10.0
        # max_lr_ = max(max_lr_, min_max_lr)
        # print('Found max_lr = %0.08f, using %0.08f' % (max_lr, max_lr_))
        # # Train
        learn.fit_one_cycle(num_epochs_, max_lr_)
        learn.save(name_)

    print ("Round %d training (unfreeze)" % (round_num + 1, ))
    name_ = '%s-R%s-unfreeze' % (name, round_num, )
    try:
        learn.load(name_)
    except:
        learn.unfreeze()

        # # Find lr
        # learn.lr_find()
        # values = sorted(zip(learn.recorder.losses, learn.recorder.lrs))
        # max_lr = values[0][1]
        # max_lr_ = max_lr / 10.0
        # max_lr_ = max(max_lr_, min_max_lr)
        # print('Found max_lr = %0.08f, using %0.08f' % (max_lr, max_lr_))
        # # Train
        learn.fit_one_cycle(num_epochs_, max_lr_)
        learn.save(name_)

    # num_epochs_ *= 2


####
# temperature scaling
val_preds, val_gt, val_feats, val_preds2 = get_predictions(learn.model, data.valid_dl)
val_preds = val_preds.to(get_device())
print ("Finding softmax coef")
targs = torch.tensor([classes.index(label.obj) if label else num_classes for label in learn.data.valid_ds.y])
coefs = list(np.arange(0.01, 3.1, 0.05))
best_preds, best_acc, best_sc = find_softmax_coef(val_preds, targs, coefs)
min_best_sc = best_sc / 2.0
max_best_sc = best_sc * 2.0
step_best_sc = min_best_sc / 10.0
coefs = list(np.arange(min_best_sc, max_best_sc, step_best_sc))
best_preds, best_acc, best_sc = find_softmax_coef(val_preds, targs, coefs)
best_preds = best_preds.to(get_device())

####### Now features
print ("Extracting train feats")
train_feats, train_labels = get_train_features(learn, augment=0)
distance_matrix_imgs = batched_dmv(val_feats, train_feats)
distance_matrix_classes = dm2cm(distance_matrix_imgs, train_labels)
class_sims = 0.5 * (2.0 - distance_matrix_classes)
class_sims = class_sims.to(get_device())
out_preds, best_p, best_score = find_mixing_proportions(best_preds, class_sims, targs)

out_preds = out_preds.to(get_device())
targs = targs.to(get_device())
print ("Raw Val top1 acc   = ", accuracy(val_preds, targs).cpu().item())
print ("Raw Val top5 acc   = ", topkacc(val_preds, targs, k=5).cpu().item())
print ("Raw Val top12 acc  = ", topkacc(val_preds, targs, k=12).cpu().item())
print ("SM  Val top1 acc   = ", accuracy(best_preds, targs).cpu().item())
print ("SM  Val top5 acc   = ", topkacc(best_preds, targs, k=5).cpu().item())
print ("SM  Val top12 acc  = ", topkacc(best_preds, targs, k=12).cpu().item())
print ("Cls Val top1 acc   = ", accuracy(class_sims, targs).cpu().item())
print ("Cls Val top5 acc   = ", topkacc(class_sims, targs, k=5).cpu().item())
print ("Cls Val top12 acc  = ", topkacc(class_sims, targs, k=12).cpu().item())
print ("Mix Val top1 acc   = ", accuracy(out_preds, targs).cpu().item())
print ("Mix Val top5 acc   = ", topkacc(out_preds, targs, k=5).cpu().item())
print ("Mix Val top12 acc  = ", topkacc(out_preds, targs, k=12).cpu().item())
thresholds = {}
thresholds['softmax'] = best_sc
thresholds['features'] = best_p
torch.save(thresholds, 'data/models/' + name + '_thresholds.pth')

print ("Saving train feats")
torch.save({"train_labels": train_labels.detach().cpu(),
            "train_feats": train_feats.detach().cpu(),
            "val_labels": targs,
            "val_feats": val_feats.detach().cpu(),
            "classes": classes,
            "thresholds": thresholds,
            }, 'data/models/' + name + '_train_val_feats.pt')


###############
#Test
test_preds, test_gt, test_feats, test_preds2 = get_predictions(learn.model, data.test_dl)
preds_t = torch.softmax(best_sm_th * test_preds, dim=1)
preds_t = torch.cat((preds_t, torch.ones_like(preds_t[:, :1])), 1)
preds_t[:, num_classes] = best_th
#Concat with val
all_gt0 = torch.cat([val_gt, train_labels], dim=0)
all_feats0 = torch.cat([val_feats, train_feats], dim=0)
dm3 = batched_dmv(test_feats, all_feats0)
cm3 = dm2cm(dm3, all_gt0)
cm3 = 0.5*(2.0 - cm3)
preds_ft_0t = cm3.clone().detach()
preds_ft_0t[:, num_classes] = best_th_feats
pit1 = thlist[0]*cm3 + thlist[1]*preds_ft_0t+thlist[2]*preds_t
if SAVE_TEST_MATRIX:
    print ("Saving test feats")
    torch.save({"test_feats": test_feats.detach().cpu(),
                "best_preds": pit1.detach().cpu(),
                "classes": classes,
                "thresholds": thresholds,
                }, 'data/models/' + name + '_test_feats.pt')
