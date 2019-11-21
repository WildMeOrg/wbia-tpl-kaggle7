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

print("torch.cuda.is_available:", torch.cuda.is_available())

df = pd.read_csv('data/train.csv')
val_fns = pd.read_pickle('data/val_fns')

fn2label = {row[1].Image: row[1].Id for row in df.iterrows()}
path2fn = lambda path: re.search('[\w-]*\.jpg$', path).group(0)

SZ = 660
BS = 16
NUM_WORKERS = 10
SEED = 0
SAVE_TRAIN_FEATS = True
SAVE_TEST_MATRIX = True
LOAD_IF_CAN = True

num_classes = len(set(df.Id))  # 1571
num_epochs = 50

name = f'DenseNet201-GeM-PCB8-{SZ}-Ring-RELU'

tfms = (
    [
        RandTransform(tfm=perspective_warp, kwargs={'magnitude': (-0.2, 0.2)}),
        RandTransform(tfm=rotate, kwargs={'degrees': (-15.0, 15.0)}),
        RandTransform(tfm=zoom, kwargs={'scale': (0.9, 1.1), 'row_pct': (0, 1), 'col_pct': (0, 1)}),
        RandTransform(tfm=brightness, kwargs={'change': (0.3, 0.7)}),
        RandTransform(tfm=contrast, kwargs={'scale': (0.5, 1.5)}),
    ],
    []
)

if not os.path.exists('data/augmentations'):
    os.mkdir('data/augmentations')

for index in range(len(df.Image)):
    if index > 10:
        break
    filename = df.Image[index]
    basename, ext = os.path.splitext(filename)
    path = os.path.join('data/crop_train', filename)
    # image = open_image_grey(path)
    image = open_image(path)
    print(path)
    print(image)
    image_ = image.apply_tfms(tfms[0], size=SZ, resize_method=ResizeMethod.SQUISH, padding_mode='reflection')
    image.save('data/augmentations/%s_original%s' % (basename, ext, ))
    image_.save('data/augmentations/%s_augmented%s' % (basename, ext, ))

data = (
    ImageListGray
    .from_df(df[df.Id != 'new_whale'], 'data/crop_train', cols=['Image'])
    .split_by_valid_func(lambda path: path2fn(path) in val_fns)
    .label_from_func(lambda path: fn2label[path2fn(path)])
    .add_test(ImageList.from_folder('data/crop_test'))
    .transform(tfms, size=SZ, resize_method=ResizeMethod.SQUISH, padding_mode='reflection')
    .databunch(bs=BS, num_workers=NUM_WORKERS, path='data')
    .normalize(imagenet_stats)
)


class CustomPCBNetwork(nn.Module):
    def __init__(self, new_model):
        super().__init__()
        self.cnn =  new_model.features
        self.head = PCBRingHead2(num_classes, 320, 4, 1920)

    def forward(self, x):
        x = self.cnn(x)
        out = self.head(x)
        return out

network_model = CustomPCBNetwork(torchvision.models.densenet201(pretrained=True))

if torch.cuda.device_count() > 1:
    print("Using", torch.cuda.device_count(), "GPUs!")
    network_model = nn.DataParallel(network_model)

learn = Learner(data, network_model,
                   metrics=[map1total, map5total, map12total],
                   loss_func=MultiCE,
                   callback_fns=[RingLoss])

# out = network_model.module.cnn(torch.rand(1,3,SZ,SZ).to(get_device()))
# print(out.shape)

learn.split([learn.model.module.cnn, learn.model.module.head])
learn.freeze()
learn.clip_grad()

LOADED = False
print ("Stage one, training only head")
if LOAD_IF_CAN:
    try:
        learn.load(name)
        LOADED = True
    except:
        LOADED = False
if not LOADED:
    num_epochs_ = num_epochs
    for round_num in range(3):
        print ("Stage one, round %d training" % (round_num + 1, ))

        # Find max_lr
        learn.lr_find()
        values = sorted(zip(learn.recorder.losses, learn.recorder.lrs))
        max_lr = values[0][1]
        max_lr /= 10.0

        # Train
        learn.fit_one_cycle(num_epochs_, max_lr)
        num_epochs_ *= 2
    learn.save(name)
print ('Stage 1 done, finetuning everything')

learn.unfreeze()

LOADED = False
if LOAD_IF_CAN:
    try:
        learn.load(name + '_unfreeze')
        LOADED = True
    except:
        LOADED = False

if not LOADED:
    num_epochs_ = num_epochs
    for round_num in range(3):
        print ("Stage two, round %d training" % (round_num + 1, ))

        # Find max_lr
        learn.lr_find()
        values = sorted(zip(learn.recorder.losses, learn.recorder.lrs))
        max_lr = values[0][1]
        max_lr /= 10.0

        # Train
        learn.fit_one_cycle(num_epochs_, max_lr)
        num_epochs_ *= 2
    learn.save(name + '_unfreeze')

print ("Stage 2 done, starting stage 3")

LOADED = False
print ("Stage 2 done, stage 3 done")


####### Validation
print ("Starting validation")
df = pd.read_csv('data/train.csv')
val_fns = pd.read_pickle('data/val_fns')
new_whale_fns = set(df[df.Id == 'new_whale'].sample(frac=1).Image.iloc[:1000])
y = val_fns.union(new_whale_fns)
classes = learn.data.classes + ['new_whale']
data = (
    ImageListGray
        .from_df(df, 'data/crop_train', cols=['Image'])
        .split_by_valid_func(lambda path: path2fn(path) in y)
        .label_from_func(lambda path: fn2label[path2fn(path)], classes=classes)
        .add_test(ImageList.from_folder('data/crop_test'))
        .transform(get_transforms(do_flip=False, max_zoom=1,
                                  max_warp=0,
                                  max_rotate=2), size=SZ, resize_method=ResizeMethod.SQUISH)
        .databunch(bs=BS, num_workers=NUM_WORKERS, path='data')
        .normalize(imagenet_stats)
)
data.train_dl.dl.batch_sampler.sampler = torch.utils.data.SequentialSampler(data.train_ds)
data.train_dl.dl.batch_sampler.drop_last = False
data.valid_dl.dl.batch_sampler.sampler = torch.utils.data.SequentialSampler(data.valid_ds)
data.valid_dl.dl.batch_sampler.drop_last = False

learn.data = data
targs = torch.tensor([classes.index(label.obj) if label else num_classes for label in learn.data.valid_ds.y])

####
val_preds, val_gt,val_feats,val_preds2 = get_predictions(learn.model,data.valid_dl)
print ("Finding softmax coef")
best_preds, best_th, best_sm_th, best_score = find_softmax_coef(val_preds,targs, [0.1, 0.2, 0.3, 0.5, 1.0, 2.0, 3.0])

####### Now features
print ("Extracting train feats")
train_feats, train_labels = get_train_features(learn, augment=0)
distance_matrix_imgs = batched_dmv(val_feats, train_feats)
distance_matrix_classes = dm2cm(distance_matrix_imgs, train_labels)
class_sims = 0.5*(2.0 - distance_matrix_classes)
class_sims_th, best_th_feats, score_feats_th = find_new_whale_th(class_sims, targs)
out_preds, thlist, best_score = find_mixing_proportions(best_preds,
                                                       class_sims,
                                                      class_sims_th,targs)
out_preds = out_preds.to(get_device())
targs = targs.to(get_device())
print ("Best mix score = ", best_score)
print ("Val top1 acc = ", accuracy(out_preds, targs).cpu().item())
print ("Val map5 = ",map5(out_preds, targs).cpu().item())
print ("Val top5 acc = ",topkacc(out_preds, targs, k=5).cpu().item())
thresholds = {}
thresholds['softmax'] = best_sm_th
thresholds['preds_th'] = best_th
thresholds['preds_th_feats'] = best_th_feats
thresholds['mix_list'] = thlist
torch.save(thresholds, 'data/models/' + name + '_thresholds.pth')

if SAVE_TRAIN_FEATS:
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
test_preds,  test_gt,test_feats,test_preds2 = get_predictions(learn.model,data.test_dl)
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

try:
    os.makedirs('subs')
except:
    pass
create_submission(pit1.cpu(), learn.data, name, classes)
print ('new_whales at 1st pos:', pd.read_csv(f'subs/{name}.csv.gz').Id.str.split().apply(lambda x: x[0] == 'new_whale').mean())
