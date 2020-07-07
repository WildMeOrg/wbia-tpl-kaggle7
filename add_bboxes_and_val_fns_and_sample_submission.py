# -*- coding: utf-8 -*-
import pandas as pd
from PIL import Image
import os
import pickle
import utool as ut

train_df = pd.read_csv('data/train.txt')
train_df = train_df.drop(columns=['Unixtime'])
train_df = train_df.rename(columns={'ID': 'Id'})

valid_df = pd.read_csv('data/valid.txt')
valid_df = valid_df.drop(columns=['Unixtime'])
valid_df = valid_df.rename(columns={'ID': 'Id'})

test_df = pd.read_csv('data/test.txt')
test_df = test_df.drop(columns=['Unixtime'])
test_df = test_df.rename(columns={'ID': 'Id'})

train_df_image_set = set(train_df.Image)
valid_df_image_set = set(valid_df.Image)
test_df_image_set  = set(test_df.Image)

train_df_id_set = set(train_df.Id)
valid_df_id_set = set(valid_df.Id)
test_df_id_set  = set(valid_df.Id)

assert len(train_df_image_set & valid_df_image_set) == 0
# assert len(train_df_image_set & test_df_image_set)  == 0
# assert len(valid_df_image_set & test_df_image_set)  == 0

assert train_df_id_set | valid_df_id_set == train_df_id_set
# assert train_df_id_set | test_df_id_set  == valid_df_id_set
# assert valid_df_id_set | test_df_id_set  == valid_df_id_set

df = pd.concat([train_df, valid_df], ignore_index=True)

names = {}
for i in range(len(df)):
    filename = df.Image[i]
    name = df.Id[i]
    if name not in names:
        names[name] = []
    names[name].append(filename)

freq_list = {}
for name in names:
    freq = len(names[name])
    if freq not in freq_list:
        freq_list[freq] = 0
    freq_list[freq] += 1
# assert 1 not in freq_list
print(ut.repr3(freq_list))

with open('data/train.csv', 'w') as csv_file:
    csv_str = df.to_csv(index=False)
    csv_file.write(csv_str)

with open('data/test.csv', 'w') as csv_file:
    csv_str = test_df.to_csv(index=False)
    csv_file.write(csv_str)

bbox_data = []
version_list = [
    ('train', df.Image),
    # ('test', test_df.Image)
]
for version, image_filenames in version_list:
    for image_filename in image_filenames:
        image_filepath = os.path.join('data', version, image_filename)

        img = Image.open(image_filepath)
        img_w, img_h = img.size

        bbox_row = list(map(str, [image_filename, 0, 0, img_w, img_h]))
        bbox_data.append(bbox_row)

columns = ['Image', 'x0', 'y0', 'x1', 'y1']
df = pd.DataFrame(bbox_data, columns=columns)
with open('data/bounding_boxes.csv', 'w') as csv_file:
    csv_str = df.to_csv(index=False)
    csv_file.write(csv_str)

with open('data/val_fns', 'wb') as pickle_file:
    pickle_str = pickle.dumps(valid_df_image_set)
    pickle_file.write(pickle_str)
