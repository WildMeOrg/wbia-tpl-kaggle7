import pandas as pd
from PIL import Image
import os
import pickle
import utool as ut

train_df = pd.read_csv('data/train.txt')
train_df = train_df.drop(columns=['Unixtime'])
train_df = train_df.rename(columns={'ID': 'Id'})

test_df = pd.read_csv('data/test.txt')
test_df = test_df.drop(columns=['Unixtime'])
test_df = test_df.rename(columns={'ID': 'Id'})

train_df_image_set = set(train_df.Image)
test_df_image_set = set(test_df.Image)
assert len(train_df_image_set & test_df_image_set) == 0
train_df_id_set = set(train_df.Id)
test_df_id_set = set(test_df.Id)

df = pd.concat([train_df, test_df], ignore_index=True)

names = {}
for i in range(len(df)):
    filename = df.Image[i]
    name = df.Id[i]
    assert name in train_df_id_set
    assert name in test_df_id_set
    if name not in names:
        names[name] = []
    names[name].append(filename)

with open('data/train.csv', 'w') as csv_file:
    csv_str = df.to_csv(index=False)
    csv_file.write(csv_str)

freq_list = {}
for name in names:
    freq = len(names[name])
    if freq not in freq_list:
        freq_list[freq] = 0
    freq_list[freq] += 1
assert 1 not in freq_list
print(ut.repr3(freq_list))

name_list = sorted(names.keys())
submission = name_list[:5]
submission_str = ' '.join(submission)

bbox_data = []
test_data = []
for image_filename in df.Image:
    image_filepath = os.path.join('data', 'train', image_filename)
    test_filepath = os.path.join('data', 'test', image_filename)

    img = Image.open(image_filepath)
    img_w, img_h = img.size

    bbox_row = list(map(str, [image_filename, 0, 0, img_w, img_h]))
    bbox_data.append(bbox_row)

    if os.path.exists(test_filepath):
        test_point = list(map(str, [image_filename, submission_str]))
        test_data.append(test_point)

columns = ['Image', 'x0', 'y0', 'x1', 'y1']
df = pd.DataFrame(bbox_data, columns=columns)
with open('data/bounding_boxes.csv', 'w') as csv_file:
    csv_str = df.to_csv(index=False)
    csv_file.write(csv_str)

columns = ['Image', 'Id']
df = pd.DataFrame(test_data, columns=columns)
with open('data/sample_submission.csv', 'w') as csv_file:
    csv_str = df.to_csv(index=False)
    csv_file.write(csv_str)

with open('data/val_fns', 'wb') as pickle_file:
    pickle_str = pickle.dumps(test_df_image_set)
    pickle_file.write(pickle_str)
