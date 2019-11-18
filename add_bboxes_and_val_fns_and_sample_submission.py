import pandas as pd
from PIL import Image
import utool as ut
import random
import os
import pickle

train_files = ut.glob('data/train/*.jpg')
test_files = ut.glob('data/test/*.jpg')

train_files = list(zip(['train'] * len(train_files), train_files))
test_files = list(zip(['test'] * len(test_files), test_files))
image_files = train_files + test_files

submission = ['new_whale', 'w_23a388d', 'w_9b5109b', 'w_9c506f6', 'w_0369a5c']
submission_str = ' '.join(submission)

seen = set([])
train_data = []
test_data = []
valid_set = set([])
for type_, image_file in image_files:
    image_filename = os.path.basename(image_file)
    assert image_filename not in seen
    seen.add(image_filename)

    img = Image.open(image_file)
    img_w, img_h = img.size

    train_point = list(map(str, [image_filename, 0, 0, img_w, img_h]))
    test_point = list(map(str, [image_filename, submission_str]))

    train_data.append(train_point)

    if type_ == 'test':
        test_data.append(test_point)
    if type_ == 'train':
        if random.uniform(0.0, 1.0) <= 0.1:
            valid_set.add(image_filename)

columns = ['Image', 'x0', 'y0', 'x1', 'y1']
df = pd.DataFrame(train_data, columns=columns)
with open('data/bounding_boxes.csv', 'w') as csv_file:
    csv_str = df.to_csv(index=False)
    csv_file.write(csv_str)

with open('data/val_fns', 'wb') as pickle_file:
    pickle_str = pickle.dumps(valid_set)
    pickle_file.write(pickle_str)

columns = ['Image', 'Id']
df = pd.DataFrame(test_data, columns=columns)
with open('data/sample_submission.csv', 'w') as csv_file:
    csv_str = df.to_csv(index=False)
    csv_file.write(csv_str)
