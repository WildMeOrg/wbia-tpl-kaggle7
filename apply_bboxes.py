from PIL import Image
import pandas as pd
import tqdm
import os


bbox_df = pd.read_csv('data/bounding_boxes.csv')


for version in ['train', 'test']:
    bbox_df.head()

    crop_path = os.path.join('data', 'crop_%s' % (version, ))

    if not os.path.exists(crop_path):
        os.mkdir(crop_path)

    df = pd.read_csv('data/%s.csv' % (version, ))
    df.head()
    df_ = df.merge(bbox_df, on=['Image'])

    index_list = list(range(len(df_)))
    for index in tqdm.tqdm(index_list):
        image_filename = df_.Image[index]
        image_filepath = os.path.join('data', version, image_filename)
        image = Image.open(image_filepath)
        area = (
            df_.x0[index],
            df_.y0[index],
            df_.x1[index],
            df_.y1[index],
        )
        cropped_image = image.crop(area)
        crop_filepath = os.path.join(crop_path, image_filename)
        cropped_image.save(crop_filepath)
