from flask import Flask
from flask_restful import Api, Resource, reqparse
from arch import make_new_network
from io import BytesIO
from PIL import Image
import utool as ut
import base64
import torch


APP = Flask(__name__)
API = Api(APP)

NETWORK = None

model_url_dict = {
    'crc': 'https://cthulhu.dyn.wildme.io/public/models/kaggle7.crc.final.pth',
}


def get_image_from_base64_str(image_base64_str):
    image = Image.open(BytesIO(base64.b64decode(image_base64_str)))
    return image


class Kaggle7(Resource):
    def post(self):
        response = {'success': False}

        ut.embed()

        try:
            parser = reqparse.RequestParser()
            parser.add_argument('image', type=str)
            parser.add_argument('config', type=dict)
            args = parser.parse_args()

            image_base64_str = args['image']
            image = get_image_from_base64_str(image_base64_str)

            config = args['config']
            model_tag = config['model_tag']
            model_url = model_url_dict.get(model_tag, None)

            assert model_url is not None, 'Model tag %r is not recognized' % (model_tag, )

            values_url = model_url.replace('.pth', '.values.pth')

            # Download files
            model_filepath = ut.grab_file_url(model_url, appname='kaggle7', check_hash=True)
            values_filepath = ut.grab_file_url(values_url, appname='kaggle7', check_hash=True)

            model_weights = torch.load(model_filepath)
            model_values = torch.load(values_filepath)

            num_classes = len(set(df.Id))
            network_model = make_new_network(num_classes, RING_HEADS, GEM_CONST)

            initialized = network_model.module.state_dict()

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

        except Exception as ex:
            message = str(ex)
            response['message'] = message

        return response


API.add_resource(Kaggle7, '/api/classify')


if __name__ == '__main__':
    APP.run(host='0.0.0.0', port=5000)
