from flask import Flask
from flask_restful import Api, Resource, reqparse
from io import BytesIO
from PIL import Image
import utool as ut
import numpy as np
import base64
import torch
from torchvision import transforms
from fastai.vision import pil2tensor, imagenet_stats
from utils import batched_dmv, dm2cm
from arch import make_new_network, L2Norm, get_device
from train_VGG16 import RING_HEADS, GEM_CONST, SZH, SZW


APP = Flask(__name__)
API = Api(APP)

NETWORK_MODEL_TAG = None
NETWORK           = None
NETWORK_VALUES    = None

model_url_dict = {
    'crc': 'https://wildbookiarepository.blob.core.windows.net/models/kaggle7.crc.final.1.pth',
}

CMODE = 'RGB'
SIZE = (SZH, SZW)
TFRM_RESIZE = transforms.Resize(SIZE)
TFRM_WHITEN = transforms.Normalize(*imagenet_stats)
TFRM_L2NORM = L2Norm()


def get_image_from_base64_str(image_base64_str):
    image = Image.open(BytesIO(base64.b64decode(image_base64_str)))
    return image


class Kaggle7(Resource):
    def post(self):
        global NETWORK_MODEL_TAG
        global NETWORK
        global NETWORK_VALUES

        response = {'success': False}

        # ut.embed()

        try:
            with ut.Timer('Pre'):
                parser = reqparse.RequestParser()
                parser.add_argument('image', type=str)
                parser.add_argument('config', type=dict)
                args = parser.parse_args()

                image_base64_str = args['image']
                image = get_image_from_base64_str(image_base64_str)

                config = args['config']
                model_tag = config.get('model_tag', None)
                num_returns = config.get('topk', 100)

                model_url = model_url_dict.get(model_tag, None)

            assert model_url is not None, 'Model tag %r is not recognized' % (model_tag, )
            if model_tag != NETWORK_MODEL_TAG:
                with ut.Timer('Loading network'):
                    print('Loading network from weights %r' % (model_tag, ))
                    values_url = model_url.replace('.pth', '.values.pth')

                    # Download files
                    model_filepath = ut.grab_file_url(model_url, appname='kaggle7', check_hash=True)
                    values_filepath = ut.grab_file_url(values_url, appname='kaggle7', check_hash=True)

                    model_values = torch.load(values_filepath)
                    classes = model_values['classes']
                    num_classes = len(classes)

                    model_weights = torch.load(model_filepath, map_location=get_device())
                    network_model, mutliple = make_new_network(num_classes, RING_HEADS, GEM_CONST, pretrained=False)

                    if mutliple:
                        pass

                    if torch.cuda.is_available():
                        network_model = network_model.cuda()

                    # model_weights = model_weights['model']
                    network_model.load_state_dict(model_weights)
                    network_model.eval()

                    NETWORK_MODEL_TAG = model_tag
                    NETWORK           = network_model
                    NETWORK_VALUES    = model_values

            print('Using network %r' % (NETWORK_MODEL_TAG, ))
            with ut.Timer('Loading input tensor'):
                input_image = image.convert(CMODE).convert('LA').convert(CMODE)
                input_image = TFRM_RESIZE(input_image)
                input_image = pil2tensor(input_image, np.float32)
                input_image = input_image.div_(255)
                input_image = TFRM_WHITEN(input_image)

                size = input_image.size()
                input_tensor = input_image.view(-1, size[0], size[1], size[2])
                input_tensor = input_tensor.to(get_device())

            # Run inference
            with ut.Timer('Inference'):
                print('Running inference on input tensor %r' % (input_tensor.size(), ))
                output = NETWORK(input_tensor)
                print('...done')
                preds_list, feats_list = output

            with ut.Timer('Post1'):
                print('Performing post-processing')
                prediction_raw = preds_list[-1][0]
                features_raw = TFRM_L2NORM(torch.cat(feats_list, dim=1))[0]

            with ut.Timer('Post2'):
                print('...classifier')
                # Post Process classification
                classifier_temp = NETWORK_VALUES['thresholds']['classifier_softmax_temp']
                classifier_prediction = torch.softmax(prediction_raw / classifier_temp, dim=0)

            with ut.Timer('Post3'):
                # Post process features
                print('...features')
                train_feats = NETWORK_VALUES['train_feats']
                train_gt = NETWORK_VALUES['train_gt']
                size = features_raw.size()
                features = features_raw.view(-1, size[0])
                distance_matrix_imgs = batched_dmv(features, train_feats)
                distance_matrix_classes = dm2cm(distance_matrix_imgs, train_gt)
                features_sim = (2.0 - distance_matrix_classes) * 0.5
                features_sim = features_sim[0]

                features_temp = NETWORK_VALUES['thresholds']['feature_softmax_temp']
                features_prediction = torch.softmax(features_sim / features_temp, dim=0)

            with ut.Timer('Post4'):
                print('...mixing')
                p = NETWORK_VALUES['thresholds']['mixing_value']
                classifier_prediction = classifier_prediction.to('cpu')
                final_prediction = p * classifier_prediction + (1.0 - p) * features_prediction

            with ut.Timer('Collection'):
                print('Collecting prediction')
                top_k_score_list, top_k_index_list = final_prediction.topk(num_returns, 0)
                top_k_score_list = top_k_score_list.detach().tolist()
                classes = NETWORK_VALUES['classes']
                top_k_class_list = ut.take(classes, top_k_index_list)

                response['scores'] = {}
                for top_k_class, top_k_score in zip(top_k_class_list, top_k_score_list):
                    response['scores'][top_k_class] = top_k_score
                response['success'] = True

            print('...done')
        except Exception as ex:
            message = str(ex)
            response['message'] = message
            print('!!!ERROR!!!')
            print(response)

        # if torch.cuda.is_available():
        #     torch.cuda.empty_cache()

        return response


API.add_resource(Kaggle7, '/api/classify')


if __name__ == '__main__':
    APP.run(host='0.0.0.0', port=5000)
