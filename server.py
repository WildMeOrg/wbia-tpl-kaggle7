from flask import Flask
from flask_restful import Api, Resource
import utool as ut


APP = Flask(__name__)
API = Api(APP)


class Kaggle7(Resource):
    def post(self):
        ut.embed()
        return False


API.add_resource(Kaggle7, '/api/classify')


if __name__ == '__main__':
    APP.run(debug=True)
