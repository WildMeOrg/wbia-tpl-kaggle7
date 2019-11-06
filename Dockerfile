FROM pytorch/pytorch

RUN pip install fastai
RUN pip install pretrainedmodels
RUN python -c 'import torchvision; torchvision.models.vgg16_bn(pretrained=True)'
RUN pip install jupyter
RUN apt-get update
RUN apt-get install -y vim

COPY *.py /opt/whale/
WORKDIR /opt/whale

# Start training, assuming training data is in data/.
CMD python train_VGG16.py
