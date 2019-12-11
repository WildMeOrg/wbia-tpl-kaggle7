# docker build --target org.wildme.ibeis.pytorch --tag wildme/pytorch:1.3-cuda10.1-cudnn7-magma100-devel-ubuntu18.04 .
# docker build --target org.wildme.ibeis.kaggle7.train --tag wildme/kaggle7:latest .
docker build --target org.wildme.ibeis.kaggle7.server --tag wildme.azurecr.io/ibeis/kaggle7:latest .
