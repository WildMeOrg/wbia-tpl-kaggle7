# docker build --target org.wildme.wbia.pytorch --tag wildme/pytorch:1.3-cuda10.1-cudnn7-magma100-devel-ubuntu18.04 .
# docker build --target org.wildme.wbia.kaggle7.train --tag wildme/kaggle7:latest .
docker build --target org.wildme.wbia.kaggle7.server --tag wildme.azurecr.io/wbia/kaggle7:latest .
