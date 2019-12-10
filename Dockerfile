##########################################################################################
# STAGE 1 - Image for Wild Me's version of PyTorch
##########################################################################################
FROM nvidia/cuda:10.1-cudnn7-devel-ubuntu18.04 as org.wildme.ibeis.pytorch

MAINTAINER Wild Me <dev@wildme.org> 

# Selectively disable Docker build caching in Azure DevOps Nightly CI builds
ARG AZURE_DEVOPS_CACHEBUSTER=0

RUN echo "ARGS AZURE_DEVOPS_CACHEBUSTER=${AZURE_DEVOPS_CACHEBUSTER}"

# Setup CUDA lib paths for local builds
ENV PATH "/usr/local/cuda/bin:${PATH}"

ENV LD_LIBRARY_PATH "/usr/local/cuda/lib64:${LD_LIBRARY_PATH}"

ENV CUDA_HOME "/usr/local/cuda"

# Add required and convenient apt-get packages
RUN apt-get update \
 && apt-get install -y --no-install-recommends \
        build-essential \
        ca-certificates \
        cmake \
        curl \
        git \
        htop \
        libjpeg-dev \
        libpng-dev \
        locate \
        tmux \
        unzip \
        vim \
 && rm -rf /var/lib/apt/lists/*

# Install Miniconda
RUN curl -o /opt/miniconda.sh -O https://repo.continuum.io/miniconda/Miniconda3-latest-Linux-x86_64.sh \
 && chmod +x /opt/miniconda.sh \
 && /opt/miniconda.sh -b -p /opt/conda \
 && rm -rf /opt/miniconda.sh

# Install dependencies for PyTorch
RUN /opt/conda/bin/conda install -y python=3.7 numpy pyyaml scipy ipython mkl mkl-include ninja cython typing \
 && /opt/conda/bin/conda install -y -c pytorch magma-cuda100 \
 && /opt/conda/bin/conda clean -ya

# Enable Conda as the defauly Python environment
ENV PATH /opt/conda/bin:$PATH

# Clone, build, and install PyTorch against local CUDA+MAGMA libraries (drop support for CUDA arch 3.5 and add 7.5*)
RUN git clone https://github.com/pytorch/pytorch.git /opt/pytorch/ \
 && cd /opt/pytorch/ \
 && git checkout v1.3.0 \
 && git submodule update --init --recursive \
 && TORCH_CUDA_ARCH_LIST="5.2 6.0 6.1 7.0+PTX 7.5+PTX" TORCH_NVCC_FLAGS="-Xfatbin -compress-all" CMAKE_PREFIX_PATH="/opt/conda/" pip install -v . \
 && rm -rf /opt/pytorch/

# Clone, build, and install PyTorch/Vision
RUN git clone https://github.com/pytorch/vision.git /opt/vision/ \
 && cd /opt/vision/ \
 && git checkout v0.4.1 \
 && pip install -v . \
 && rm -rf /opt/vision/

# Set provisional work directory
WORKDIR /workspace/

# Update permissions in the file-system to be fully writable by all
RUN chmod -R a+w .

##########################################################################################
# STAGE 2 - Image for whale-identification-2018
##########################################################################################
FROM org.wildme.ibeis.pytorch as org.wildme.ibeis.kaggle7.train

# Install additional conda dependencies
RUN conda install -y jupyter notebook \
 && conda install -c conda-forge jupyter_contrib_nbextensions \
 && conda clean -ya

# Install additional PyPI dependencies
RUN pip install fastai pretrainedmodels

# Pre-download pre-trained VGG-16 model with Batch Norm
RUN python -c 'import torchvision; torchvision.models.densenet201(pretrained=True)'

# Copy local Python code from repo into container
COPY ./*.py /opt/whale/

# Destination for data
RUN mkdir -p /data

# Add symlink to /data
RUN ln -s /data /opt/whale/data 

# Set workdir to the main repository for convenience
WORKDIR /opt/whale/

# Update permissions in the file-system to be fully writable by all
RUN chmod -R a+w .

# Start training, assuming training data is mapped into /data
ENTRYPOINT ["python", "train_VGG16.py"]

# Optional commands to the training script are supported as CLI arguments
CMD []

# Send proper stop signal on container termination
STOPSIGNAL SIGTERM

##########################################################################################
# STAGE 2 - Image for whale-identification-2018
##########################################################################################
FROM org.wildme.ibeis.pytorch as org.wildme.ibeis.kaggle7.server

# Install additional conda dependencies
RUN conda install -y jupyter notebook \
 && conda install -c conda-forge jupyter_contrib_nbextensions \
 && conda clean -ya

# Install additional PyPI dependencies
RUN pip install fastai pretrainedmodels flask_restful utool

# Pre-download pre-trained VGG-16 model with Batch Norm
RUN python -c 'import torchvision; torchvision.models.densenet201(pretrained=True)'

# Copy local Python code from repo into container
COPY ./*.py /opt/whale/

# Destination for data
RUN mkdir -p /data

# Add symlink to /data
RUN ln -s /data /opt/whale/data 

# Set workdir to the main repository for convenience
WORKDIR /opt/whale/

# Update permissions in the file-system to be fully writable by all
RUN chmod -R a+w .

# Start training, assuming training data is mapped into /data
ENTRYPOINT ["python", "server.py"]

# Optional commands to the training script are supported as CLI arguments
CMD []

# Send proper stop signal on container termination
STOPSIGNAL SIGTERM
