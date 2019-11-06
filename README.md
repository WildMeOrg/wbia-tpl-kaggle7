# Humpback Whale Identification - Kaggle Winning Solution #7

Fork of https://github.com/ducha-aiki/whale-identification-2018

Heavily based on https://github.com/radekosmulski/whale

## To build (and publish) Docker image

```
./build.sh
# ./publish.sh  # Publish to Dockerhub (requires authentication)
```

## Download the training data

1. Clone this repository. cd into data. Download competition data by running ```kaggle competitions download -c humpback-whale-identification```. You might need to agree to competition rules on competition website if you get a 403.
2. Create the train directory and extract files via running ```mkdir train && unzip train.zip -d train```
3. Do the same for test: ```mkdir test && unzip test.zip -d test```
4. Go back to top-level directory ``` cd ../```
4. Extract boxes ```python apply_bboxes.py```

## To run with Docker

```
docker pull wildme/ibeis/kaggle7:latest

# Map the local ./data folder into the /data/ folder inside the container (which is symlinked from /opt/whale/data/)
NV_GPU=1,3 nvidia-docker container run -d --name kaggle7 -v $(pwd)/data/:/data/ wildme/ibeis/kaggle7:latest
```