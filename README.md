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
docker pull wildme/kaggle7:latest

# Map the local ./data folder into the /data/ folder inside the container (which is symlinked from /opt/whale/data/)
NV_GPU=1,3 nvidia-docker container run -it --rm --name kaggle7 -v $(pwd)/data/:/data/ --ipc=host wildme/kaggle7:latest
NV_GPU=1,3 nvidia-docker container run -it --rm --name kaggle7 -v $(pwd)/data/:/data/ --ipc=host --entrypoint="/bin/bash" wildme/kaggle7:latest
```


## TO use Flukebook data

```
wget https://cthulhu.dyn.wildme.io/public/datasets/flukebook.id.fluke.encounters.tar.gz
rm -rf flukebook.id.fluke.encounters/
targzx flukebook.id.fluke.encounters.tar.gz
mkdir -p data_flukebook
mkdir -p data_flukebook/train
mkdir -p data_flukebook/test
cp -R flukebook.id.fluke.encounters/train/manifest/ data_flukebook/train/
cp -R flukebook.id.fluke.encounters/test/manifest/ data_flukebook/test/
cp -R flukebook.id.fluke.encounters/train.txt data_flukebook/train.txt
cp -R flukebook.id.fluke.encounters/test.txt data_flukebook/test.txt

mv data/ data_old/
mv data_flukebook/ data/
python add_bboxes_and_val_fns_and_sample_submission.py
```
