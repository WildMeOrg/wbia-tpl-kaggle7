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


## To use Flukebook CRC(5) data

```
wget https://cthulhu.dyn.wildme.io/public/datasets/flukebook.id.fluke.all.tar.gz
rm -rf flukebook.id.fluke.all/
targzx flukebook.id.fluke.all.tar.gz
mv data/ data_OLD/
rm -rf data/
mkdir -p data/
mkdir -p data/train/
mkdir -p data/test/
mkdir -p data/models/
cp -R flukebook.id.fluke.all/train/manifest/*.jpg data/train/
cp -R flukebook.id.fluke.all/valid/manifest/*.jpg data/train/
cp -R flukebook.id.fluke.all/valid/manifest/*.jpg data/test/
cp -R flukebook.id.fluke.all/train.txt data/train.txt
cp -R flukebook.id.fluke.all/valid.txt data/valid.txt
cp -R flukebook.id.fluke.all/valid.txt data/test.txt

python add_bboxes_and_val_fns_and_sample_submission.py
python apply_bboxes.py
cp pretrained.pth data/models/
```




mv data/ data_OLD/
rm -rf data/

mkdir -p data/
mkdir -p data/train/
mkdir -p data/test/

cp -R flukebook.id.fluke.crc.all/train/manifest/*.jpg data/train/
cp -R flukebook.id.fluke.crc.all/valid/manifest/*.jpg data/train/
cp -R flukebook.id.fluke.crc.all/train.txt data/train.crc.txt
cp -R flukebook.id.fluke.crc.all/valid.txt data/valid.crc.txt

cp -R data_kaggle2/wbia/export-encounters/train/manifest/*.jpg data/train/
cp -R data_kaggle2/wbia/export-encounters/valid/manifest/*.jpg data/train/
cp -R data_kaggle2/wbia/export-encounters/train.txt data/train.kaggle.txt
cp -R data_kaggle2/wbia/export-encounters/valid.txt data/valid.kaggle.txt

cp -R data_kaggle2/test/*.jpg data/test/
cp -R data_kaggle2/test.csv data/test.csv

sed '1d' data/train.kaggle.txt > data/train.kaggle.stripped.txt
sed '1d' data/valid.kaggle.txt > data/valid.kaggle.stripped.txt

cat data/train.crc.txt data/train.kaggle.stripped.txt > data/train.txt
cat data/valid.crc.txt data/valid.kaggle.stripped.txt > data/valid.txt

python add_bboxes_and_val_fns_and_sample_submission.py
