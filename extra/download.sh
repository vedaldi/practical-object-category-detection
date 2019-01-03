#!/bin/bash

mkdir -p data/tmp

cp -vf extra/{mandatory,cover,signs-sample-image}.jpg data/

(
    cd data/tmp
    wget -c -nc http://benchmark.ini.rub.de/Dataset_GTSDB/TrainIJCNN2013.zip
    unzip -n TrainIJCNN2013.zip
)

mkdir -p data/signs
mogrify -path data/signs -format jpeg data/tmp/TrainIJCNN2013/*.ppm 
