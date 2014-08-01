#!/bin/bash

mkdir -p data/tmp

(
    cd data/tmp
    wget -c nc http://benchmark.ini.rub.de/Dataset_GTSDB/TrainIJCNN2013.zip
    unzip -n TrainIJCNN2013.zip
)
