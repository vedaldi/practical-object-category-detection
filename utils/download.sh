#!/bin/bash

mkdir -p data/tmp

(
    cd data/tmp
    wget -c nc http://benchmark.ini.rub.de/Dataset/GTSRB_Final_Training_Images.zip
    wget -c nc http://benchmark.ini.rub.de/Dataset/GTSRB_Final_Test_Images.zip
    wget -c nc http://benchmark.ini.rub.de/Dataset_GTSDB/TrainIJCNN2013.zip
    wget -c nc http://benchmark.ini.rub.de/Dataset_GTSDB/TestIJCNN2013.zip
    unzip GTSRB_Final_Training_Images.zip
    unzip GTSRB_Final_Test_Images.zip
    unzip TrainIJCNN2013.zip
    unzip TestIJCNN2013.zip
)
