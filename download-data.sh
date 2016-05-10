#!/bin/bash

DATA_DIR=data

mkdir ${DATA_DIR}
cd ${DATA_DIR}

# SMS Spam collection
curl -O https://archive.ics.uci.edu/ml/machine-learning-databases/00228/smsspamcollection.zip
unzip smsspamcollection.zip SMSSpamCollection

# Housing
curl -O https://archive.ics.uci.edu/ml/machine-learning-databases/housing/housing.data
