#!/bin/bash

# Download dataset
mkdir -p ./downloaded_datasets/musdb18hq
wget -O ./downloaded_datasets/musdb18hq/musdb18hq.zip https://zenodo.org/records/3338373/files/musdb18hq.zip?download=1

# Unzip dataset
mkdir -p ./data/musdb18hq
unzip ./downloaded_datasets/musdb18hq/musdb18hq.zip -d ./data/musdb18hq/