# Acquisition
This directory contains the jupyter notebooks responsible for the acquisition of the EEG signals. This includes the generation of music sessions and the communication with the LSL system.

## General setup
0. Don't forget to follow the general code setup steps including setting up the virtual environment and installing the requirements

1. Download the [GTZAN Dataset - Music Genre classification](https://www.kaggle.com/datasets/andradaolteanu/gtzan-dataset-music-genre-classification) on kaggle.

2. Move the GTZAN dataset to the `data` folder and rename the directory from `Data` to `gtzan`. Make sure that the `gtzan` directory contains the folder `genres_original`. The other directories inside `gtzan` can be removed. Also make sure that each genre directory contains 100 songs.

## Functionality

### [Session generation](session-generator.ipynb)
This notebook is responsible for generating the distinct sessions for the acquisition. The whole dataset is split in 20 distinct sessions each containing 50 songs. Each session is balanced according to their respective genre, so that each genre is represented by 5 songs. The distinct sessions are stored in an auto-generated folder in `data` called `music_sessions`. Each session directory contains the `.wav` files for each song.