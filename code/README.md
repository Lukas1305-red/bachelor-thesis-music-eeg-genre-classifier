# Code for the thesis

As mentioned before, this directory consists of all the code snippets responsible for various different tasks along the EEG pipeline (Acquisition, Denoising, Feature-extraction and Classification). Each of these steps has a designated directory with their own README. Below, you will find an overview of all directories:

1. **Acquisition**: This folder contains all the code necessary for acquiring and segmenting the EEG data, so supporting the main experiment. Refer to this [README](acquisition/README.md).
2. **Denoising**: This folder contains functions responsible for cleaning the measured EEG signal up. Refer to this [README](denoising/README.md).
3. **Feature-Extraction**: This folder handles feature extraction of cleaned EEG channel data in each domain. Refer to the respective [README](feature_extraction/README.md) for more information.
4. **Classification**: In this directory, the classification pipeline can be found including feature analysis and evaluations. Refer to the respective [README](classification/README.md).

## Important Note

Please make sure that you have followed the steps of setting up the virtual environment as it is needed throughout all pieces of code, no matter their purpose.