# Decoding Music Perception: EEG-Based Genre Classification

Welcome to the repository of my Bachelor's thesis, where I explore the classification of music genres on the basis of EEG signals (meaning brain activity). By employing a prototype of EEG headphones, I record my brain activity while listenting to different kinds of music available through the GTZAN dataset (a dataset optimized for genre classification tasks). The acquired data then is processed by special denoising and feature extraction techniques to ultimately classify a music genre according to the signal data.

Big thanks to my supervisor Simon Pistrosch and the [Chair of Health Informatics at TUM](https://www.kiinformatik.mri.tum.de/en/chichair-health-informatics) for supporting me throughout this journey and giving valuable insights. I learned a lot during those four months!

As a quick link, you can find the written thesis [here](thesis/build/thesis.pdf). Enjoy :)
The end results and analysis of the Genre classification can be found [here](code/classification/README.md)

If you have any questions, feel free to connect :)

## Abstract
Music plays a key role in everyday life, being entertaining and engaging complex
cognitive and emotional processes in the human brain. Nonetheless, the neural mech-
anisms behind music perception remain unknown. Therefore, this thesis explores
whether Electroencephalography (EEG) signals can be used to reliably decode music
perception, aiming to bridge the gap between Music Information Retrieval (MIR) and
neurosciences. A dedicated dataset for EEG based Music Genre classification is created
to contribute to this area, named the Multimodal Signals & EEG for Classification
of Genres (MuSiCGen) dataset. By utilising the GTZAN music dataset, EEG signals
have been recorded by a multimodal biosignal headset while listening to music. The
recorded signal then underwent a custom denoising process to eliminate most artifacts,
followed by an exhaustive feature extraction combined with specialised feature selection
methods. Based on these parameters, machine learning classifiers are trained to predict
the corresponding genres, which are then analysed regarding accuracy, F1-score, and
their confusion matrices. To compare with a state-of-the-art EEG dataset, the same
approach is followed with the Database for Emotion Analysis using Physiological
Signals (DEAP), which is evaluated in the same way as the newly introduced dataset.
Results demonstrate that genre-specific neural patterns can be detected above chance
level. Concreteley, mean accuracies of 68.3% and 20.2% were measured for binary
classification on the DEAP dataset and ten-class classification on the MuSiCGen dataset,
respectively. Although signal quality and device limitations posed a challenge, the key
findings show promising evidence that EEG signals possess meaningful information
about musical perception. Overall, this thesis lays the groundwork for further re-
search in the area of EEG-based MIR by providing a novel database alongside detailed
methodology for Music Genre Recognition (MGR) using EEG signals. Future studies
can build upon this thesis by employing larger subject pools, advanced deep learning
architectures, or multimodal data fusion.

## Project structure
This project has 3 main directories:

- In the `thesis` folder, you will find the LATEX files of my written thesis including the thesis as a pdf file as well. For further information, please refer to the [README](thesis/README.md) in the thesis directory. This file contains all the information you need to run and build the thesis yourself.

- The `code` directory contains all relevant code responsible to support the EEG signal pipeline during the thesis project. Mainly python scripts and Jupyter notebooks will await you in this folder. For further information, please also pay attention to the [README](code/README.md). This especially contains content on the further seperation of this folder.

- The `data` folder is still empty, but in this folder, you will not only put all necessary data but the generated data from some code sections will also be put there.

## Code environment setup

If you are intending to use the code in this project, please follow this guide. Also make sure to follow this guideline for the nix environment: [README](/_README.md):
### Prerequisites

Ensure you have the following installed:
- Python 3.8 or higher
- `pip` (Python package manager)

### Setting Up the Virtual Environment

1. Create a virtual environment:
  ```bash
  python3 -m venv .venv
  ```

2. Activate the virtual environment:
  - On macOS/Linux:
    ```bash
    source .venv/bin/activate
    ```
  - On Windows:
    ```bash
    .venv\Scripts\activate.bat
    ```

4. Install the required dependencies:
  ```bash
  pip install -r requirements.txt
  ```


### Setting up MATLAB
For the MATLAB functions, not much of setup has to be done. You just have to make sure that you have one of the newest versions/licences of MATLAB installed on your machine. For a tutorial setup, please visit the official [MATLAB Homepage](https://matlab.mathworks.com/)

In order to run the MATLAB functions in this repo, you have to download the [load_xdf function](https://github.com/xdf-modules/xdf-Matlab) via Github. You then have to add the function on the top level to MATLAB's path. There are 2 possibilities for that: 

1. In the GUI, go to `File / Set Path / Add Folder` and add the path to the package.
2. In the command line, type in the following command:  
  ```bash
  addpath('C:\\path\\to\\xdf_repo\\Matlab\\xdf-Matlab')
  ```
