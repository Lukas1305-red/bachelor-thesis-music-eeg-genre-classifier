# Decoding Music Perception: EEG-Based Genre Classification

Welcome to the repository of my Bachelor's thesis, where I explore the classification of music genres on the basis of EEG signals (meaning brain activity). By employing a prototype of EEG headphones, I record my brain activity while listenting to different kinds of music available through the GTZAN dataset (a dataset optimized for genre classification tasks). The acquired data then is processed by special denoising and feature extraction techniques to ultimately classify a music genre according to the signal data.

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