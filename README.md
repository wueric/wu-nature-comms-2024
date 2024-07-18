# Natural image reconstruction from retinal ganglion cell spikes 

All of the code used to fit the encoding models and generate reconstructions is provided in this repository. Because the
underlying datasets are large, and the full procedure for fitting encoding models and generating reconstructions is computationally intensive, requiring several
days of GPU computing, prefit models and a small subset of the data for demonstrating reconstruction are provided.

## Dependencies and installation 

### Required dependencies

* numpy (https://numpy.org/)
* scipy (https://scipy.org/)
* Pytorch (https://pytorch.org/
* statsmodels (https://www.statsmodels.org/stable/index.html)
* shapely (https://shapely.readthedocs.io/en/stable/manual.html)
* scikit-image (https://scikit-image.org/)
* tqdm (https://tqdm.github.io/)
* h5py (https://www.h5py.org/)
* cython (https://cython.org/)
* pybind11 (https://pybind11.readthedocs.io/en/stable/)
* lpips (https://github.com/richzhang/PerceptualSimilarity) **needed to run the full repo, but not for the demonstration notebooks.**

A working C and C++ compiler, as well as a working CUDA installation compatible with Pytorch, are required as well.

The software was developed and tested with Python 3.9, using Pytorch 1.11.0. The software is expected to run using newer versions of Python and Pytorch. 

### Special hardware requirements
* NVIDIA GPU 
  * with VRAM at least 8GB for reconstructions
  * with VRAM at least 32GB for refitting encoding likelihood models to the complete dataset

### Installation
1. Set up a conda (https://docs.conda.io/en/latest/) environment with Python >=3.9
2. Install dependencies with `conda install numpy scipy matplotlib pybind11 cython statsmodels shapely h5py tqdm scikit-image`
3. Install Pytorch from https://pytorch.org/get-started/locally/. Note that a CUDA-compatible installation is required.
4. Download the repository https://github.com/wueric/movie_upsampling, and install the Python package by running the setup script `python setup.py install` in the root directory of the repository. This package contains compiled code to temporally upsample the visual stimulus on an NVIDIA GPU.
5. Download the repository https://github.com/wueric/spikebinning, and install the Python package by running the setup script `python setup.py install` in the root directory of the repository. This package contains compiled code to discretize the observed spike trains into time bins. **Needed to run the full repo, but not for the demonstration notebooks.**

## Provided models and demonstration data

Fitted models and demonstration data are provided in a .zip file.

## Running the demonstration notebooks

The demonstration notebook for flashed reconstructions is `SUBMISSION_flashed_reconstruction_demo.ipynb`. You will need to change
the path specified at the top of the file to point to the location of the provided models and data.

The demonstration notebook for jittered eye movements movie reconstructions is `SUBMISSION_jitter_reconstruction_demo.ipynb`.
You will need to change
the path specified at the top of the file to point to the location of the provided models and data.

## Running the full analysis pipeline

Running the full analysis pipeline on the complete dataset requires software to parse custom data file formats. This software is not currently publicly available. 

#### For flashed reconstruction
1. Fitting LNBRC encoding models to flashed data, consisting of:
   1. Performing a grid search over possible encoding model hyperparameters over a small subset of cells, with test partition encoding log likelihood as the objective
   2. Using the hyperparameters found in the grid search, fitting LNBRC models to every cell using convex minimization

  The shell script `glm_grid_and_fit.sh` using the option `flash` executes both the grid search and fitting. This procedure must be repeated for every RGC cell type included in the preparation.
2. Reconstruction, consisting of:
   1. Performing a grid search over possible reconstruction hyperparameters, with reconstruction MS-SSIM over an 80-image subset of the test partition as the objective
   2. Using the hyperparameters in the grid search, reconstructing the images using the half-quadratic splitting algorithm described in the manuscript
   
  The shell script `flash_reconstruct_grid_and_generate.sh` executes both the grid search and reconstruction steps.

3. Evaluation of image quality, using `eval_reconstruction_quality_flashed.sh`

#### For jitter eye movements movie reconstruction
1. Fitting LNBRC encoding models to jitter eye movements movie data, consisting of
  1. Performing a grid search over possible encoding model hyperparameters over a small subset of cells, with test partition encoding log likelihood as the objective
  2. Using the hyperparameters found in the grid search, fitting LNBRC models to every cell using convex minimization

  The shell script `glm_grid_and_fit.sh` using the option `jitter` executes both the grid search and fitting. This procedure must be repeated for every RGC cell type included in the preparation.
2. Reconstruction, consisting of
   1. Performing a grid search over possible reconstruction hyperparameters in the case that eye movements are known exactly
   2. Using the above hypeparameters, performing a second grid search over possible weights on the eye movements probability term
   3. Using the hyperparameters from (i) and (ii), reconstruct the images for the joint estimation case, the known eye movements case, and the zero (ignore) eye movements case.

  The shell script `eye_movements_reconstruct_grid_and_generate.sh` executes all of these steps.
3. Evaluation of image quality, using `eval_reconstruction_quality_eye_movements.sh`
