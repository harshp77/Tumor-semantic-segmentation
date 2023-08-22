# Tumor-semantic-segmentation

#### By Harsh Pandey and Subhanshu Arya

## Introduction
Accurate segmentation of organs-at-risks (OARs) is the key step for efficient planning of radiation therapy for kidney tumor treatment. In this project, we use the UNet architecture to segment kidney, which will aid in image-guided radiation therapy (IGRT). MONAI is being used to load the three-dimensional data. And PyTorch for subsequent data training and testing. The UNet architecture used for training is described below:

> Enabling remote communities for better healthcare monitoring, aiding inexperienced radiologists

> X-Ai based solution - this assisting model will even explain how our model has come with this solution.
 
> Federative solution -- utilizing different nodular hospital for participating in cross silo federated learning. (In process)

![image](https://github.com/harshp77/Tumor-semantic-segmentation/assets/76607486/1ffe8ba8-5abb-484a-882b-7a0401c5f6b2)


--------------------------------------Expected Output---------------------------------------- 


## Software implementation

All source code used to generate the results and figures in the paper are in
the `code` folder.
The calculations and figure generation are all ran inside
[Jupyter notebooks](http://jupyter.org/).


## Data

#### [MICCAI 2015 Head and Neck Auto Segmentation Challenge](http://www.imagenglab.com/wiki/mediawiki/index.php?title=2015_MICCAI_Challenge)

Executing the following code will download, extract, and split the dataset.

```bash
    cd data
    python download.py miccai
```

![image](https://user-images.githubusercontent.com/76607486/200163824-01b5d438-402b-41a3-9044-53cfb0771bb4.png)

## Requirements

*We highly recommend using the specified versions of the listed packages.*

### Base Requirements

1. Python (3.7)
2. [pynrrd](https://github.com/mhe/pynrrd) (0.4) - For loading MICCAI data in `.nrrd` format
3. Tqdm - For displaying progress bars
4. PyTorch (1.7)
5. Torchvision (0.8)
6. [Albumentations](https://github.com/albumentations-team/albumentations) (0.5) - For data augmentation and transforms
7. [MONAI](https://github.com/Project-MONAI/MONAI) (0.3) - For domain specific models, losses, metrics, etc
8. [PyTorch-Lightning](https://github.com/PyTorchLightning/pytorch-lightning) (1.0)

## Dependencies

You'll need a working Python environment to run the code.

We use `python` virtual environments to manage the project dependencies in
isolation.
Thus, you can install our dependencies without causing conflicts with your
setup (even with different Python versions).

setting up the virtual Env -

``` bash
    python -m venv IterX
    cd IterX
    source bin/activate
```
after activating the isolated local env install all the dependency using the requirements.txt file provided below

``` bash
    pip install requirements.txt
```


## results


Executeing the Jupyter notebooks individually.
To do this, you must first start the notebook server by running the following in powershell :

``` bash
    python -m notebook 
```

This will start the server and open your default web browser to the Jupyter
interface. In the page, go into the `code/notebooks` folder and select the
notebook that you wish to view/run.

The notebook is divided into cells (some have text while other have code).
Each cell can be executed using `Shift + Enter`.
Executing text cells does nothing and executing code cells runs the code
and produces it's output.
To execute the whole notebook, run all cells in order.


So far after runing x epochs we have the following performance

![image](https://user-images.githubusercontent.com/76607486/200165602-54f9f0da-fd80-4775-8f12-11f93327eb54.png)

### Additional Requirements

We use [Weights and Biases](https://github.com/wandb/client) for keeping a track of all our experiments and results.

