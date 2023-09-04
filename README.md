
<div align="center">
<h1 align="center">
          <img src="https://img.icons8.com/?size=512&id=55494&format=png" width="80" />
        <img src="https://img.icons8.com/?size=512&id=kTuxVYRKeKEY&format=png" width="80" />
<br>Tumor Semantic Segmentation
</h1>
<h3>◦ Performs semantic segmentation at 3D Dicom images </h3>
<h3>◦ Developed with the software and tools listed below.</h3>

<p align="center">
<img src="https://img.shields.io/badge/Jupyter-F37626.svg?style&logo=Jupyter&logoColor=white" alt="Jupyter" />
<img src="https://img.shields.io/badge/Python-3776AB.svg?style&logo=Python&logoColor=white" alt="Python" />
<img src="https://img.shields.io/badge/Markdown-000000.svg?style&logo=Markdown&logoColor=white" alt="Markdown" />
<img src="https://img.shields.io/badge/Numpy-000000.svg?style&logo=Numpy&logoColor=white" alt="Markdown" />
<img src="https://img.shields.io/badge/Scikit-000000.svg?style&logo=Scikitlearn&logoColor=white" alt="Markdown" />
<img src="https://img.shields.io/badge/Tensorflow-000000.svg?style&logo=Tensorflow&logoColor=white" alt="Markdown" />
<img src="https://img.shields.io/badge/Keras-000000.svg?style&logo=Keras&logoColor=white" alt="Markdown" />
<img src="https://img.shields.io/badge/Monai-000000.svg?style&logo=Monai&logoColor=white" alt="Markdown" />

</p>
</div>

---

## 📒 Table of Contents
- [📒 Table of Contents](#-table-of-contents)
- [📍 Overview](#-overview)
- [⚙️ Features](#-features)
- [📂 Project Structure](#project-structure)
- [🧩 Modules](#modules)
- [🚀 Getting Started](#-getting-started)
- [👏 Contacts](#-contacts)

---
## 📝 Project Blog: Kidney Tumor Segmentation with AI

In the field of medical imaging and radiation therapy, accurate segmentation of tumors plays a pivotal role in ensuring precise and effective treatment. One such endeavor is the "Tumor-semantic-segmentation" project, where we harness the power of artificial intelligence (AI) to revolutionize kidney tumor segmentation.

### 🩺 The Challenge of Kidney Tumor Segmentation

Kidney tumors come in various shapes and sizes, making manual segmentation a daunting task for radiologists. Traditional approaches often lead to subjective results, leaving room for errors that can impact patient treatment outcomes. Our project seeks to address these challenges by employing state-of-the-art techniques in AI and computer vision.

### 🤖 Residual UNet: The Hero of Kidney Tumor Segmentation

At the heart of our solution lies the UNet architecture, a deep-learning model known for its exceptional performance in image segmentation tasks. UNet's ability to capture intricate details while preserving context makes it the perfect candidate for kidney tumor segmentation.
![image](https://github.com/harshp77/Tumor-semantic-segmentation/assets/76607486/69e84c5f-c665-4339-a818-e886942bd766)


### 📊 Leveraging MONAI and PyTorch

We understand that working with medical imaging data requires a specialized toolkit. That's why we turn to MONAI, a domain-specific framework designed for medical imaging, to handle our three-dimensional data. Complemented by PyTorch, we create a robust pipeline for training and testing our UNet model. Working with Dicom images has been one the most complications attached to it due to its number of axis to work on but MONAI has made it much simpler providing us the augmentation techniques and the pipeline structure.
![image](https://github.com/harshp77/Tumor-semantic-segmentation/assets/76607486/4a374bbb-62fa-4f32-bd71-9382d4004dc1)


### 🌟 Empowering Remote Healthcare

Our project's impact goes beyond just accurate segmentation. By enabling remote healthcare monitoring, we aim to extend the benefits of our model to underserved communities. Patients can receive expert-level care even in remote areas, thanks to our AI-driven solution.

### 🧠 Explainable AI

We don't stop at segmentation; we want to provide insights. Our model's explainability feature helps radiologists and patients understand why a particular segmentation was made. This transparency fosters trust in AI-assisted diagnostics.

### 🌐 The Future: Federated Learning

As we continue to evolve, we're exploring federated learning, a technique that allows our model to learn from multiple sources, including different hospitals. This collaborative approach enhances the model's generalization while respecting data privacy and security.


Stay tuned for updates, and let's make a difference together! 🌟




## 📍 Our AIM

Accurate segmentation of organs-at-risk (OARs) is crucial for efficient radiation therapy planning in kidney tumor treatment. In this project, we leverage the UNet architecture to perform semantic segmentation of kidney tumors, facilitating image-guided radiation therapy (IGRT). We utilize the MONAI library to handle three-dimensional data loading and PyTorch for subsequent training and testing. Our UNet architecture is designed to:

- Enable remote healthcare monitoring in underserved communities.
- Provide an explainable AI-based solution for radiologists.
- Explore federated learning across different hospitals (work in progress).
![image](https://github.com/harshp77/Tumor-semantic-segmentation/assets/76607486/f12b62de-a6f2-412c-bbf9-89c4afe56a5f)


---

## ⚙️ Features

- Accurate segmentation of kidney tumors.
- Image-guided radiation therapy (IGRT) support.
- Utilizes the UNet architecture.
- Explains AI-based decisions.
- Ongoing work on federated learning.

---

## 📂 Project Structure

### Modules

<details open><summary>Root</summary>

| File                                                                                                 | Summary                   |
| ---                                                                                                  | ---                       |
| [preprocess.py](https://github.com/harshp77/Tumor-semantic-segmentation.git/blob/main/preprocess.py) | HTTPStatus Exception: 429 |
| [runner.ipynb](https://github.com/harshp77/Tumor-semantic-segmentation.git/blob/main/runner.ipynb)   | HTTPStatus Exception: 429 |
| [runner.py](https://github.com/harshp77/Tumor-semantic-segmentation.git/blob/main/runner.py)         | HTTPStatus Exception: 429 |
| [testing.ipynb](https://github.com/harshp77/Tumor-semantic-segmentation.git/blob/main/testing.ipynb) | HTTPStatus Exception: 429 |
| [utilities.py](https://github.com/harshp77/Tumor-semantic-segmentation.git/blob/main/utilities.py)   | HTTPStatus Exception: 429 |

</details>


## Data

#### [MICCAI 2015 Head and Neck Auto Segmentation Challenge](http://www.imagenglab.com/wiki/mediawiki/index.php?title=2015_MICCAI_Challenge)

Executing the following code will download, extract, and split the dataset.

```bash
    cd data
    python download.py miccai
```

![image](https://user-images.githubusercontent.com/76607486/200163824-01b5d438-402b-41a3-9044-53cfb0771bb4.png)

## 🚀 Getting Started

Executeing the Jupyter notebooks individually.
To do this, you must first start the notebook server by running the following in powershell :

``` bash
    python -m notebook 
```

### Software implementation

All source code used to generate the results and figures in the paper are in
the `code` folder.
The calculations and figure generation are all ran inside
[Jupyter notebooks](http://jupyter.org/).

This will start the server and open your default web browser to the Jupyter
interface. In the page, go into the `code/notebooks` folder and select the
notebook that you wish to view/run.

The notebook is divided into cells (some have text while other have code).
Each cell can be executed using `Shift + Enter`.
Executing text cells does nothing and executing code cells runs the code
and produces it's output.
To execute the whole notebook, run all cells in order.

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


## Results


So far after runing x epochs we have the following performance

![image](https://github.com/harshp77/Tumor-semantic-segmentation/assets/76607486/1ffe8ba8-5abb-484a-882b-7a0401c5f6b2)


![image](https://user-images.githubusercontent.com/76607486/200165602-54f9f0da-fd80-4775-8f12-11f93327eb54.png)


![image](https://github.com/harshp77/Tumor-semantic-segmentation/assets/76607486/fb492786-3a17-4c3b-ad89-58cc5617fd35)



### Additional Requirements

We use [Weights and Biases](https://github.com/wandb/client) for keeping a track of all our experiments and results.


<!-- Contact -->
## 🤝 Contact

Your Name - [Harsh Pandey]() - harsh20101@iiitnr.edu.in

Project Link: [https://github.com/harshp77/AIFD](https://github.com/harshp77/AIFD)







