# MNIST Classification Task
## Overview

This project implements three different models for classifying handwritten digits from the MNIST dataset:

1. Random Forest (RF).
2. Feed-Forward Neural Network (NN).
3. Convolutional Neural Network (CNN).

Each model is encapsulated in a class that implements the MnistClassifierInterface, ensuring a consistent API for training and inference. A wrapper class, MnistClassifier, allows selection of the desired model via a parameter.

## Project Structure

* `main.py` - file containing thr implementation of the following classes:
  - MnistClassifierInterface
  - MnistRandomForestClassifier
  - MnistNeuralNetworkClassifier
  - MnistCNNClassifier
  - MnistClassifier
* `demo.ipynb` - jupyter notebook with a demonstration of the solution
* `requirements.txt`

## Installation

1. Clone the repository:
```sh
git clone https://github.com/MykhailoHordun/Winstars_test_task.git
cd task_1
```
2. Create a virtual environment (optional):
```sh
python -m venv venv
venv\Scripts\activate
```
3. Install dependencies:
```dh
pip install -r requirements.txt
```

## Usage

1. Open `demo.ipynb`
2. Run all cells

## Results

The CNN model generally achieves the highest accuracy on MNIST, followed by the NN, with the Random Forest model performing worst due to the nature of image data.
