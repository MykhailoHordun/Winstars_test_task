# Animal Recognition Pipeline
## Overview

This project implements a machine learning pipeline that combines Named Entity Recognition (NER) and Image Classification to determine whether a user-provided statement about an animal in an image is correct.

## Pipeline Flow:

1. The user provides a text statement (e.g., "There is a cow in the picture.") and an image.
2. The NER model extracts animal names from the text.
3. The image classification model predicts the animal category in the image.
4. The system returns `True` if the predicted image label matches the extracted animal from the text, otherwise `False`.

## Project Structure

```bash
.
└── task_2/
    ├── images/                      # Folder with images for testing
    │   ├── cat.jpg                  
    │   ├── cow.jpg                  
    │   ├── dog.jpg                  
    │   └── random_image.jpg         
    ├── animal_dataset_criation.py   # Script for creating NER dataset for extraction names of animals 
    ├── train_classifier.py          # Script to create and train image classification model
    ├── train_ner.py                 # Script to create and train NER model
    ├── inference_classifier.py      # Script to load and use pretrained image classification model
    ├── inference_ner.py             # Script to load and use pretrained NER model
    ├── pipeline.py                  # Python script for the entire pipeline that takes 2 inputs (text and image) and provides 1 boolean value as an output
    ├── utils.py                     # File with usefull functions
    ├── cleaned_animal_dataset.json  # Dataset for NER model reaining
    ├── data_analysis.ipynb          # Jupyter Notebook with exploratory data analysis of dataset
    ├── demo.ipynb                   # Jupyter Notebook with demonstarion of the pipeline work
    ├── requirements.txt             # List of required dependencies
    ├── label_encoder.pkll           # File with LabelEncoder used in testing of pretrained image classification model
    └── readme.md                    # Documentation
```

## Installation

1. Clone the repository:
```sh
git clone https://github.com/MykhailoHordun/Winstars_test_task.git
cd task_2
```
2. Create a virtual environment (optional):
```sh
python -m venv venv
venv\Scripts\activate
```
3. Install dependencies:
```sh
pip install -r requirements.txt
```
4. Install the pretrained image classification model and NER model from [Google Drive](https://drive.google.com/drive/folders/1EicajHaWMH1dGHXn1WKgURGUDd_YKWY3?usp=sharing) and place them into the `task_2` folder.
Alternatively, you can train both models by following the **Train Models** instructions.

## Train Model
### Train the Image Classification Model

1. To train image classification model Kaggle [Animal Image Dataset](https://www.kaggle.com/datasets/iamsouravbanerjee/animal-image-dataset-90-different-animals) was used.
Download it and place into the `task_2/data` folder.
2. Run with default arguments:
```sh
python train_classifier.py
```
If you want to change any arguments run next script:
```sh
python train_classifier.py --dataset_path="path/to/your/dataset" \
                           --labels_file="path/to/labels/file" \
                           --batch_size=your_batch_size \
                           --epochs=num_of_training_epochs \
                           --output_model="path/to/save_madel/file.h5"
```

### Train the NER Model

To train NER model run:
```sh
python train_ner.py
```

## Running Inference
### Classify an Image

```sh
python inference_classifier.py --model_path="path/to/pretrained/model.h5" \
                               --image_path="path/to/image.jpg"
```
Or if you keep the default output model path while training use:
```sh
python inference_classifier.py --image_path="path/to/image.jpg"
```

### Extract Animals from Text (NER)

```sh
python inference_ner.py --text="Your input text"
```

### Running the Full Pipeline
```sh
python pipeline.py --text="Your input text" --img_path="path/to/image.jpg"
```

## Implementation Details

**NER Model:** 

In this implementation we use a fine-tuned BERT model to recognize animal names in text. 
To fine-tune this model we collected dataset using wikipedia articles about animals from 
[Animal Image Dataset](https://www.kaggle.com/datasets/iamsouravbanerjee/animal-image-dataset-90-different-animals). The process of collecting, cleaning and analyzing the dataset
can be found in `animal_dataset_criation.py` and `data_analysis.ipynb`. The resulting dataset is stored in `cleaned_animal_dataset.json` file.

**Image Classification Model**

In this implementation we use a fine-tuned EfficientNetB3 model for animal image classification.
To fine-tune this model we used [Animal Image Dataset](https://www.kaggle.com/datasets/iamsouravbanerjee/animal-image-dataset-90-different-animals).
This dataset contains 90 different classes of animals (60 photos per class). To expand the dataset, we used data augmentation during training.

## Results
The pipeline successfully extracts animal names from text and compares them with the predicted animal class from images. 
The CNN classifier achieves high accuracy on the test dataset, and the NER model effectively identifies animal names.
Overall, the system provides a reliable method for verifying textual claims against image content.
