import numpy as np
import cv2
import pickle
import tensorflow as tf
from transformers import BertTokenizerFast, BertForTokenClassification, pipeline
import inflect

def plural_to_singular(plural_animal: str) -> str:
    p = inflect.engine()
    singular_animal = p.singular_noun(plural_animal)
    return singular_animal if singular_animal else plural_animal

# Function to save LabelEncoder
def save_label_encoder(label_encoder, filename="./label_encoder.pkl"):
    with open(filename, "wb") as f:
        pickle.dump(label_encoder, f)

# Function to load LabelEncoder
def load_label_encoder(filename="./label_encoder.pkl"):
    with open(filename, "rb") as f:
        return pickle.load(f)


######### FUNCTIONS FOR NER MODEL #########

# Function to load NER-model
def load_ner_model(model_dir):
    tokenizer = BertTokenizerFast.from_pretrained(model_dir)
    model = BertForTokenClassification.from_pretrained(model_dir)
    nlp = pipeline('ner', model=model, tokenizer=tokenizer, aggregation_strategy="simple")
    return nlp

# Function to extract animals from text
def extract_animals_from_text(nlp, text):
    ner_results = nlp(text)
    animals = [plural_to_singular(entity["word"].strip().lower()) for entity in ner_results if "ANIMAL" in entity["entity_group"]]
    if animals:
        return animals
    return None

######### FUNCTIONS FOR IMAGE CLASSIFICATION MODEL ######### 

# Function to preprocess an image
def preprocess_image(image_path, img_size=224):
    image = cv2.imread(image_path)
    image = cv2.resize(image, (img_size, img_size))
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = image / 255.0
    return image

# Function to load image classification model
def load_image_classifier(model_path):
    model = tf.keras.models.load_model(model_path)
    return model

# Function to predict label of a given image 
def predict_image_class(model, img_path, encoder_path):
    img = preprocess_image(img_path) 
    label_encoder = load_label_encoder(encoder_path)
    y_pred = np.argmax(model.predict(img.reshape(1, img.shape[0], img.shape[1], img.shape[2]), verbose=False), axis=-1)
    return label_encoder.inverse_transform(y_pred)[0]