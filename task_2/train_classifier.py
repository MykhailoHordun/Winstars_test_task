import argparse
import os
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.applications import EfficientNetB3
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import GlobalAveragePooling2D, Dense, Dropout
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import ReduceLROnPlateau

from utils import preprocess_image, save_label_encoder

# Function to load dataset and encode labels
def load_dataset(dataset_path, labels_file, img_size):
    with open(labels_file, 'r') as f:
        animal_names = f.read().split('\n')
    
    data, labels = [], []
    for animal in animal_names:
        animal_dir = os.path.join(dataset_path, animal)
        for img_name in os.listdir(animal_dir):
            img_path = os.path.join(animal_dir, img_name)
            data.append(preprocess_image(img_path, img_size))
            labels.append(animal)
    
    data = np.array(data)
    labels = np.array(labels)
    label_encoder = LabelEncoder()
    labels = label_encoder.fit_transform(labels)
    save_label_encoder(label_encoder, "label_encoder.pkl")
    
    return train_test_split(data, labels, test_size=0.2, random_state=42), label_encoder

# Function to build the EfficientNetB3-based model
def build_model(num_classes, img_size):
    base_model = EfficientNetB3(weights='imagenet', include_top=False, input_shape=(img_size, img_size, 3))
    
    # Fine-tune top layers
    for layer in base_model.layers[-50:]:
        layer.trainable = True
    
    # Define the model architecture
    model = Sequential([
        base_model,
        GlobalAveragePooling2D(),
        Dense(512, activation='relu'),
        Dropout(0.5),
        Dense(num_classes, activation='softmax')
    ])
    
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4), 
                  loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    return model

# Main function to handle argument parsing and model training
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_path", default="./data/animal_classification_data/animals/animals", help="Path to the dataset")
    parser.add_argument("--labels_file", default="./data/animal_classification_data/name of the animals.txt", help="Path to the labels file")
    parser.add_argument("--img_size", type=int, default=24, help="Image size (square)")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size for training")
    parser.add_argument('--epochs', type=int, default=50, help="Number of training epochs")
    parser.add_argument("--output_model", default="./animal_classification_model.h5", help="Path to save the trained model")
    args = parser.parse_args()

    # Load dataset and label encoder
    (X_train, X_test, y_train, y_test), label_encoder = load_dataset(args.dataset_path, args.labels_file, args.img_size)

    # Data augmentation for training
    datagen = ImageDataGenerator(
        rotation_range=20,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
        fill_mode='nearest' 
    )

    # Build and compile the model
    model = build_model(len(label_encoder.classes_), args.img_size)

    # Learning rate scheduler callbeck
    lr_scheduler = ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=5, min_lr=1e-6)

    # Train the model
    model.fit(
        datagen.flow(X_train, y_train, batch_size=args.batch_size),
        epochs=args.epochs,
        validation_data=(X_test, y_test),
        callbacks=[lr_scheduler],
        verbose=2
    )

    # Save trained model
    model.save(args.output_model)
    
if __name__ == "__main__":
    main()
