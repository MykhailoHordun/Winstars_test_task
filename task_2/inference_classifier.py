import argparse
from utils import load_image_classifier, predict_image_class

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", default='./animal_classifier.h5', help="Path to the pretrained model")
    parser.add_argument("--image_path", type=str, required=True, help="Path to the image to be classified")
    parser.add_argument("--image_size", type=int, default=224, help="Image size (square)")
    parser.add_argument("--encoder_path", type=str, required=True, help="Path to the label_encoder")
    args = parser.parse_args()

    # Loading image classification model
    model = load_image_classifier(args.model_path)
    # Predicting the label of the photo
    y_pred = predict_image_class(model, args.image_path, args.image_size, args.encoder_path)
    print(y_pred)

if __name__ == "__main__":
    main()