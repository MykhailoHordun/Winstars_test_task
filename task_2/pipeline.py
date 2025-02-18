import argparse
from utils import load_ner_model, extract_animals_from_text, load_image_classifier, predict_image_class

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--text", type=str, required=True, help="Input text for animal extraction")
    parser.add_argument("--image_path", type=str, required=True, help="Path to the image to be classified")
    parser.add_argument("--ner_model_dir", default="./ner_model", help="Directory containing the pretrained NER model")
    parser.add_argument("--img_model_path", default='./animal_classification_model.h5', help="Path to the pretrained image classification model")
    parser.add_argument("--encoder_path", type=str, default="./label_encoder.pkl", help="Path to the label_encoder")
    args = parser.parse_args()

    # Load both models
    nlp = load_ner_model(args.ner_model_dir)
    img_model = load_image_classifier(args.img_model_path)

    # Extraction of animal names from user prompts
    extracted_animals = extract_animals_from_text(nlp, args.text)
    if extracted_animals is None:
        print(False)
        return
    
    # Prediction of image class
    predicted_animal = predict_image_class(img_model, args.image_path, args.encoder_path)
    
    # Comparison of results
    for animal in extracted_animals:
        if animal == predicted_animal:
            print(True)
            return
    
    print(False)
    return


if __name__ == "__main__":
    main()