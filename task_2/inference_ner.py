import argparse
from utils import load_ner_model, extract_animals_from_text 

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_dir", default="./ner_model", help="Directory with pretrained model")
    parser.add_argument("--text", type=str, required=True, help="Input text for animal extraction")
    args = parser.parse_args()

    # Loading NER model
    nlp = load_ner_model(args.model_dir)
    # Extracting animal names from text
    animals = extract_animals_from_text(nlp, args.text)
    print(animals)

if __name__ == "__main__":
    main()

