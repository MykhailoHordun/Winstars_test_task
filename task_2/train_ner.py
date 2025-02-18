import argparse
from transformers import BertTokenizerFast, BertForTokenClassification, TrainingArguments, Trainer
from datasets import Dataset
import json
import numpy as np

# Define label mapping
label_list = ["O", "B-ANIMAL", "I-ANIMAL"]
label_to_id = {label: i for i, label in enumerate(label_list)}
id_to_label = {i: label for label, i in label_to_id.items()}

# Load the annotated dataset from JSON file
def load_dataset(file):
    with open(file, "r", encoding="utf-8") as f:
        data = json.load(f)
    return data

# Funcion to tokenize sentences and align the entity labels with token offsets
def tokenize_and_align_labels(example):
    
    # Tokenize the sentence while returning offset mappings
    tokenized_inputs = tokenizer(example['sentence'], truncation=True, return_offsets_mapping=True, padding=True)
    offsets = tokenized_inputs.pop('offset_mapping')
    # Initialize labels as "O" for each token
    token_labels = ["O"] * len(tokenized_inputs["input_ids"])

    for entity in example['entities']:
        start_char, end_char, label = entity['start'], entity['end'], entity['label']
        first = True # Indicator for beginning of entity span
        for i, (token_start, token_end) in enumerate(offsets):
            # Check for overlap between token and entity span
            if token_end > start_char and token_start < end_char:
                if first:
                    token_labels[i] = "B-" + label
                    first = False
                else:
                    token_labels[i] = "I-" + label
    
    # Convert string labels to their corresponding integer IDs
    tokenized_inputs["labels"] = [label_to_id[label] for label in token_labels]
    return tokenized_inputs

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", default="bert-base-cased", help="Name of pretrained model")
    parser.add_argument("--dataset_path", default="./cleaned_animal_dataset.json", help="Path to dataset file")
    parser.add_argument("--output_dir", default="./ner_model", help="Directory for saving model")
    parser.add_argument("--num_train_epochs", type=int, default=1, help="Number of training epochs")
    args = parser.parse_args()

    # Load dataset from JSON file
    data = load_dataset(args.dataset_path)
    
    # Create a HF Dataset object from the list of examples
    dataset = Dataset.from_list(data)
    global tokenizer
    tokenizer = BertTokenizerFast.from_pretrained(args.model_name)
    model = BertForTokenClassification.from_pretrained(
        args.model_name,
        num_labels = len(label_list),
        id2label=id_to_label,
        label2id=label_to_id
    )

    # Tokenize the dataset, align the labels, and remove the entities column
    tokenized_dataset = dataset.map(tokenize_and_align_labels, batched=False, remove_columns=['entities'])

    # Split into training and eveluation sets (simple 90/10)
    split = tokenized_dataset.train_test_split(test_size=0.3)
    train_dataset = split['train']
    eval_dataset = split['test']

    # Define training arguments
    training_args = TrainingArguments(
        output_dir=args.output_dir,
        num_train_epochs=args.num_train_epochs,
        per_device_train_batch_size=1,
        per_device_eval_batch_size=1, 
        evaluation_strategy="epoch",
        save_strategy="epoch",
        logging_dir="./logs",
        logging_steps=50,
        weight_decay=0.01,
        learning_rate=5e-5,
        report_to="none",
    )

    # Define a compute metric function
    def compute_metrics(p):
        predictions, labels = p
        predictions = np.argmax(predictions, axis=2)

        true_labels = []
        true_predictions = []

        for pred, lab in zip(predictions, labels):
            for p_val, l_val in zip(pred, lab):
                if l_val != -100:
                    true_labels.append(l_val)
                    true_predictions.append(p_val)
        
        accuracy = np.sum(np.array(true_predictions) == np.array(true_labels)) / len(true_labels)
        return {"accuracy": accuracy}

    # Initialize the Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        compute_metrics=compute_metrics,
        tokenizer=tokenizer
    )

    # Train the model
    trainer.train()

    # Save the fine-tuned model and tokenizer
    trainer.save_model(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)
    print("Training finished")

    

if __name__ == "__main__":
    main()