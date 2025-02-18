import requests
from bs4 import BeautifulSoup
import re 
import json
import nltk

nltk.download('punkt')

def fetch_page_text(url):
    """
    Function to fetch the textual content of a web page given its URL.
    """
    response = requests.get(url)
    if response.status_code == 200:
        soup = BeautifulSoup(response.text, 'html.parser')
        # Remove unwanted script and style elements
        for element in soup(['script', 'style']):
            element.decompose()

        # Extract text 
        text = soup.get_text(separator=" ")
        return clean_text(text)
    else:
        print(f"Failed to fetch {url}")
        return ""

def clean_text(text):    
    """
    Cleans the fetched text by removing excessive whitespace, non-ASCII characters,
    and normalizing spaces.
    """
    text = re.sub(r'[^\x00-\x7F]+', '', text)  # Remove non-ASCII characters
    new_text_list = []
    text_list = text.split('\n')
    for t in text_list:
        t = re.sub(r'[\(\[].*?[\)\]]', "", t) # Remove [] () and their content from text
        # Normalize spaces
        t = re.sub(r'\s+', ' ', t)  
        t = re.sub(r' (?=[/.,:;])', '', t) 
        t = t.replace('/', '').replace('\"', '') # Remove / \" characters
        new_text_list.append(t.strip())
    return new_text_list

def annotate_sentence(sentence, animal_list):
    """
    Searches for animal names in the sentence using regex and returns a list
    of tuples indicating the start, end indices, and the tag "ANIMAL". 
    """
    entities = []
    for animal in animal_list:
        # Use word boundaries to ensure we match whole words (case insensitive)
        pattern = r'\b{}\b'.format(re.escape(animal))
        for match in re.finditer(pattern, sentence, re.IGNORECASE):
            entities.append({'start': match.start(), 'end': match.end(), 'label': "ANIMAL"})
    return entities

def process_text(text, animal_list):
    """
    Splits the full text into sentences, annotates each sentence with animal entities,
    and returns a dataset in which each entry is a dict with the sentence and its entities.
    """
    sentences = nltk.sent_tokenize(text)
    dataset = []
    for sent in sentences:
        entities = annotate_sentence(sent, animal_list)
        if entities: # Only include sentences that contain at leas one animal
            dataset.append({
                "sentence": sent,
                "entities": entities,
            })
    return dataset

def main():

    root_url = "https://en.wikipedia.org/wiki"
    # List that contains animals from classification dataset
    animal_list = ['antelope', 'badger', 'bat', 'bear', 'bee', 'beetle', 'bison', 'boar', 'butterfly', 'cat', 'caterpillar', 
                'chimpanzee', 'cockroach', 'cow', 'coyote', 'crab', 'crow', 'deer', 'dog', 'dolphin', 'donkey', 'dragonfly', 
                'duck', 'eagle', 'elephant', 'flamingo', 'fly', 'fox', 'goat', 'goldfish', 'goose', 'gorilla', 'grasshopper', 
                'hamster', 'hare', 'hedgehog', 'hippopotamus', 'hornbill', 'horse', 'hummingbird', 'hyena', 'jellyfish', 'kangaroo', 
                'koala', 'ladybugs', 'leopard', 'lion', 'lizard', 'lobster', 'mosquito', 'moth', 'mouse', 'octopus', 'okapi', 'orangutan', 
                'otter', 'owl', 'ox', 'oyster', 'panda', 'parrot', 'pelecaniformes', 'penguin', 'pig', 'pigeon', 'porcupine', 'possum', 
                'raccoon', 'rat', 'reindeer', 'rhinoceros', 'sandpiper', 'seahorse', 'seal', 'shark', 'sheep', 'snake', 'sparrow', 'squid', 
                'squirrel', 'starfish', 'swan', 'tiger', 'turkey', 'turtle', 'whale', 'wolf', 'wombat', 'woodpecker', 'zebra', 'badgers', 
                'bats', 'bears', 'bees', 'beetles', 'boars', 'butterflies', 'cats', 'caterpillars', 'chimpanzees', 'cockroaches', 'cows', 'coyotes', 
                'crabs', 'crows', 'deer', 'dogs', 'dolphins', 'donkeys', 'dragonflies', 'ducks', 'eagles', 'elephants', 'flamingos', 'flies', 'foxes', 
                'goats', 'goldfish', 'geese', 'gorillas', 'grasshoppers', 'hamsters', 'hares', 'hedgehogs', 'hippopotamuses', 'hippopotami', 'hornbills', 
                'horses', 'hummingbirds', 'hyenas', 'jellyfish', 'kangaroos', 'koalas', 'ladybugs', 'leopards', 'lions', 'lizards', 'lobsters', 'mosquitoes', 'moths', 'mice', 
                'octopuses', 'octopi', 'okapi', 'orangutans', 'otters', 'owls', 'oxen', 'oysters', 'pandas', 'parrots', 'pelecaniformes', 'penguins', 'pigs', 
                'pigeons', 'porcupines', 'possums', 'raccoons', 'rats', 'reindeer', 'rhinoceroses', 'randpipers', 'seahorses', 'seals', 'sharks', 'sheep', 'snakes', 
                'sparrows', 'squid', 'squirrels', 'starfish', 'swans', 'tigers', 'turkeys', 'turtles', 'whales', 'wolves', 'wombats', 'woodpeckers', 'zebras']

    full_dataset = []
    for anim in animal_list:
        url = f'{root_url}/{anim}'
        print(f"Fetching from {url}")
        text = fetch_page_text(url)
        text = list(set(text))
        for t in text:
            if t:
                dataset = process_text(t, animal_list)
                full_dataset.extend(dataset)
    
    # Save the annotated dataset to a JSON file
    with open('./222animal_ner_dataset.json', 'w', encoding='utf-8') as f:
        json.dump(full_dataset, f, ensure_ascii=False, indent=2)
    print("Dataset saved to animal_ner_dataset.json")


if __name__ == "__main__":
    main()