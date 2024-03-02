import json
from sklearn.model_selection import train_test_split
##from nltk.tokenize import WhitespaceTokenizer
import re

# load the dataset
def load_dataset(file_path):
    with open(file_path, 'r') as file:
        return json.load(file)

# perform BIO encoding
def bio_encode(case):
    text = case['data']['text']
    annotations = case['annotations'][0]['result']
    ##tokens = text.split()
    pattern = r'\S+|\n+'
    tokens = re.findall(pattern, text)
    if text == "(See Principles of Statutory Interpretation by Justice G.P. Singh, 9th Edn., 2004 at p. \n\n 438.).":
        print(tokens)
    ##tk = WhitespaceTokenizer() 
    ##tokens = tk.tokenize(text)
    labels = ['O'] * len(tokens)
    
    for annotation in annotations:
        start, end = annotation['value']['start'], annotation['value']['end']
        label = annotation['value']['labels'][0]
        span_start, span_end = None, None
        current_pos = 0
        for i, token in enumerate(tokens):
            current_end = current_pos + len(token)
            if current_pos <= start < current_end:
                span_start = i
            if current_pos < end <= current_end:
                span_end = i
                break
            current_pos += len(token) + 1
        
        if span_start is not None and span_end is not None:
            labels[span_start] = 'B_' + label
            for i in range(span_start + 1, span_end + 1):
                labels[i] = 'I_' + label
    
    return {'text': text, 'labels': labels}

# process the dataset
def process_dataset(file_path):
    data = load_dataset(file_path)
    train_data, val_data = train_test_split(data, test_size=0.15, random_state=42)    
    return train_data, val_data

# save JSON files
def save_to_json(data, file_name):
    with open(file_name, 'w') as file:
        json.dump(data, file)

if __name__ == "__main__":
    file_path = "C:\\Users\\playf\\VScode\\NLP-A2\\NER\\NER_TRAIN_JUDGEMENT.json"
    train_data, val_data = process_dataset(file_path)
    test_data = load_dataset('C:\\Users\\playf\\VScode\\NLP-A2\\NER\\NER_TEST_JUDGEMENT.json')
    
    save_to_json(train_data, 'C:\\Users\\playf\\VScode\\NLP-A2\\NER\\NER_train_data.json')
    save_to_json(val_data, 'C:\\Users\\playf\\VScode\\NLP-A2\\NER\\NER_val_data.json')
    save_to_json(test_data, 'C:\\Users\\playf\\VScode\\NLP-A2\\NER\\NER_test_data.json')
    print("Data splits saved successfully.")
    
    train_data_bio = {case['id']: bio_encode(case) for case in train_data}
    val_data_bio = {case['id']: bio_encode(case) for case in val_data}
    test_data_bio = {case['id']: bio_encode(case) for case in test_data}
    
    save_to_json(train_data_bio, 'C:\\Users\\playf\\VScode\\NLP-A2\\NER\\NER_train.json')
    save_to_json(val_data_bio, 'C:\\Users\\playf\\VScode\\NLP-A2\\NER\\NER_val.json')
    save_to_json(test_data_bio, 'C:\\Users\\playf\\VScode\\NLP-A2\\NER\\NER_test.json')
    print("BIO encodings saved successfully.")
