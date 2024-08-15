import json
from sklearn.preprocessing import OneHotEncoder
import numpy as np

def read_json(path):
    with open(path, 'r', encoding='utf-8') as file:
        json_data = json.load(file)
        return json_data

def encode_words(words_list, word_to_index):
    vector = np.zeros(len(word_to_index))
    for word in words_list:
        if word in word_to_index:
            vector[word_to_index[word]] = 1
    return vector

def encode_json_data(path):
    json_data = read_json(path)
    data = json.loads(json_data)
    all_words = set()
    for item in data:
        all_words.update(item["words"])
    word_to_index = {word: i for i, word in enumerate(all_words)}
    keys = set()
    for item in data:
        keys.update(item.keys())
    keys.discard('imagerys_words')
    keys.discard('iid')
    key_values = {key: [item[key] for item in data] for key in keys}
    encoders = {key: OneHotEncoder(sparse=False).fit(np.array(values).reshape(-1, 1)) for key, values in key_values.items()}
    encoded_data = {}
    for item in data:
        encoded_vectors = []
        for key in keys:
            value = item[key]
            encoded_value = encoders[key].transform(np.array([[value]]))
            encoded_vectors.append(encoded_value.flatten())
        words_vec = encode_words(item["imagery_words"], word_to_index)
        full_vector = np.concatenate(encoded_vectors + [words_vec])
        encoded_data[item["iid"]] = full_vector

    return encoded_data



