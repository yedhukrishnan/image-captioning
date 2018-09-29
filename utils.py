from keras.utils import to_categorical
from keras.preprocessing.sequence import pad_sequences
from keras.preprocessing.text import Tokenizer
import numpy as np

# Function to one-hot encode the output captions
def one_hot_encode(data, vocab_size):
    ylist = list()
    for row in data:
        encoded = to_categorical(row, num_classes = vocab_size)
        ylist.append(encoded)
    y = np.array(ylist)
    y = y.reshape(data.shape[0], data.shape[1], vocab_size)
    return y

def load_data(images_path, captions_path):
    images = np.load(images_path)
    captions = np.load(captions_path)
    return (images, captions)

def generate_tokenizer(captions):
    tokenizer = Tokenizer()
    tokenizer.fit_on_texts(captions)
    return tokenizer

def encode_sequences(tokenizer, max_caption_length, lines):
    X = tokenizer.texts_to_sequences(lines)
    X = pad_sequences(X, maxlen = max_caption_length, padding = 'post')
    return X