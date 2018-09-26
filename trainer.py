import numpy as np
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.utils import to_categorical

# Load captions data
caption_dataset = np.load('data/y_data.npy')

# Generate tokens
tokenizer = Tokenizer()
tokenizer.fit_on_texts(caption_dataset)
# print(tokenizer.word_index)

# Maximum caption length
max_length = max(len(line.split()) for line in caption_dataset)
# print(max_length)

# Encode sequences
y = tokenizer.texts_to_sequences(caption_dataset)
y = pad_sequences(y, maxlen = max_length, padding = 'post')

# Function to one-hot encode the output captions
def one_hot_encode(data, vocab_size):
    ylist = list()
    for row in data:
        encoded = to_categorical(row, num_classes = vocab_size)
        ylist.append(encoded)
    y = np.array(ylist)
    y = y.reshape(data.shape[0], data.shape[1], vocab_size)
    return y

# One hot encode the captions
y = one_hot_encode(y, len(tokenizer.word_index) + 1)

# print(y)
