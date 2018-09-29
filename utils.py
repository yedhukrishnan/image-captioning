from keras.utils import to_categorical
from keras.preprocessing.sequence import pad_sequences
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

def max_length(lines):
    return max(len(line.split()) for line in lines)

# encode and pad sequences
def encode_sequences(tokenizer, length, lines):
	# integer encode sequences
	X = tokenizer.texts_to_sequences(lines)
	# pad sequences with 0 values
	X = pad_sequences(X, maxlen=length, padding='post')
	return X