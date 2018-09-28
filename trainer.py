import numpy as np
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.utils import to_categorical
from keras.models import Sequential
from keras.layers import Dense, Activation, Dropout, Conv2D, MaxPooling2D, LSTM, Flatten, Reshape, Embedding, RepeatVector, TimeDistributed
from keras.layers.normalization import BatchNormalization
from keras.utils import plot_model
from keras.callbacks import ModelCheckpoint

# Load captions data
caption_dataset = np.load('data/y_data.npy')
x = np.load('data/x_data.npy')

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

def get_caption_model(n_y, max_y):
    # print("n_x", n_x)
    # print("n_y", n_y)
    # print("max_x", max_x)
    # print("max_y", max_y)

    ## WORK IN PROGRESS ###
    model = Sequential()

    model.add(Conv2D(64, (3, 3), strides = 2, input_shape = (250, 250, 3)))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size = (2, 2), padding='valid'))

    model.add(Conv2D(32, (3, 3), strides = 2))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size = (2, 2), padding='valid'))

    model.add(Conv2D(32, (5, 5), strides = 1))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size = (2, 2), padding='valid'))

    model.add(Flatten())
    model.add(Dense(500))
    # model.add(Activation('relu'))

    
    model.add(Embedding(500, 500, input_length = 500, mask_zero = True))

    # model.add(Reshape((1, 500, 1)))
    model.add(LSTM(500))
    model.add(RepeatVector(max_y))
    model.add(LSTM(500, return_sequences = True))
    model.add(TimeDistributed(Dense(n_y, activation='softmax')))

    return model

print(x.shape)
print(y.shape)

def max_length(lines):
    return max(len(line.split()) for line in lines)


vocab_size = len(tokenizer.word_index) + 1
max_y = max_length(caption_dataset[0])


train_x = x[:500, :, :, :]
train_y = y[:500, :, :]

test_x = x[500:600, :, :, :]
test_y = y[500:600, :, :]

model = get_caption_model(len(tokenizer.word_index) + 1, 41) # hardcoding max_y for now
model.compile(optimizer='adam', loss='categorical_crossentropy')
print(model.summary())

model.fit(x, y, epochs=30, batch_size=64, verbose=2)
# filename = 'model.h5'
# checkpoint = ModelCheckpoint(filename, monitor='val_loss', verbose=1, save_best_only=True, mode='min')
# model.fit(train_x, train_y, epochs=30, batch_size=64, validation_data=(test_x, test_y), callbacks=[checkpoint], verbose=2)



