import numpy as np
from keras.preprocessing.text import Tokenizer
from keras.models import Sequential
from keras.layers import Dense, Activation, Dropout, Conv2D, MaxPooling2D, LSTM, Flatten, Reshape, Embedding, RepeatVector, TimeDistributed, ZeroPadding2D, Concatenate
from keras.layers.normalization import BatchNormalization
from keras.utils import plot_model
from keras.callbacks import ModelCheckpoint
from utils import *

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

# One hot encode the captions
y = one_hot_encode(y, len(tokenizer.word_index) + 1)

# print(y)

def get_caption_model(n_y, max_y):
    model = Sequential()
    model.add(ZeroPadding2D((1,1),input_shape=(250,250,3)))
    model.add(Conv2D(64, 3, 3, activation='relu'))
    model.add(ZeroPadding2D((1,1)))
    model.add(Conv2D(64, 3, 3, activation='relu'))
    model.add(MaxPooling2D((2,2), strides=(2,2)))

    model.add(ZeroPadding2D((1,1)))
    model.add(Conv2D(128, 3, 3, activation='relu'))
    model.add(ZeroPadding2D((1,1)))
    model.add(Conv2D(128, 3, 3, activation='relu'))
    model.add(MaxPooling2D((2,2), strides=(2,2)))

    model.add(ZeroPadding2D((1,1)))
    model.add(Conv2D(256, 3, 3, activation='relu'))
    model.add(ZeroPadding2D((1,1)))
    model.add(Conv2D(256, 3, 3, activation='relu'))
    model.add(ZeroPadding2D((1,1)))
    model.add(Conv2D(256, 3, 3, activation='relu'))
    model.add(MaxPooling2D((2,2), strides=(2,2)))

    model.add(ZeroPadding2D((1,1)))
    model.add(Conv2D(512, 3, 3, activation='relu'))
    model.add(ZeroPadding2D((1,1)))
    model.add(Conv2D(512, 3, 3, activation='relu'))
    model.add(ZeroPadding2D((1,1)))
    model.add(Conv2D(512, 3, 3, activation='relu'))
    model.add(MaxPooling2D((2,2), strides=(2,2)))

    model.add(ZeroPadding2D((1,1)))
    model.add(Conv2D(512, 3, 3, activation='relu'))
    model.add(ZeroPadding2D((1,1)))
    model.add(Conv2D(512, 3, 3, activation='relu'))
    model.add(ZeroPadding2D((1,1)))
    model.add(Conv2D(512, 3, 3, activation='relu'))
    model.add(MaxPooling2D((2,2), strides=(2,2)))

    model.add(Flatten())
    model.add(Dense(4096, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(4096, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(1000, activation='softmax'))

    return model

print(x.shape)
print(y.shape)


vocab_size = len(tokenizer.word_index) + 1
max_caption_len = max_length(caption_dataset[0])

train_x = x[:500, :, :, :]
train_y = y[:500, :, :]

test_x = x[500:600, :, :, :]
test_y = y[500:600, :, :]

train_captions_x = encode_sequences(tokenizer, max_caption_len, caption_dataset[:500])

image_model = get_caption_model(0, 0)

image_model.layers.pop()
for layer in image_model.layers:
    layer.trainable = False

language_model = Sequential()
language_model.add(Embedding(vocab_size, 256, input_length=max_caption_len))
language_model.add(LSTM(output_dim=128, return_sequences=True))
language_model.add(TimeDistributed(Dense(128)))

# let's repeat the image vector to turn it into a sequence.
print("Repeat model loading")
image_model.add(RepeatVector(max_caption_len))
print("Repeat model loaded")
# the output of both models will be tensors of shape (samples, max_caption_len, 128).
# let's concatenate these 2 vector sequences.
print("Merging")
model = Sequential()
model.add(Concatenate([image_model, language_model]))
# let's encode this vector sequence into a single vector
model.add(LSTM(256, return_sequences=False))
# which will be used to compute a probability
# distribution over what the next word in the caption should be!
model.add(Dense(vocab_size))
model.add(Activation('softmax'))

model.compile(loss='categorical_crossentropy', optimizer='rmsprop')
print("Merged")

print("train_captions_x.shape: ", train_captions_x.shape)
print("train_y.shape: ", train_y.shape)

# print(model.summary())

model.fit([train_x, train_captions_x], train_y, batch_size=1, epochs=5)

model.save_weights('image_caption_weights.h5')

# model = get_caption_model(len(tokenizer.word_index) + 1, 41) # hardcoding max_y for now
# model.compile(optimizer='adam', loss='categorical_crossentropy')
# print(model.summary())

# model.fit(x, y, epochs=30, batch_size=64, verbose=2)
# filename = 'model.h5'
# checkpoint = ModelCheckpoint(filename, monitor='val_loss', verbose=1, save_best_only=True, mode='min')
# model.fit(train_x, train_y, epochs=30, batch_size=64, validation_data=(test_x, test_y), callbacks=[checkpoint], verbose=2)



