import numpy as np
from keras.preprocessing.text import Tokenizer
from keras.models import Sequential
from keras.layers import Dense, Dropout, Conv2D, MaxPooling2D, LSTM, Flatten, Embedding, RepeatVector, TimeDistributed, ZeroPadding2D, Concatenate, Activation
from keras.layers.normalization import BatchNormalization
# from keras.utils import plot_model
# from keras.callbacks import ModelCheckpoint
from utils import *

# Generate input/output data
images, captions = load_data('data/x_data.npy', 'data/y_data.npy')
tokenizer = generate_tokenizer(captions)
max_caption_length = max(len(line.split()) for line in captions)
vocab_size = len(tokenizer.word_index) + 1
encoded_captions = encode_sequences(tokenizer, max_caption_length, captions)
one_hot_captions = one_hot_encode(encoded_captions, vocab_size)

def get_image_model():
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

def get_language_model(vocab_size):
    language_model = Sequential()
    language_model.add(Embedding(vocab_size, 256, input_length = max_caption_length))
    language_model.add(LSTM(output_dim = 128, return_sequences = True))
    language_model.add(TimeDistributed(Dense(128)))
    return language_model

image_model = get_image_model()

image_model.layers.pop()
for layer in image_model.layers:
    layer.trainable = False

language_model = get_language_model(vocab_size)

image_model.add(RepeatVector(max_caption_length))

model = Sequential()
model.add(Concatenate([image_model, language_model]))
model.add(LSTM(256, return_sequences = False))
model.add(Dense(vocab_size))
model.add(Activation('softmax'))

model.compile(loss = 'categorical_crossentropy', optimizer = 'rmsprop')

print("encoded_captions.shape: ", encoded_captions.shape)
print("one_hot_captions.shape: ", one_hot_captions.shape)

model.fit([images, encoded_captions], one_hot_captions, batch_size = 1, epochs = 5)
model.save_weights('image_caption_weights.h5')

# model = get_image_model(len(tokenizer.word_index) + 1, 41) # hardcoding max_y for now
# model.compile(optimizer='adam', loss='categorical_crossentropy')
# print(model.summary())

# model.fit(x, y, epochs=30, batch_size=64, verbose=2)
# filename = 'model.h5'
# checkpoint = ModelCheckpoint(filename, monitor='val_loss', verbose=1, save_best_only=True, mode='min')
# model.fit(train_x, train_y, epochs=30, batch_size=64, validation_data=(test_x, test_y), callbacks=[checkpoint], verbose=2)



