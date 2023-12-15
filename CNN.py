import numpy as np
import os
from PIL import Image
from keras.utils import to_categorical
from sklearn.model_selection import train_test_split
from keras import layers
from keras import Sequential
from keras.layers import Dense




lookup = dict()
reverselookup = dict()
count = 0
for j in os.listdir('./leapgestrecog/leapGestRecog/00/'):
    if not j.startswith('.'): # If running this code locally, this is to
                              # ensure you aren't reading in hidden folders
        lookup[j] = count
        reverselookup[count] = j
        count = count + 1
lookup

x_data = []
y_data = []
datacount = 0
for i in range(0, 10): # Loop over the ten top-level folders
    for j in os.listdir('./leapgestrecog/leapGestRecog/0' + str(i) + '/'):
        if not j.startswith('.'): # Again avoid hidden folders
            count = 0 # To tally images of a given gesture
            for k in os.listdir('leapgestrecog/leapGestRecog/0' +
                                str(i) + '/' + j + '/'):
                                # Loop over the images
                img = Image.open('leapgestrecog/leapGestRecog/0' +
                                 str(i) + '/' + j + '/' + k).convert('L')
                                # Read in and convert to greyscale
                img = img.resize((320, 120))
                arr = np.array(img)
                x_data.append(arr)
                count = count + 1
            y_values = np.full((count, 1), lookup[j])
            y_data.append(y_values)
            datacount = datacount + count
x_data = np.array(x_data, dtype = 'float32')
y_data = np.array(y_data)
y_data = y_data.reshape(datacount, 1) # Reshape to be the correct size

# x_data = np.load('x_data.npy')
# y_data = np.load('y_data.npy')
y_data = to_categorical(y_data)

x_data = x_data.reshape((datacount, 120, 320, 1))
x_data /= 255
y_data.shape
x_data.shape

x_train,x_further,y_train,y_further = train_test_split(x_data,y_data,test_size = 0.2)
x_validate,x_test,y_validate,y_test = train_test_split(x_further,y_further,test_size = 0.5)


dropout_rate=0.1
weight_constraint=0
weightactivation='relu'
optimizer='adam'
learn_rate=0.01
momentum=0
init_mode='uniform'

model = Sequential()

model.add(layers.Conv2D(32, (3, 3), strides=(2, 2), kernel_initializer=init_mode, activation=weightactivation, input_shape=(120, 320,1)))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(32, (3, 3), activation='relu'))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(64, (2, 2), activation='relu'))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(64, (2, 2), activation='relu'))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Flatten())
model.add(layers.Dense(128))
model.add(layers.Dropout(dropout_rate))
model.add(Dense(10,kernel_initializer=init_mode, activation='softmax'))

model.summary()




model.compile(optimizer='Adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])
model.fit(x_train, y_train, epochs=10, batch_size=40, verbose=1, validation_data=(x_validate, y_validate))

[loss, acc] = model.evaluate(x_test,y_test,verbose=1)
print("Accuracy:" + str(acc))