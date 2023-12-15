import numpy as np
import os
from PIL import Image
from keras.utils import to_categorical
from sklearn.model_selection import train_test_split
from keras import layers
from keras import Sequential
from keras.layers import Dense
from sklearn.utils import shuffle
import cv2
imlist = os.listdir('./imgfolder_b')

image1 = np.array(Image.open('./imgfolder_b' + '/' + imlist[0]))  # open one image to get size
# plt.imshow(im1)

m, n = image1.shape[0:2]  # get the size of the images
total_images = len(imlist)  # get the 'total' number of images

# create matrix to store all flattened images
immatrix = np.array([np.array(Image.open('./imgfolder_b'+ '/' + images).convert('L')).flatten()
                     for images in sorted(imlist)], dtype='f')

print(immatrix.shape)

input("Press any key")

#########################################################
## Label the set of images per respective gesture type.
##
label = np.ones((total_images,), dtype=int)

samples_per_class = int(total_images / 5)
print("samples_per_class - ", samples_per_class)
s = 0
r = samples_per_class
for classIndex in range(5):
    label[s:r] = classIndex
    s = r
    r = s + samples_per_class

'''
# eg: For 301 img samples/gesture for 4 gesture types
label[0:301]=0
label[301:602]=1
label[602:903]=2
label[903:]=3
'''

data, Label = shuffle(immatrix, label, random_state=2)
train_data = [data, Label]

(X, y) = (train_data[0], train_data[1])

# Split X and y into training and testing sets

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=4)

X_train = X_train.reshape(X_train.shape[0], 200, 200, 1)
X_test = X_test.reshape(X_test.shape[0], 200, 200, 1)

X_train = X_train.astype('float32')
X_test = X_test.astype('float32')

# normalize
X_train /= 255
X_test /= 255

# convert class vectors to binary class matrices
Y_train = to_categorical(y_train, 5)
Y_test = to_categorical(y_test, 5)

dropout_rate=0.1
weight_constraint=0
weightactivation='relu'
optimizer='adam'
learn_rate=0.01
momentum=0
init_mode='uniform'

model = Sequential()

model.add(layers.Conv2D(32, (3, 3), strides=(2, 2), kernel_initializer=init_mode, activation=weightactivation, input_shape=(200, 200, 1)))
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
model.add(Dense(5, kernel_initializer=init_mode, activation='softmax'))
model.summary()




model.compile(optimizer='Adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])
model.fit(X_train, Y_train, epochs=10, batch_size=32, verbose=1, validation_split=0.2)

img = cv2.imread('punch.png', 0)
#img = cv2.flip(img, 0)
print('image shape 3', img.shape)

img = img.reshape(1, 200, 200)
pred=model.predict(img)
print(pred[0])
# if pred[0] == [1, 0, 0, 0, 0]:
#     print("the gesture is ok")
# elif pred[0] == [0, 1, 0, 0, 0]:
#     print("the gesture is nothing")
# elif pred[0] == [0, 0, 1, 0, 0]:
#     print("the gesture is peace")
# elif pred[0] == [0, 0, 0, 1, 0]:
#     print("the gesture is punch")
# elif pred[0] == [0, 0, 0, 0, 1]:
#     print("the gesture is stop")
