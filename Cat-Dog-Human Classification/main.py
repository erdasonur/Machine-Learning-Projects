import numpy as np
import cv2
import os
from tqdm import tqdm
import random
import pickle
from skimage.transform import resize

DATADIR = "./PetImages"
CATEGORIES = ["Dog", "Cat", "Human"]


training_data = []

def create_training_data():
    for category in CATEGORIES:  # do dogs , cats and Humans

        path = os.path.join(DATADIR, category)  # create path to dogs and cats
        class_num = CATEGORIES.index(category)  # get the classification  (0 or a 1). 0=dog 1=cat

        for img in tqdm(os.listdir(path)):  # iterate over each image per dogs and cats
            try:
                img_array = cv2.imread(os.path.join(path, img), cv2.IMREAD_GRAYSCALE)  # convert to array
                new_array = cv2.resize(img_array, (32, 32))  # resize to normalize data size
                training_data.append([new_array, class_num])  # add this to our training_data
            except Exception as e:
                pass
            #except Exception as e:  # in the interest in keeping the output clean...
             #   print("general exception", e, os.path.join(path, img))
            # except OSError as e:
            #    print("OSErrroBad img most likely", e, os.path.join(path,img))


create_training_data()

random.shuffle(training_data)

X = []
y = []

for features, label in training_data:
    X.append(features)
    y.append(label)

print(X[0].reshape(-1, 32, 32, 1))

X = np.array(X).reshape(-1, 32, 32, 1)

pickle_out = open("X.pickle", "wb")
pickle.dump(X, pickle_out)
pickle_out.close()

pickle_out = open("y.pickle","wb")
pickle.dump(y, pickle_out)
pickle_out.close()

pickle_in = open("X.pickle","rb")
X = pickle.load(pickle_in)

pickle_in = open("y.pickle","rb")
y = pickle.load(pickle_in)

X = X / 255
from keras.utils import to_categorical
y = to_categorical(y)
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Conv2D, MaxPooling2D
model = Sequential()

model.add(Conv2D(32, (5, 5), input_shape=(32, 32, 1)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Conv2D(32, (5, 5)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Flatten())  # this converts our 3D feature maps to 1D feature vectors

model.add(Dense(500))

model.add(Dense(3))
model.add(Activation('softmax'))

model.compile(loss='categorical_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])

model.fit(X, y, batch_size=32, epochs=10, validation_split=0.3)

my_image = cv2.imread('cat.jpg')

#Resize the image
my_image_resized = resize(my_image, (32, 32, 1))
prediction = model.predict(np.array([my_image_resized],))


number_to_class = ['dog', 'cat', 'human']
index = np.argsort(prediction[0, :])
print('Most likely class:', number_to_class[index[2]], 'probability', prediction[0, index[2]])
print('Second likely class:', number_to_class[index[1]], 'probability',prediction[0, index[1]])
print('Third likely class:', number_to_class[index[0]], 'probability',prediction[0, index[0]])
cv2.imshow("image",my_image)
cv2.waitKey(0)
cv2.destroyAllWindows()