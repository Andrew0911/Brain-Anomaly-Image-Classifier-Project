import tensorflow as tf
import os, sys, cv2
from PIL import Image
import numpy as np
from tensorflow import keras
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Dense, Flatten, Lambda, BatchNormalization
from sklearn.metrics import f1_score

# path-ul folderului cu imaginile .png
folder = 'D:/Facultate/ANUL 2/SEMESTRUL 2/IA/ML/Proiect IA/data/data/'

# path-ul folderului care contine toate datele
folder_general = 'D:/Facultate/ANUL 2/SEMESTRUL 2/IA/ML/Proiect IA/data/'

# extragem numele imaginilor si le retinem intr-un vector
files = [f for f in os.listdir(folder)]

def citire_train(folder, files):

    # primele 15000 de poze le salvam ca imagini_train
    imagini_train = []

    for i in range(0, 15000):

        #citim fiecare imagine specificand si path-ul respectiv
        img = Image.open(folder + files[i])
        
        # dorim vectori de dimensiune 3
        imagini_train.append(np.array(img))

    return np.array(imagini_train)

def citire_validation(folder, files):
    
    # urmatoarele 2000 de poze le salvam ca imagini_validation
    imagini_validation = []

    for i in range(15000, 17000):

        #citim fiecare imagine specificand si path-ul respectiv
        img = Image.open(folder + files[i])
        
        # dorim vectori de dimensiune 3
        imagini_validation.append(np.array(img))

    return np.array(imagini_validation)

def citire_test(folder, files):

    # ultimele poze vor fi salvate ca imagini_test
    imagini_test = []

    for i in range(17000, 22149):

        # citim fiecare imagine specificand si path-ul respectiv
        img = Image.open(folder + files[i])
        
         # dorim vectori de dimensiune 3
        imagini_test.append(np.array(img))

    return np.array(imagini_test)

def citire_labels(folder, num):

    labels = []
    multiplied_labels = []

    f = open(folder)
    f.readline()

    labels = [int(line.split(',')[-1]) for line in f.readlines()]

    if num == 1 :
        return labels
    elif num == 2 :
        for i in labels:
            multiplied_labels.extend([i, i])
    elif num == 3 :
        for i in labels:
            multiplied_labels.extend([i, i, i])
    elif num == 4 :
        for i in labels:
            multiplied_labels.extend([i, i, i, i])

    return multiplied_labels


# citim toate datele utilizand functiile descrise mai sus

imagini_train = citire_train(folder=folder, files=files)

imagini_validation = citire_validation(folder=folder, files=files)

imagini_test = citire_test(folder=folder, files=files)

train_labels = np.array(citire_labels(folder_general + 'train_labels.txt', 1))

validation_labels = np.array(citire_labels(folder_general + 'validation_labels.txt', 1))


# definim modelul

CNN_model = Sequential()
    
CNN_model.add(Conv2D(16, (3,3), activation = 'relu', input_shape = (224, 224, 3)))
CNN_model.add(BatchNormalization())
    
CNN_model.add(Lambda(lambda x : x / 255))
    
CNN_model.add(MaxPooling2D((2, 2)))
    
CNN_model.add(Conv2D(32, (3,3), activation = 'relu'))
CNN_model.add(BatchNormalization())
   
CNN_model.add(MaxPooling2D((2, 2)))
    
CNN_model.add(Conv2D(64, (3,3), activation = 'relu'))
CNN_model.add(BatchNormalization())
   
CNN_model.add(MaxPooling2D((2, 2)))
    
CNN_model.add(Conv2D(128, (3,3), activation = 'relu'))
CNN_model.add(BatchNormalization())
   
CNN_model.add(MaxPooling2D((2, 2)))
    
CNN_model.add(Flatten())
CNN_model.add(Dense(256, activation = 'relu'))
CNN_model.add(Dense(128, activation = 'relu'))
CNN_model.add(Dense(64, activation = 'relu'))
CNN_model.add(Dense(32, activation = 'relu'))
CNN_model.add(Dense(1, activation = 'sigmoid'))

# compilam modelul

CNN_model.compile(optimizer = 'adam', loss = tf.losses.BinaryCrossentropy(), metrics = ['accuracy'])

# afisam sumarul

CNN_model.summary()

# antrenam modelul in 20 de epoci, in batch-uri de 64

CNN_model.fit(imagini_train, train_labels, epochs=20, batch_size = 64)

# Testam acuratetea si pierderea

rezultat = CNN_model.evaluate(imagini_validation, validation_labels)
print('Loss:', rezultat[0])
print('Accuracy:', rezultat[1])

# f1-score

predicted_validation = CNN_model.predict(imagini_validation)
print(f1_score(predicted_validation, validation_labels))

labels = CNN_model.predict(imagini_test)
test_labels = []

# structuram label-urile corect

for label in labels:
    if(label > 0.5):
        test_labels.append(1)
    else:
        test_labels.append(0)
        
test_labels = np.array(test_labels)
        
# crearea fisierului .csv dupa formatul cerut
output_formatat = []

for i in range(0, 5149):
    output_formatat.append('0' + str(i + 17001) + ',' + str(test_labels[i]))
    
np.savetxt('submission.csv', output_formatat, "%s", header="id,class", comments="")
