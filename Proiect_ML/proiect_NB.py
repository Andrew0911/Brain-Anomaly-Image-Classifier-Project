import os, sys
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
from sklearn.naive_bayes import MultinomialNB
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
        
        # convertim imaginea la grayscale si o transformam intr-un vector de dimensiune 1
        img_array = np.array(img.convert("L")).flatten()

        imagini_train.append(img_array)

    # transformam vectorul dorit in numpy.array
    imagini_train = np.array(imagini_train)

    return imagini_train

def citire_validation(folder, files):
    
    # urmatoarele 2000 de poze le salvam ca imagini_validation
    imagini_validation = []

    for i in range(15000, 17000):

        #citim fiecare imagine specificand si path-ul respectiv
        img = Image.open(folder + files[i])
        
        # convertim imaginea la grayscale si o transformam intr-un vector de dimensiune 1
        img_array = np.array(img.convert("L")).flatten()

        imagini_validation.append(img_array)

    # transformam vectorul dorit in numpy.array
    imagini_validation = np.array(imagini_validation)

    return imagini_validation

def citire_test(folder, files):

    # ultimele poze vor fi salvate ca imagini_test
    imagini_test = []

    for i in range(17000, 22149):

        # citim fiecare imagine specificand si path-ul respectiv
        img = Image.open(folder + files[i])
        
        # convertim imaginea la grayscale si o transformam intr-un vector de dimensiune 1
        img_array = np.array(img.convert("L")).flatten()

        imagini_test.append(img_array)

    # transformam vectorul dorit in numpy.array
    imagini_test = np.array(imagini_test)

    return imagini_test

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

def generare_intervale(min, max, num_intervale):

    return np.linspace(min, max, num_intervale)

def imparte_in_intervale(data, intervale):

    divided_data = np.zeros(data.shape)

    for i in range(len(data)) :
        divided_data[i] = np.digitize(data[i], intervale)
        divided_data[i] -= 1

    return divided_data

# citim toate datele utilizand functiile descrise mai sus

imagini_train = citire_train(folder=folder, files=files)

imagini_validation = citire_validation(folder=folder, files=files)

imagini_test = citire_test(folder=folder, files=files)

train_labels = np.array(citire_labels(folder_general + 'train_labels.txt', 1))

validation_labels = np.array(citire_labels(folder_general + 'validation_labels.txt', 1))

# generam intervale pentru a segmenta datele
intervale = generare_intervale(0, 224, 4)

# impartim datele dupa numarul de intervale generat

train_segmentat = imparte_in_intervale(imagini_train, intervale)

validation_segmentat = imparte_in_intervale(imagini_validation, intervale)

test_segmentat = imparte_in_intervale(imagini_test, intervale)

# adaugam si datele de validare pentru antrenare impreuna cu label-urile respective

train_segmentat = np.concatenate((train_segmentat, validation_segmentat), axis = 0)

train_labels = np.concatenate((train_labels, validation_labels), axis = 0)

# definim modelul nostru
NB_model = MultinomialNB()

# antrenam modelul
NB_model.fit(train_segmentat, train_labels)
        
# testam accuracy-ul
print(NB_model.score(validation_segmentat, validation_labels))

# f1-score
predicted_validation = NB_model.predict(validation_segmentat)
print(f1_score(predicted_validation, validation_labels))

# prezicem etichetele corespunzatoare setului de date pentru testare
test_labels = NB_model.predict(test_segmentat)

# crearea fisierului .csv dupa formatul cerut
output_formatat = []

for i in range(0, 5149):
    output_formatat.append('0' + str(i + 17001) + ',' + str(test_labels[i]))
    
np.savetxt('submission.csv', output_formatat, "%s", header="id,class", comments="")
