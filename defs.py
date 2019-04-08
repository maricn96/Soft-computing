import math
import numpy as np
import cv2
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation
from keras.utils import np_utils
import matplotlib.pylab as pylab
import matplotlib.pyplot as plt

def promena_velicine_regiona(region):
    return cv2.resize(region, (28, 28), interpolation = cv2.INTER_NEAREST)

def udaljenost_2_tacke(x1, y1, x2, y2): #X
    p1 = [x1, y1]
    p2 = [x2, y2]
    return math.sqrt(((p1[0]-p2[0])**2)+((p1[1]-p2[1])**2))

def povrsina_visina(a, b, c):
    par = (a + b + c) / 2
    povr = 2 * math.sqrt(par * (par - a) * (par - b) * (par - c)) / a
    return povr

def jel_na_liniji(x, y, x1, y1, x2, y2, ): #X
    if (y2 < y and y < y1 and x2 > x and x > x1):
        return 1

    return 0

def konture(slika):  # vraca koordinate svih kontura na slici
    im, konture, hijerarhija = cv2.findContours(slika.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    koordinate = []
    regions_array = []

    for kontura in konture:
        (x, y, w, h) = cv2.boundingRect(kontura)
        area = cv2.contourArea(kontura)

        if (h > 11 and area > 25 and area < 900 and w > 9):
            koord_jednog_broja = (x, y, w, h)
            koordinate.append(koord_jednog_broja)
            region = slika[y:y + h + 1, x:x + w + 1]
            regions_array.append([promena_velicine_regiona(region), (x, y, w, h)])
            cv2.rectangle(slika, (x, y), (x + w, y + h), (0, 255, 0), 2)

    return koordinate


def izdvoj_cifre(slika): #maska koju propustamo da bi uklonili liniju i sum
    low_white = np.array([180, 180, 180])
    up_white = np.array([240, 240, 240])
    maska = cv2.inRange(slika, low_white, up_white)

    kernel = (5, 5)
    brojevi = cv2.bitwise_and(slika, slika, mask=maska)

    gaus = cv2.GaussianBlur(cv2.cvtColor(brojevi, cv2.COLOR_BGR2GRAY), kernel, 0)

    return gaus


def prati_broj(brojeva_proslo, frejm, x, y):
    prosao = 0
    if ((len(brojeva_proslo)) > 0):
        for broj in brojeva_proslo:
            if (broj[0] == x):
                if(broj[1] == y):#ako se poklapa x ili y koord vec je prosao
                    prosao = 1

            if (prosao == 0):
                if udaljenost_2_tacke(x, y, broj[0], broj[1]) < 20:
                    if ((frejm - broj[2]) < 20):
                        prosao = 1

    if (prosao == 0):
        brojeva_proslo.append((x, y, frejm)) #ubacivanje u listu zajedno sa brojem frejma
        return 1

    return 0


def neuronska_kreiranje(): #kreiramo mrezu sa slojevima

    n_input = 784  # input layer (28x28 pixels)
    n_hidden1 = 512  # 1st hidden layer
    n_hidden2 = 256  # 2nd hidden layer
    n_hidden3 = 128  # 3rd hidden layer

    mreza = Sequential()
    mreza.add(Dense(n_hidden1, input_shape=(n_input,)))
    mreza.add(Activation('relu'))
    mreza.add(Dropout(0.2))

    mreza.add(Dense(n_hidden1))
    mreza.add(Activation('relu'))
    mreza.add(Dropout(0.2))

    mreza.add(Dense(n_hidden2))
    mreza.add(Activation('relu'))
    mreza.add(Dropout(0.2))

    mreza.add(Dense(n_hidden3))
    mreza.add(Activation('relu'))
    mreza.add(Dropout(0.2))

    mreza.add(Dense(10, activation='softmax'))

    return mreza


def treniranje_mreze(mreza, x_trening, y_trening, x_test, y_test):
    y_tren = np_utils.to_categorical(y_trening, 10) #10 -> broj klasa
    y_test = np_utils.to_categorical(y_test, 10)

    mreza.compile(loss='categorical_crossentropy', optimizer='sgd', metrics=['accuracy'])

    mreza.fit(x_trening, y_tren, epochs=10, batch_size=256, verbose=1, validation_data=(x_test, y_test))

    return mreza


def za_neuronsku(izvuceni_brojevi):

    pripremljeni_brojevi = []

    for broj in izvuceni_brojevi:
        pripremljeni_brojevi.append((broj/255).flatten())

    pripremljeni_brojevi = np.array(pripremljeni_brojevi, np.float32)

    return pripremljeni_brojevi