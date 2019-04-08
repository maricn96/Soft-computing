import keras
import defs
import cv2
import numpy as np


def invertuj(slika):
    slika = 255 - slika

    slika2 = cv2.erode(slika, (3, 3), iterations=1)

    sl = cv2.dilate(slika2, (3, 3), iterations=1)

    _, prag = cv2.threshold(sl, 128, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)

    return prag

(trening_skup, tr_oznake), (test_skup, test_oznake) = keras.datasets.mnist.load_data()


trening = []
test = []
for i in range(len(trening_skup)):
    trening.append(invertuj(trening_skup[i]))

for i in range(len(test_skup)):
    test.append(invertuj(test_skup[i]))

inputs_train = defs.za_neuronsku(trening)

inputs_test = defs.za_neuronsku(test)

mreza_kreirana = defs.neuronska_kreiranje()

mreza_kreirana = defs.treniranje_mreze(mreza_kreirana, inputs_train, tr_oznake, inputs_test, test_oznake)


mreza_kreirana.save('n_mreza.h5')