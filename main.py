import cv2
from keras import models
import numpy as np
import defs
import ispis
from array import array
import matplotlib.pyplot as plt

brojac = 0

suma = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]

mreza_ucitaj = models.load_model('n_mreza.h5')
for brojac in range(0, 10):

    vc = cv2.VideoCapture('assets_za_910/video-' + str(brojac) + '.avi')

    prvi_frejm = vc.read()[1]
    frejm_to_rgb = cv2.cvtColor(prvi_frejm, cv2.COLOR_BGR2RGB)

    frejm_za_liniju = cv2.cvtColor(frejm_to_rgb, cv2.COLOR_RGB2HSV)

    # DETEKCIJA PLAVE LINIJE
    low_blue = np.array([93, 81, 3])
    up_blue = np.array([125, 255, 255])
    mask = cv2.inRange(frejm_za_liniju, low_blue, up_blue)

    can = cv2.Canny(mask, 75, 150)
    plt.imshow(can)
    plt.show()

    maks_distanca = 0
    retval_za_plavu = []

    lines = cv2.HoughLinesP(can, 1, np.pi/180, 90, 90, 10)  # obican houghlines radi drugacije i mnogo je kompleksniji za procesor

    for line in lines:
        for x1, y1, x2, y2 in line:
            udaljenost = defs.udaljenost_2_tacke(x1, y1, x2, y2)
            if (maks_distanca < udaljenost):
                maks_distanca = udaljenost
                x1_zbir = x1
                y1_zbir = y1
                x2_zbir = x2
                y2_zbir = y2

    # DETEKCIJA ZELENE LINIJE
    low_green = np.array([24, 51, 71])
    up_green = np.array([101, 255, 255])
    mask2 = cv2.inRange(frejm_za_liniju, low_green, up_green)
    can2 = cv2.Canny(mask2, 75, 150)
    maks_distanca_g = 0
    retval_za_zelenu = []

    lines_g = cv2.HoughLinesP(can2, 1, np.pi / 180, 90, 90, 10)  # obican houghlines radi drugacije i mnogo je kompleksniji za procesor

    for linee in lines_g:
        for x11, y11, x22, y22 in linee:
            udaljenostt = defs.udaljenost_2_tacke(x11, y11, x22, y22)
            if (maks_distanca_g < udaljenostt):
                maks_distanca_g = udaljenostt
                x1_razlika = x11
                y1_razlika = y11
                x2_razlika = x22
                y2_razlika = y22
    # print(retval_za_zelenu)

    ###################################################3
    print('Video-' + str(brojac) + '.avi.')

    brojeva_proslo = []
    broj_frejmova = 0

    while vc.isOpened():
        success, frejm = vc.read()
        broj_frejmova += 1

        if success is not True:
            break

        cv2.line(frejm, (x1_zbir, y1_zbir), (x2_zbir, y2_zbir), (0, 0, 255), 2)
        cv2.line(frejm, (x1_razlika, y1_razlika), (x2_razlika, y2_razlika), (0, 0, 255), 2)

        izdvojene_cifre = defs.izdvoj_cifre(frejm)
        lista_koordinata = defs.broj_kontura(izdvojene_cifre)

        for koordinata in lista_koordinata:
            (x, y, w, h) = koordinata
            cv2.rectangle(frejm, (x, y), (x + w, y + h), (0, 255, 0), 1)

            koord_zbir = (defs.udaljenost_2_tacke(x1_zbir, y1_zbir, x2_zbir, y2_zbir), defs.udaljenost_2_tacke(x1_zbir, y1_zbir, x, y), defs.udaljenost_2_tacke(x2_zbir, y2_zbir, x, y))
            visina_zbir = defs.povrsina_visina(koord_zbir[0], koord_zbir[1], koord_zbir[2])

            koord_razlika = (defs.udaljenost_2_tacke(x1_razlika, y1_razlika, x2_razlika, y2_razlika), defs.udaljenost_2_tacke(x1_razlika, y1_razlika, x, y), defs.udaljenost_2_tacke(x2_razlika, y2_razlika, x, y))
            visina_razlika = defs.povrsina_visina(koord_razlika[0], koord_razlika[1], koord_razlika[2])

            cifra = defs.uzmi_broj(izdvojene_cifre, x, y, h, w) #slicica koju prima neuronska
            #plt.imshow(cifra) -> dobro uzima
            #plt.show()
            cifra = 255 - cifra

            _, cifra = cv2.threshold(cifra, 128, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU) #smanjuje zamucenje


            #print(h_zbir)
            if defs.jel_na_liniji(x1_zbir, y1_zbir, x2_zbir, y2_zbir, x, y) == 1:
                if visina_zbir < 4:  #otprilike na 4 je broj kod linije
                    if defs.track_number(x, y, brojeva_proslo, broj_frejmova):
                        number = defs.predict_tracked_number(cifra, mreza_ucitaj)
                        suma[brojac] = suma[brojac] + number
            if defs.jel_na_liniji(x1_razlika, y1_razlika, x2_razlika, y2_razlika, x, y) == 1:
                if visina_razlika < 4:  #otprilike na 4 je broj kod linije
                    if defs.track_number(x, y, brojeva_proslo, broj_frejmova):
                        number = defs.predict_tracked_number(cifra, mreza_ucitaj)
                        suma[brojac] = suma[brojac] - number

        defs.cv2.putText(frejm, str(suma[brojac]), (15, 70), cv2.FONT_HERSHEY_DUPLEX, 2, (0, 0, 255), 3)
        cv2.imshow("video-" + str(brojac), frejm)
        if cv2.waitKey(1) & 0xFF == ord('e'):
            break

    cv2.destroyAllWindows()
    vc.release()

ispis.unos_podataka(suma)
