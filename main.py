import cv2
from keras import models
import numpy as np
import defs
import ispis
from array import array
import matplotlib.pyplot as plt

suma = [0] * 10
brojac = 0

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

    maks_distanca = 0
    retval_za_plavu = []

    lines = cv2.HoughLinesP(mask, 1, np.pi/180, 90, 90, 10)  # obican houghlines radi drugacije i mnogo je kompleksniji za procesor

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
    maks_distanca_g = 0
    retval_za_zelenu = []

    lines_g = cv2.HoughLinesP(mask2, 1, np.pi / 180, 90, 90, 10)  # obican houghlines radi drugacije i mnogo je kompleksniji za procesor

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

    ###################################################

    brojeva_proslo = []
    broj_frejmova = 0

    while vc.isOpened():
        broj_frejmova += 1 #za kontrolisanje brojeva koji prelaze liniju
        success, frejm = vc.read()

        if success is not True:
            break

        cv2.line(frejm, (x1_zbir, y1_zbir), (x2_zbir, y2_zbir), (0, 0, 255), 2)
        cv2.line(frejm, (x1_razlika, y1_razlika), (x2_razlika, y2_razlika), (0, 0, 255), 2)


        #izdvoj cifre
        low_white = np.array([180, 180, 180])
        up_white = np.array([240, 240, 240])
        maska_white = cv2.inRange(frejm, low_white, up_white)

        kernel = (5, 5)
        brojevi = cv2.bitwise_and(frejm, frejm, mask=maska_white)

        izdvojene_cifre = cv2.GaussianBlur(cv2.cvtColor(brojevi, cv2.COLOR_BGR2GRAY), kernel, 0)

        lista_koordinata = defs.konture(izdvojene_cifre)

        for koordinata in lista_koordinata:
            x, y, w, h = koordinata

            koord_zbir = (defs.udaljenost_2_tacke(x1_zbir, y1_zbir, x2_zbir, y2_zbir), defs.udaljenost_2_tacke(x1_zbir, y1_zbir, x, y), defs.udaljenost_2_tacke(x2_zbir, y2_zbir, x, y))
            visina_zbir = defs.povrsina_visina(koord_zbir[0], koord_zbir[1], koord_zbir[2])

            koord_razlika = (defs.udaljenost_2_tacke(x1_razlika, y1_razlika, x2_razlika, y2_razlika), defs.udaljenost_2_tacke(x1_razlika, y1_razlika, x, y), defs.udaljenost_2_tacke(x2_razlika, y2_razlika, x, y))
            visina_razlika = defs.povrsina_visina(koord_razlika[0], koord_razlika[1], koord_razlika[2])

            cv2.rectangle(frejm, (x, y), (x + w, y + h), (0, 255, 0), 1)
            #isecanje cifre unutar konture
            cifra = izdvojene_cifre[y:y + h, x:x + w]
            #resize na 28x28
            cifra = cv2.resize(cifra, (28, 28), interpolation=cv2.INTER_NEAREST)

            #plt.imshow(cifra) -> dobro uzima
            #plt.show()
            cifra = 255 - cifra

            _, cifra = cv2.threshold(cifra, 120, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU) #smanjuje zamucenje


            #print(visina_zbir)
            if (visina_zbir < 4):  # otprilike na 4 je broj malo ispod linije
                if (defs.jel_na_liniji(x, y, x1_zbir, y1_zbir, x2_zbir, y2_zbir) == 1):
                    if (defs.prati_broj(brojeva_proslo, broj_frejmova, x, y) == 1): #tek kad prodje registrujemo ga
                        niz = []
                        niz.append(cifra)
                        niz = defs.za_neuronsku(niz)

                        cifra_sa_neuronske = np.array(niz, np.float32)
                        broj_konacan = mreza_ucitaj.predict_classes(cifra_sa_neuronske)
                        suma[brojac] = suma[brojac] + broj_konacan[0]
            if (visina_razlika < 4):  # otprilike na 4 je broj malo ispod linije
                if (defs.jel_na_liniji(x, y, x1_razlika, y1_razlika, x2_razlika, y2_razlika) == 1):
                    if (defs.prati_broj(brojeva_proslo, broj_frejmova, x, y) == 1): #tek kad prodje registrujemo ga
                        niz = []
                        niz.append(cifra)
                        niz = defs.za_neuronsku(niz)
                        cifra_sa_neuronske = np.array(niz, np.float32)
                        broj_konacan = mreza_ucitaj.predict_classes(cifra_sa_neuronske)
                        suma[brojac] = suma[brojac] - broj_konacan[0]

        ispis.ispis_sume(frejm, str(suma[brojac]))
        cv2.imshow("video-" + str(brojac), frejm)
        if cv2.waitKey(1) & 0xFF == ord('e'):
            break

    cv2.destroyAllWindows()
    vc.release()

ispis.unos_podataka(suma)