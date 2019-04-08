import os
import cv2

def unos_podataka(niz):
    print('\nRezultati sacuvani u out.txt')
    os.remove("out.txt")
    file = open('out.txt', 'w+')
    file.write('RA 27/2015 Nikola Maric \n')
    file.write('file sum')
    for index in range(0, 10):
        file.write('\n' + 'video-' + str(index) + '.avi' + '\t' + str(niz[index]))
    file.close()

def ispis_sume(frejm, suma):
    cv2.putText(frejm, suma, (15, 70), cv2.FONT_HERSHEY_DUPLEX, 2, (0, 0, 255), 3)