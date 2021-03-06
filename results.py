import cv2
import numpy as np
import os

tp = 0
tn = 0
fp = 0
fn = 0

path = 'C:/Users/hriro/Desktop/MRIsplitted2/doctor'
for ti in os.listdir(path):
    if "tif" in ti:
        print(ti)
        doctorMask = cv2.imread(path + "/" + ti, cv2.IMREAD_GRAYSCALE)

        ti = ti[:-9]
        ti = ti.replace('_', '')
        ti = ti + "_pred.png"

        (t1, predictionMask) = cv2.threshold(cv2.imread("tifonlypreds/" + ti, cv2.IMREAD_GRAYSCALE), 100, 255, cv2.THRESH_BINARY)

        #doctorMask = cv2.resize(doctorMask, (128, 128))

        predictionMask = np.array(predictionMask)
        predictionMask = np.where(predictionMask > 0, 1, predictionMask)

        doctorMask = np.array(doctorMask)
        doctorMask = np.where(doctorMask > 0, 1, doctorMask)

        compare = np.subtract(predictionMask, doctorMask)

        y, x = predictionMask.shape

        for i in range(0, y):
            for j in range(0, x):
                if compare[i, j] == 0:
                    if predictionMask[i, j] == 0:
                        tn += 1
                    if predictionMask[i, j] == 1:
                        tp += 1
                elif compare[i, j] == 1:
                    fp += 1
                else:
                    fn += 1

sum = tp + tn + fp + fn
print("True Positive Pixels : " + str(tp) + " (" + str("{:.2f}".format(tp / sum * 100)) + "%)")
print("True Negative Pixels : " + str(tn) + " (" + str("{:.2f}".format(tn / sum * 100)) + "%)")
print("False Positive Pixels : " + str(fp) + " (" + str("{:.2f}".format(fp / sum * 100)) + "%)")
print("False Negative Pixels : " + str(fn) + " (" + str("{:.2f}".format(fn / sum * 100)) + "%)")
print("DSC : " + str("{:.2f}".format(2*tp/(2*tp + fp + fn))))
