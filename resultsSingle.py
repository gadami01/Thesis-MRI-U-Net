import cv2
import numpy as np
import os
import matplotlib.pyplot as plt

tp = 0
tn = 0
fp = 0
fn = 0

#pathDoctor = 'C:/Users/hriro/Desktop/MRIsplitted/doctor'
#pathPrediction = "preds/"

pathDoctor = "15_mask.tif"
pathPrediction = "15_pred.png"

ti = ""

#for ti in os.listdir(pathDoctor):
    #if "tif" in ti:
if True:
    if True:
        print(ti)
        doctorMask = cv2.imread(pathDoctor , cv2.IMREAD_GRAYSCALE)

        ti = ti[:-9]
        ti = ti.replace('_', '')
        ti = ti + "_pred.png"

        (t1, predictionMask) = cv2.threshold(cv2.imread(pathPrediction, cv2.IMREAD_GRAYSCALE), 100, 255, cv2.THRESH_BINARY)

        cv2.imwrite("newpred.png", predictionMask)

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


im0 = cv2.imread("15_mask.tif")
plt.imshow(im0)

plt.show()

im1 = cv2.imread("15_pred.png")
plt.imshow(im1)

plt.show()

im2 = cv2.imread("newpred.png")
plt.imshow(im2)

plt.show()
