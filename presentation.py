from scipy.io import loadmat
import matplotlib.pyplot as plt
import os
import cv2
import numpy as np
from skimage import draw
from PIL import Image
from scipy import interpolate

patient = "AA"
timestamp = "2"

path = "C:/Users/hriro/Desktop/Initial & repeat MRI in MS-Free Dataset/" + patient + "/" + timestamp
filename = "IM_00239.tif"

print("Patient : " + patient + "\nTimestamp : " + timestamp + "\nSlice : " + filename + "\n\nLesions Found: ")
image = cv2.imread(path + "/" + filename)
imagegray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

maskimg = np.zeros(imagegray.shape)

masknumber = 1

if os.path.exists(path + "/" + filename[:-4] + ".plq"):
    print(filename[:-4] + ".plq")
    lesion = loadmat(path + "/" + filename[:-4] + ".plq")

    xi = np.array(lesion.get('xi'))
    yi = np.array(lesion.get('yi'))

    xpoints = np.empty(xi.shape[0])
    ypoints = np.empty(yi.shape[0])
    for i in range(0, xi.shape[0]):
        xpoints[i] = int(np.round(xi[i]))
        ypoints[i] = int(np.round(yi[i]))
        imagegray[int(np.round(yi[i])), int(np.round(xi[i]))] = 255

    t = np.arange(len(xpoints))
    ti = np.linspace(0, t.max(), 10 * t.size)

    xpoints = interpolate.interp1d(t, xpoints, kind='cubic')(ti)
    ypoints = interpolate.interp1d(t, ypoints, kind='cubic')(ti)

    pr, pc = draw.polygon(ypoints, xpoints)
    maskimg[pr, pc] = 255

    row, col = draw.polygon_perimeter(ypoints, xpoints)
    image[row, col, 1] = 255

while os.path.exists(path + "/" + filename[:-4] + "_" + str(masknumber) + ".plq"):
    print(filename[:-4] + "_" + str(masknumber) + ".plq")
    lesion = loadmat(path + "/" + filename[:-4] + "_" + str(masknumber) + ".plq")

    xi = np.array(lesion.get('xi'))
    yi = np.array(lesion.get('yi'))

    xpoints = np.empty(xi.shape[0])
    ypoints = np.empty(yi.shape[0])
    for i in range(0, xi.shape[0]):
        xpoints[i] = int(np.round(xi[i]))
        ypoints[i] = int(np.round(yi[i]))
        imagegray[int(np.round(yi[i])), int(np.round(xi[i]))] = 255

    t = np.arange(len(xpoints))
    ti = np.linspace(0, t.max(), 10 * t.size)

    xpoints = interpolate.interp1d(t, xpoints, kind='cubic')(ti)
    ypoints = interpolate.interp1d(t, ypoints, kind='cubic')(ti)

    pr, pc = draw.polygon(ypoints, xpoints)
    maskimg[pr, pc] = 255

    masknumber += 1

    row, col = draw.polygon_perimeter(ypoints, xpoints)

    image[row, col, 1] = 255

maskimgresized = Image.fromarray(maskimg)
maskimgresized = maskimgresized.resize((128, 128))

imageresized = Image.fromarray(image)
imageresized = imageresized.resize((128, 128))

plt.imshow(imagegray, cmap='gray')

plt.show()

plt.imshow(image)

plt.show()

plt.imshow(maskimg, cmap='gray')

plt.show()

plt.imshow(imageresized)

plt.show()

plt.imshow(maskimgresized, cmap='gray')

plt.show()
