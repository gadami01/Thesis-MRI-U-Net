from scipy.io import loadmat
import os
import cv2
import numpy as np
from PIL import Image, ImageOps
from skimage import draw
from scipy import interpolate

patient = "AA"
timestamp = "1"

for tj in os.listdir("C:/Users/hriro/Desktop/Initial & repeat MRI in MS-Free Dataset/"):
    for tq in range(1, 3):
        patient = str(tj)
        timestamp = str(tq)
        path = "C:/Users/hriro/Desktop/Initial & repeat MRI in MS-Free Dataset/" + patient + "/" + timestamp

        os.makedirs("C:/Users/hriro/Desktop/MRIdataset3/" + patient + "/" + timestamp)

        for ti in os.listdir(path):
            filename = str(ti)

            #or filename.endswith(".TIF") or
            if filename.endswith("bmp"):
                print(filename)
                image = cv2.imread(path + "/" + filename)
                imagegray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

                maskimg = np.zeros(imagegray.shape)
                # if filename.endswith(".TIF"):
                #     maskimg = np.zeros((512, 512))
                # else:
                #     maskimg = np.zeros((378, 378))

                masknumber = 1

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

                    t = np.arange(len(xpoints))
                    ti = np.linspace(0, t.max(), 10 * t.size)

                    xpoints = interpolate.interp1d(t, xpoints, kind='cubic')(ti)
                    ypoints = interpolate.interp1d(t, ypoints, kind='cubic')(ti)

                    pr, pc = draw.polygon(ypoints, xpoints)
                    maskimg[pr, pc] = 255

                    masknumber += 1

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

                    t = np.arange(len(xpoints))
                    ti = np.linspace(0, t.max(), 10 * t.size)

                    xpoints = interpolate.interp1d(t, xpoints, kind='cubic')(ti)
                    ypoints = interpolate.interp1d(t, ypoints, kind='cubic')(ti)

                    pr, pc = draw.polygon(ypoints, xpoints)
                    maskimg[pr, pc] = 255

                maskimg = np.array(maskimg)
                mimage = Image.fromarray(np.uint8(maskimg), mode="L")
                mimage = mimage.resize((256, 256))
                mimage = ImageOps.grayscale(mimage)
                mimage.save("C:/Users/hriro/Desktop/MRIdataset3/" + patient + "/" + timestamp + "/" + filename[:-4] +
                            "_mask.tif")

                imagegray = np.array(imagegray)
                gimage = Image.fromarray(np.uint8(imagegray), mode="L")
                gimage = gimage.resize((256, 256))
                gimage = ImageOps.grayscale(gimage)
                gimage.save("C:/Users/hriro/Desktop/MRIdataset3/" + patient + "/" + timestamp + "/" + filename[:-4] +
                            ".tif")
