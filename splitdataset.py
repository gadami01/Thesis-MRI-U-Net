import os
import random
import shutil

path = "C:/Users/hriro/Desktop/raw/"

for ti in os.listdir(path + "train"):
    if "mask" not in ti:
        rand = random.randint(0, 10)
        if rand < 8:
            shutil.copyfile(path + "train/" + ti, path + "train2/" + ti)
            shutil.copyfile(path + "train/" + ti[:-4] + "_mask.tif", path + "train2/" + ti[:-4] + "_mask.tif")
        else:
            shutil.copyfile(path + "train/" + ti, path + "test2/" + ti)
            shutil.copyfile(path + "train/" + ti[:-4] + "_mask.tif", path + "doctor/" + ti[:-4] + "_mask.tif")
