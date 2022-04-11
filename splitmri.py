import os
import shutil
import xlsxwriter
import numpy as np

data = np.array(['Patient', 'Time', 'Original Name', 'New Name', 'Split'])

patient = "AA"
timestamp = "1"

testCounter = 0
trainingCounter = 0

for tj in os.listdir("C:/Users/hriro/Desktop/MRIdataset3/"):

    split = np.random.choice(2, 1, p=[0.2, 0.8])
    s = ""

    if split == 0:
        s = "Test"
    else:
        trainingCounter += 1
        s = "Training"

    patientCounter = 1

    for tq in range(1, 3):

        patient = str(tj)
        timestamp = str(tq)

        path = "C:/Users/hriro/Desktop/MRIdataset3/" + patient + "/" + timestamp

        for ti in os.listdir(path):

            if "mask" not in ti:
                original = str(ti)

                new = ""

                if split == 0:
                    new = str(testCounter) + ".tif"
                    testCounter += 1
                else:
                    new = str(trainingCounter) + "_" + str(patientCounter) + ".tif"

                data = np.vstack([data, [patient, timestamp, original, new, s]])
                data = np.vstack([data, [patient, timestamp, original[:-4] + "_mask.tif", new[:-4] + "_mask.tif", s]])
                patientCounter += 1
print(data)
print(data.shape)

workbook = xlsxwriter.Workbook("relations.xlsx")
worksheet = workbook.add_worksheet()

for i in range(0, data.shape[0]):
    worksheet.write(i, 0, data[i][0])
    worksheet.write(i, 1, data[i][1])
    worksheet.write(i, 2, data[i][2])
    worksheet.write(i, 3, data[i][3])
    worksheet.write(i, 4, data[i][4])

    if i != 0:
        if data[i][4] == "Training":
            src = "C:/Users/hriro/Desktop/MRIdataset3/" + data[i][0] + "/" + data[i][1] + "/" + data[i][2]
            dst = "C:/Users/hriro/Desktop/MRIsplitted3/train/" + data[i][3]
            shutil.copyfile(src, dst)
        else:
            src = "C:/Users/hriro/Desktop/MRIdataset3/" + data[i][0] + "/" + data[i][1] + "/" + data[i][2]
            dst = "C:/Users/hriro/Desktop/MRIsplitted3/test/" + data[i][3]
            shutil.copyfile(src, dst)

workbook.close()
