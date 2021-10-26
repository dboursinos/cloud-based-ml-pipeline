from PIL import Image
from sklearn.model_selection import train_test_split
import pickle
import csv
import numpy as np
from collections import Counter
from minio import Minio
import io
import os

minioClient = Minio(os.environ.get('MLFLOW_S3_ENDPOINT_URL').split('//')[1],
                  access_key=os.environ.get('AWS_ACCESS_KEY_ID'),
                  secret_key=os.environ.get('AWS_SECRET_ACCESS_KEY'),
                  secure=False)

train_images = [] # images
train_labels = [] # corresponding labels
dataset_path="../Dataset/GTSRB/Training"
input_side_size=30

# loop over all 43 classes
for c in range(43):
    prefix = dataset_path + '/' + format(c, '05d') + '/' # subdirectory for class
    gtFile = open(prefix + 'GT-'+ format(c, '05d') + '.csv') # annotations file
    gtReader = csv.reader(gtFile, delimiter=';') # csv parser for annotations file
    next(gtReader) # skip header
    # loop over all images in current annotations file
    for row in gtReader:
        img=Image.open(prefix + row[0]) # the 1th column is the filename
        img=img.resize([input_side_size,input_side_size],Image.ANTIALIAS)
        img_rgb=np.array(img)
        train_images.append(img_rgb)
        train_labels.append(row[7]) # the 8th column is the label
    gtFile.close()

train_images=np.array(train_images,dtype='float32')/255.
train_labels=np.array(train_labels,dtype='int')

train_images, test_images, train_labels, test_labels = train_test_split(train_images, train_labels, test_size=0.1, random_state=42)
train_images, calibration_images, train_labels, calibration_labels = train_test_split(train_images, train_labels, test_size=0.2, random_state=42)

data={'x_train':train_images,
'y_train': train_labels,
'x_validation' : calibration_images,
'y_validation' : calibration_labels,
'x_test': test_images,
'y_test': test_labels}

p_data=pickle.dumps(data)
minioClient.put_object("dataset", "GeneratedData/preprocessed_data.pickle", data=io.BytesIO(p_data), length=len(p_data))
