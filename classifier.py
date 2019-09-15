import os, cv2, itertools # cv2 -- OpenCV
import numpy as np 
import pandas as pd 
 
import matplotlib.pyplot as plt
'exec(%matplotlib inline)'

TEST_DIR = './negative_images/'
TRAIN_DIR = './plate_number/'
ROWS = 8
COLS = 8
CHANNELS = 3

train_images = [TRAIN_DIR+i for i in os.listdir(TRAIN_DIR)]
test_images = [TEST_DIR+i for i in os.listdir(TEST_DIR)]

file_path = "C:\\Users\\LEKE-ARIYO\\Documents\\HNG\\Plate-Number-Classification"
def read_image(file_path):
  img = cv2.imread(file_path, cv2.IMREAD_COLOR)
  return cv2.resize(img, (ROWS, COLS),interpolation=cv2.INTER_CUBIC)


def prep_data(images):
  m = len(images)
  n_x = ROWS*COLS*CHANNELS
  
  X = np.ndarray((n_x,m), dtype=np.uint8)
  y = np.zeros((1,m))
  print("X.shape is {}".format(X.shape))
  
  for i,image_file in enumerate(images) :
    image = read_image(image_file)
    X[:,i] = np.squeeze(image.reshape((n_x,1)))
    if '-' in image_file.lower() :
      y[0,i] = 1
    else : # for test data
      y[0,i] = image_file.split('/')[-1].split('.')[0]
      
    if i%10 == 0 :
      print("Proceed {} of {}".format(i, m))
    
  return X,y

X_train, y_train = prep_data(train_images)
X_test, test_idx = prep_data(test_images)

print("Train shape: {}".format(X_train.shape))
print("Test shape: {}".format(X_test.shape))
X_test, test_idx = prep_data(test_images)

classes = {
           1: 'Plate Number'}

def show_images(X, y, idx) :
  image = X[idx]
  image = image.reshape((ROWS, COLS, CHANNELS))
  plt.figure(figsize=(4,2))
  plt.imshow(image)
  plt.title("This is a {}".format(classes[y[idx,0]]))
  plt.show()

show_images(X_train.T, y_train.T, 48)
