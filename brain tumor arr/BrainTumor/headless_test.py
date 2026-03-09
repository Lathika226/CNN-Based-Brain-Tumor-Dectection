import os
import cv2 # type: ignore
import numpy as np # type: ignore
import matplotlib.pyplot as plt # type: ignore
from tensorflow.keras.models import model_from_json, Sequential # type: ignore
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense # type: ignore
from tensorflow.keras.utils import custom_object_scope # type: ignore
from tensorflow.keras.models import Model # type: ignore 
import shutil

artifacts_dir = r"C:\Users\tinku\.gemini\antigravity\brain\6f56396c-71b4-4727-82ef-edab2e27b0fa"

disease = ['Normal','Benign']

def cropTumorRegion():
    img = cv2.imread('myimg.png')
    orig = cv2.imread('test1.png')
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    thresh = cv2.threshold(gray, 30, 255, cv2.THRESH_BINARY)[1]
    contours, _ = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    min_area = 0.95*180*35
    max_area = 1.05*180*35
    result = orig.copy()
    life = 0
    for c in contours:
        area = cv2.contourArea(c)
        if life == 0:
            life = len(c)
        cv2.drawContours(result, [c], -1, (0, 0, 255), 10)
        if area > min_area and area < max_area:
            cv2.drawContours(result, [c], -1, (0, 255, 255), 10)
    return result, life    

with open('Model/segmented_model.json', "r") as json_file:
    loaded_model_json = json_file.read()
    with custom_object_scope({'Model': Model}):
        segmented_model = model_from_json(loaded_model_json)
segmented_model.load_weights("Model/segmented_weights.h5")

classifier = Sequential() 
classifier.add(Conv2D(32, (3, 3), input_shape=(64, 64, 1), activation='relu'))
classifier.add(MaxPooling2D(pool_size=(2, 2)))
classifier.add(Conv2D(32, (3, 3), activation='relu'))
classifier.add(MaxPooling2D(pool_size=(2, 2)))
classifier.add(Flatten())
classifier.add(Dense(128, activation='relu'))
classifier.add(Dense(2, activation='softmax'))
classifier.load_weights("Model/model_weights.h5")

def getTumorRegion(filename):
    img = cv2.imread(filename,0)
    img = cv2.resize(img,(64,64), interpolation = cv2.INTER_CUBIC)
    img = img.reshape(1,64,64,1)
    img = (img-127.0)/127.0
    preds = segmented_model.predict(img)
    preds = preds[0]
    orig = cv2.imread(filename,0)
    orig = cv2.resize(orig,(300,300),interpolation = cv2.INTER_CUBIC)
    cv2.imwrite("test1.png",orig)    
    segmented_image = cv2.resize(preds,(300,300),interpolation = cv2.INTER_CUBIC)
    cv2.imwrite("myimg.png",segmented_image*255)
    edge_detection, lifespan = cropTumorRegion()
    return segmented_image*255, edge_detection, lifespan

files_to_test = ['testImages/12.png']

for filename in files_to_test:
    img = cv2.imread(filename)
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img_resized = cv2.resize(img_gray, (64,64))
    im2arr = np.array(img_resized)
    im2arr = im2arr.reshape(1,64,64,1)
    XX = np.asarray(im2arr)
        
    predicts = classifier.predict(XX)
    cls = np.argmax(predicts)
    
    if cls == 0:
        plt.figure(figsize=(10,6))
        img_rgb = cv2.cvtColor(cv2.resize(img, (800,500)), cv2.COLOR_BGR2RGB)
        plt.imshow(img_rgb)
        plt.title('Classification Result: '+disease[cls])
        plt.axis('off')
        plt.savefig(os.path.join(artifacts_dir, 'output_normal.png'))
    elif cls == 1:
        segmented_image, edge_image, lifespan = getTumorRegion(filename)
        
        plt.figure(figsize=(10,6))
        img_rgb = cv2.cvtColor(cv2.resize(img, (800,500)), cv2.COLOR_BGR2RGB)
        plt.imshow(img_rgb)
        plt.title('Classification Result: '+disease[cls])
        plt.axis('off')
        plt.savefig(os.path.join(artifacts_dir, 'output_tumor1.png'))
        
        plt.figure(figsize=(15,5))
        plt.subplot(1,2,1)
        plt.imshow(segmented_image, cmap='gray')
        plt.title('Tumor Extracted Image')
        plt.axis('off')
        
        plt.subplot(1,2,2)
        plt.imshow(cv2.cvtColor(edge_image, cv2.COLOR_BGR2RGB))
        plt.title('Tumor Boundary')
        plt.axis('off')
        
        plt.tight_layout()
        plt.savefig(os.path.join(artifacts_dir, 'output_tumor2.png'))
        
print("Saved to artifacts!")
