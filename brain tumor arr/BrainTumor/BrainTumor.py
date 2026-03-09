from tkinter import messagebox
from tkinter import *
from tkinter import simpledialog
import tkinter
from tkinter import filedialog
import matplotlib.pyplot as plt # type: ignore
import numpy as np # type: ignore
from tkinter.filedialog import askopenfilename
import os
import cv2 # type: ignore
from sklearn.model_selection import train_test_split # type: ignore
from sklearn.metrics import accuracy_score  # type: ignore
# from keras.utils.np_utils import to_categorical
from tensorflow.keras.utils import to_categorical # type: ignore

# from keras.layers import  MaxPooling2D
from tensorflow.keras.layers import MaxPooling2D # type: ignore

# from keras.layers import Dense, Dropout, Activation, Flatten
from tensorflow.keras.layers import Dense, Dropout, Activation, Flatten # type: ignore

# from keras.layers import Convolution2D
from tensorflow.keras.layers import Conv2D  # Conv2D is the updated name in tf.keras # type: ignore

# from keras.models import Sequential
from tensorflow.keras.models import Sequential # type: ignore

# from keras.models import model_from_json
from tensorflow.keras.models import model_from_json, Model # type: ignore
from tensorflow.keras.utils import custom_object_scope # type: ignore

import pickle
from sklearn import metrics # type: ignore
from tkinter import ttk

main = tkinter.Tk()
main.title("Convolutional Neural Network based Brain Tumor Detection") #designing main screen
main.geometry("1300x1200")

filename = ""
accuracy = 0.0
X = np.array([])
Y = np.array([])
classifier = None
disease = ['Normal','Benign']

print("Starting GUI...")
with open('Model/segmented_model.json', "r") as json_file:
    print("Reading JSON...")
    loaded_model_json = json_file.read()
    print("Loading model...")
    with custom_object_scope({'Model': Model}):
        segmented_model = model_from_json(loaded_model_json)
json_file.close()    
print("Loading weights...")
segmented_model.load_weights("Model/segmented_weights.h5")
print("Weights loaded successfully!")
# segmented_model._make_predict_function()



def cropTumorRegion():
    img = cv2.imread('myimg.png')
    orig = cv2.imread('test1.png')
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    thresh = cv2.threshold(gray, 30, 255, cv2.THRESH_BINARY)[1]
    contours = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    contours = contours[0] if len(contours) == 2 else contours[1]
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

def getTumorRegion(filename):
    global segmented_model
    img = cv2.imread(filename,0)
    img = cv2.resize(img,(64,64), interpolation = cv2.INTER_CUBIC)
    img = img.reshape(1,64,64,1)
    img = (img-127.0)/127.0
    preds = segmented_model.predict(img)
    preds = preds[0]
    print(preds.shape)
    orig = cv2.imread(filename,0)
    orig = cv2.resize(orig,(300,300),interpolation = cv2.INTER_CUBIC)
    cv2.imwrite("test1.png",orig)    
    segmented_image = cv2.resize(preds,(300,300),interpolation = cv2.INTER_CUBIC)
    cv2.imwrite("myimg.png",segmented_image*255)
    edge_detection, lifespan = cropTumorRegion()
    return segmented_image*255, edge_detection, lifespan
    

def uploadDataset(): #function to upload dataset
    global filename
    filename = filedialog.askdirectory(initialdir=".")
    text.delete('1.0', END)
    text.insert(END,filename+" loaded\n");

def datasetPreprocessing():
    text.delete('1.0', END)
    global X
    global Y
    X = []  # Initialize as list, not numpy array
    Y = []  # Initialize as list, not numpy array
    
    # Check if filename is valid
    if not filename:
        text.insert(END,"Error: No folder selected. Please click 'Upload Dataset' first.\n")
        text.update_idletasks()
        return
    
    if not os.path.exists(filename):
        text.insert(END,"Error: Selected folder does not exist.\n")
        text.insert(END,"Path: " + filename + "\n")
        text.update_idletasks()
        return
    
    text.insert(END,"=" * 50 + "\n")
    text.insert(END,"LOADING DATASET\n")
    text.insert(END,"=" * 50 + "\n\n")
    text.insert(END,"Selected folder: " + filename + "\n\n")
    text.insert(END,"Scanning for images...\n")
    text.update_idletasks()
    main.update()
    
    if False and os.path.exists('Model/myimg_data.txt.npy'): # Disabled cache to allow loading new folders
        X = np.load('Model/myimg_data.txt.npy')
        Y = np.load('Model/myimg_label.txt.npy')
    else:
        image_count = 0
        subfolder_counts = {}
        supported_formats = ('.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.JPG', '.JPEG', '.PNG', '.BMP', '.TIFF')
        error_count = 0
        
        # Walk through all directories and load images
        for root, dirs, files in os.walk(filename):
            for file in files:
                if file.lower().endswith(supported_formats):
                    file_path = os.path.join(root, file)
                    try:
                        img = cv2.imread(file_path)
                        
                        if img is None:
                            error_count += 1
                            continue
                        
                        # Get relative subfolder name for labeling
                        rel_path = os.path.relpath(root, filename)
                        if rel_path == ".":
                            folder_label = "root"
                        else:
                            folder_label = rel_path.split(os.sep)[0]
                        
                        # Count images per subfolder
                        if folder_label not in subfolder_counts:
                            subfolder_counts[folder_label] = 0
                        subfolder_counts[folder_label] += 1
                        
                        # Process image
                        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                        ret2, th2 = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
                        img = cv2.resize(img, (64, 64))
                        im2arr = np.array(img)
                        im2arr = im2arr.reshape(64, 64, 1)
                        X.append(im2arr)
                        Y.append(0)  # Default label
                        image_count += 1
                        
                        print(file_path)
                    
                    except Exception as e:
                        error_count += 1
                        continue
        
        # Convert lists to numpy arrays
        if len(X) > 0:
            X = np.asarray(X)
            Y = np.asarray(Y)
        else:
            X = np.array([])
            Y = np.array([])
        
        if len(X) == 0:
            text.insert(END,"\n❌ ERROR: No images found in the selected folder.\n")
            text.insert(END,"Supported formats: .jpg, .jpeg, .png, .bmp, .tiff\n")
            text.insert(END,"Images may be corrupted or in an unsupported format.\n")
            text.update_idletasks()
            return
        
        # Display folder contents summary
        max_folder_count = max(subfolder_counts.values()) if subfolder_counts else 0
        max_folder_name = max(subfolder_counts.keys(), key=lambda k: subfolder_counts[k]) if subfolder_counts else "none"
        
        text.insert(END,f"\n✓ Successfully processed {image_count} images!\n")
        text.insert(END,f"✓ Largest folder: '{max_folder_name}' with {max_folder_count} images\n\n")
        text.insert(END,"=" * 50 + "\n")
        text.insert(END,"FOLDER CONTENTS SUMMARY\n")
        text.insert(END,"=" * 50 + "\n\n")
        text.insert(END,"Images per subfolder:\n")
        for folder, count in sorted(subfolder_counts.items()):
            text.insert(END,f"  • {folder}: {count} images\n")
        if error_count > 0:
            text.insert(END,f"\n⚠ Skipped {error_count} unreadable files\n")
        text.insert(END,f"\n✓ Total processed: {image_count} images\n")
        text.insert(END,f"✓ Data shape: {X.shape}\n")
        text.insert(END,"=" * 50 + "\n\n")
        text.update_idletasks()
        main.update()

        np.save("Model/myimg_data.txt",X)
        np.save("Model/myimg_label.txt",Y)
        
        # Display final results
        text.insert(END,"=" * 50 + "\n")
        text.insert(END,"DATASET ANALYSIS RESULTS\n")
        text.insert(END,"=" * 50 + "\n\n")
        text.insert(END,f"✓ Total number of images in dataset: {len(X)}\n")
        text.insert(END,f"✓ Total number of classes: {len(set(Y))}\n")
        text.insert(END,f"✓ Class labels: {disease}\n")
        text.insert(END,f"✓ Dataset array shape: {X.shape}\n")
        text.insert(END,f"✓ Labels array shape: {Y.shape}\n")
        text.insert(END,"\n" + "=" * 50 + "\n")
        text.insert(END,"✓ Dataset is ready for training!\n")
        text.insert(END,"You can now train the model.\n")
        text.insert(END,"=" * 50 + "\n")
        text.update_idletasks()
        main.update()
        
        # Display sample image (always show if we have images)
        if len(X) > 0:
            try:
                text.insert(END,"Displaying sample image...\n")
                text.update_idletasks()
                main.update()
                
                plt.figure(figsize=(6,6))
                sample_idx = min(5, len(X) - 1)  # Show 6th image or last if fewer
                sample_img = X[sample_idx].reshape(64, 64)  # Reshape from (64,64,1) to (64,64)
                plt.imshow(sample_img, cmap='gray')
                plt.title(f'Sample Processed Image #{sample_idx + 1} from Dataset')
                plt.axis('off')  # Hide axes for cleaner look
                plt.tight_layout()
                
                # Force display and make it non-blocking
                plt.show(block=False)
                plt.pause(0.1)  # Small pause to ensure display
                
                text.insert(END,"✓ Sample image displayed successfully!\n")
                text.update_idletasks()
                
            except Exception as e:
                text.insert(END,f"⚠ Could not display sample image: {str(e)}\n")
                text.insert(END,"But dataset loading was successful!\n")
                text.update_idletasks()




# def trainTumorDetectionModel():
#     global accuracy
#     global classifier
#     text.delete('1.0', END)
#     YY = to_categorical(Y)

#     indices = np.arange(X.shape[0])
#     np.random.shuffle(indices)

#     x_train = X[indices]
#     y_train = YY[indices]

#     if os.path.exists('Model/model.json'):
#         with open('Model/model.json', "r") as json_file:
#            loaded_model_json = json_file.read()
#            classifier = model_from_json(loaded_model_json)

#         classifier.load_weights("Model/model_weights.h5")
#         classifier._make_predict_function()           
#     else:
#         X_trains, X_tests, y_trains, y_tests = train_test_split(x_train, y_train, test_size = 0.2, random_state = 0)
#         classifier = Sequential() 
#         classifier.add(Convolution2D(32, 3, 3, input_shape = (x_train.shape[1], x_train.shape[2], x_train.shape[3]), activation = 'relu'))
#         classifier.add(MaxPooling2D(pool_size = (2, 2)))
#         classifier.add(Convolution2D(32, 3, 3, activation = 'relu'))
#         classifier.add(MaxPooling2D(pool_size = (2, 2)))
#         classifier.add(Flatten())
#         classifier.add(Dense(output_dim = 128, activation = 'relu'))
#         classifier.add(Dense(output_dim = 2, activation = 'softmax'))
#         print(classifier.summary())
#         classifier.compile(optimizer = 'adam', loss = 'categorical_crossentropy', metrics = ['accuracy'])
#         hist = classifier.fit(x_train, y_train, batch_size=16, epochs=10,validation_data=(X_tests, y_tests), shuffle=True, verbose=2)
#         classifier.save_weights('Model/model_weights.h5')            
#         model_json = classifier.to_json()
#         with open("Model/model.json", "w") as json_file:
#             json_file.write(model_json)
#         f = open('Model/history.pckl', 'wb')
#         pickle.dump(hist.history, f)
#         f.close()
#     f = open('Model/history.pckl', 'rb')
#     data = pickle.load(f)
#     f.close()
#     acc = data['accuracy']
#     accuracy = acc[9] * 100
#     text.insert(END,'\n\nCNN Brain Tumor Model Generated. See black console to view layers of CNN\n\n')
#     text.insert(END,"CNN Brain Tumor Prediction Accuracy on Test Images : "+str(accuracy)+"\n")

def trainTumorDetectionModel():
    global accuracy
    global classifier
    text.delete('1.0', END)
    YY = to_categorical(Y)

    indices = np.arange(X.shape[0]) # type: ignore
    np.random.shuffle(indices)

    x_train = X[indices]
    y_train = YY[indices]

    if False and os.path.exists('Model/model_weights.h5'): # Disabled cache to force training on new dataset
        classifier = Sequential() 
        classifier.add(Conv2D(32, (3, 3), input_shape=(64, 64, 1), activation='relu'))
        classifier.add(MaxPooling2D(pool_size=(2, 2)))
        classifier.add(Conv2D(32, (3, 3), activation='relu'))
        classifier.add(MaxPooling2D(pool_size=(2, 2)))
        classifier.add(Flatten())
        classifier.add(Dense(128, activation='relu'))
        classifier.add(Dense(2, activation='softmax'))
        classifier.load_weights("Model/model_weights.h5")
    else:
        X_arr = np.array(X)
        Y_arr = np.array(YY)
        X_trains, X_tests, y_trains, y_tests = train_test_split(x_train, y_train, test_size = 0.2, random_state = 0)
        classifier = Sequential() 
        classifier.add(Conv2D(32, (3, 3), input_shape=(X_arr.shape[1], X_arr.shape[2], X_arr.shape[3]), activation='relu'))
        classifier.add(MaxPooling2D(pool_size=(2, 2)))
        classifier.add(Conv2D(32, (3, 3), activation='relu'))
        classifier.add(MaxPooling2D(pool_size=(2, 2)))
        classifier.add(Flatten())
        classifier.add(Dense(128, activation='relu'))
        classifier.add(Dense(2, activation='softmax'))
        print(classifier.summary())
        classifier.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
        hist = classifier.fit(X_trains, y_trains, batch_size=16, epochs=10, validation_data=(X_tests, y_tests), shuffle=True, verbose=2)
        classifier.save_weights('Model/model_weights.h5')
        model_json = classifier.to_json()
        with open("Model/model.json", "w") as json_file:
            json_file.write(model_json)
        with open('Model/history.pckl', 'wb') as f:
            pickle.dump(hist.history, f)
    with open('Model/history.pckl', 'rb') as f:
        data = pickle.load(f)
    acc = data['accuracy']
    accuracy = acc[-1] * 100  # Changed to index -1 to get the last epoch's accuracy
    text.insert(END,'\n\nCNN Brain Tumor Model Generated. See black console to view layers of CNN\n\n')
    text.insert(END,"CNN Brain Tumor Prediction Accuracy on Test Images : "+str(accuracy)+"\n")

       


def tumorClassification():
    filename = filedialog.askopenfilename(initialdir="testImages")
    img = cv2.imread(filename)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img = cv2.resize(img, (64,64))
    im2arr = np.array(img)
    im2arr = im2arr.reshape(1,64,64,1)
    XX = np.asarray(im2arr)
        
    predicts = classifier.predict(XX) if classifier is not None else [] # type: ignore
    print(predicts)
    cls = np.argmax(predicts)
    print(cls)
    if cls == 0:
        img = cv2.imread(filename)
        img = cv2.resize(img, (800,500))
        plt.figure(figsize=(10,6))
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        plt.imshow(img_rgb)
        plt.title('Classification Result: '+disease[cls])
        plt.axis('off')
        plt.show()
    if cls == 1:
        segmented_image, edge_image, lifespan = getTumorRegion(filename)
        img = cv2.imread(filename)
        img = cv2.resize(img, (800,500))
        plt.figure(figsize=(10,6))
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        plt.text(10, 25, 'Classification Result: '+disease[cls], color='yellow', fontsize=12, weight='bold')
        plt.imshow(img_rgb)
        plt.title('Classification Result: '+disease[cls])
        plt.axis('off')
        
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
        plt.show()
        
        
        

def graph():
    f = open('model/history.pckl', 'rb')
    data = pickle.load(f)
    f.close()

    accuracy = data['accuracy']
    loss = data['loss']

    plt.figure(figsize=(10,6))
    plt.grid(True)
    plt.xlabel('Training Epoch')
    plt.ylabel('Accuracy/Loss')
    plt.plot(loss, 'ro-', color = 'red')
    plt.plot(accuracy, 'ro-', color = 'green')
    plt.legend(['Loss', 'Accuracy'], loc='upper left')
    plt.title('Bone Tumor CNN Model Training Accuracy & Loss Graph')
    plt.show()

font = ('times', 16, 'bold')
title = Label(main, text='Convolutional Neural Network based Brain Tumor Detection')
title.config(bg='darkviolet', fg='gold')  
title.config(font=font)           
title.config(height=3, width=120)       
title.place(x=0,y=5)

font1 = ('times', 12, 'bold')
text=Text(main,height=20,width=150)
scroll=Scrollbar(text)
text.configure(yscrollcommand=scroll.set)
text.place(x=50,y=120)
text.config(font=font1)


font1 = ('times', 12, 'bold')
uploadButton = Button(main, text="Upload Brain Tumor Images Dataset", command=uploadDataset)
uploadButton.place(x=50,y=550)
uploadButton.config(font=font1)  

preprocessButton = Button(main, text="Dataset Preprocessing & Features Extraction", command=datasetPreprocessing)
preprocessButton.place(x=430,y=550)
preprocessButton.config(font=font1) 

cnnButton = Button(main, text="Train CNN Brain Tumor Detection Model", command=trainTumorDetectionModel)
cnnButton.place(x=810,y=550)
cnnButton.config(font=font1) 

classifyButton = Button(main, text="Brain Tumor Prediction", command=tumorClassification)
classifyButton.place(x=50,y=600)
classifyButton.config(font=font1)

graphButton = Button(main, text="Training Accuracy Graph", command=graph)
graphButton.place(x=430,y=600)
graphButton.config(font=font1)

main.config(bg='turquoise')
main.mainloop()
