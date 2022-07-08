import numpy as np
import cv2
import os
from matplotlib import pyplot as plt
import pywt
from sklearn import svm
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import StandardScaler
import seaborn as sn
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix

def get_cropped_image_if_face(image_path):
    img = cv2.imread(image_path)
    plt.axis('off')
    plt.imshow(img)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    face_cascade = cv2.CascadeClassifier('./haarcascades/haarcascade_frontalface_default.xml')
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)
    if len(faces) != 1:
        return None
    (x, y, w, h) = faces[0]
    face_img = cv2.rectangle(img, (x, y), (x + w, y + h), (0, 0, 255), 5)
    face_img = np.flip(face_img, axis=-1)
    plt.axis('off')
    plt.imshow(face_img)

    return face_img

def create_paths(path_to_data):
    img_dirs = []
    for entry in os.scandir(path_to_data):
        if entry.is_dir():
            img_dirs.append(entry.path)

def create_cropped_images(img_dirs, path_to_cr_data):
    cropped_image_dirs = []
    celebrity_file_names_dict = {}
    for img_dir in img_dirs:
        count = 1
        celebrity_name = img_dir.split('/')[-1]
        celebrity_file_names_dict[celebrity_name] = []
        for entry in os.scandir(img_dir):
            try:
                roi_color = get_cropped_image_if_face(entry.path)
            except:
                print("Unable to create cropped image")
                print(entry.path)

            if roi_color is not None:
                cropped_folder = path_to_cr_data + celebrity_name
                if not os.path.exists(cropped_folder):
                    os.makedirs(cropped_folder)
                    cropped_image_dirs.append(cropped_folder)
                    print("Generating cropped images in folder: ", cropped_folder)

                cropped_file_name = celebrity_name + str(count) + ".png"
                cropped_file_path = cropped_folder + "/" + cropped_file_name

                cv2.imwrite(cropped_file_path, roi_color)
                celebrity_file_names_dict[celebrity_name].append(cropped_file_path)
                count += 1

    return celebrity_file_names_dict

def w2d(img, mode='haar', level=1):
    imArray = img
    #Datatype conversions
    #convert to grayscale
    imArray = cv2.cvtColor(imArray,cv2.COLOR_RGB2GRAY)
    #convert to float
    imArray =  np.float32(imArray)
    imArray /= 255
    # compute coefficients
    coeffs=pywt.wavedec2(imArray, mode, level=level)
    #Process Coefficients
    coeffs_H=list(coeffs)
    coeffs_H[0] *= 0
    # reconstruction
    imArray_H=pywt.waverec2(coeffs_H, mode)
    imArray_H *= 255
    imArray_H =  np.uint8(imArray_H)

    return imArray_H

def create_labels(celebrity_file_names_dict):
    class_dict = {}
    count = 0
    for celebrity_name in celebrity_file_names_dict.keys():
        class_dict[celebrity_name] = count
        count = count + 1
    class_dict


def create_input_image(training_image_path):
    img = cv2.imread(training_image_path)
    scalled_raw_img = cv2.resize(img, (32, 32))
    img_har = w2d(img, 'db1', 5)
    scalled_img_har = cv2.resize(img_har, (32, 32))
    combined_img = np.vstack((scalled_raw_img.reshape(32 * 32 * 3, 1), scalled_img_har.reshape(32 * 32, 1)))

    return combined_img

def create_inputs(celebrity_file_names_dict,class_dict):
    X, y = [], []
    for celebrity_name, training_files in celebrity_file_names_dict.items():
        for training_image_path in training_files:
            try:
                combined_img = create_input_image(training_image_path)
                X.append(combined_img)
                y.append(class_dict[celebrity_name])
            except:
                print("Couldn't find path")

def classify_data(X, y):
    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42)

    model_params = {
        'svm': {
            'model': svm.SVC(gamma='auto', probability=True),
            'params': {
                'svc__C': [1, 10, 100, 1000],
                'svc__kernel': ['rbf', 'linear']
            }
        },
        'random_forest': {
            'model': RandomForestClassifier(),
            'params': {
                'randomforestclassifier__n_estimators': [1, 5, 10]
            }
        },
        'logistic_regression': {
            'model': LogisticRegression(solver='liblinear', multi_class='auto'),
            'params': {
                'logisticregression__C': [1, 5, 10]
            }
        }
    }
    scores = []
    best_estimators = {}
    for algo, mp in model_params.items():
        pipe = make_pipeline(StandardScaler(), mp['model'])
        clf = GridSearchCV(pipe, mp['params'], cv=5, return_train_score=False)
        clf.fit(X_train, y_train)
        scores.append({
            'model': algo,
            'best_score': clf.best_score_,
            'best_params': clf.best_params_
        })
        best_estimators[algo] = clf.best_estimator_

    print(best_estimators['svm'].score(X_test, y_test))
    print(best_estimators['random_forest'].score(X_test, y_test))
    print(best_estimators['logistic_regression'].score(X_test, y_test))

def create_confusion_matrix(best_clf, X_test, y_test):
    cm = confusion_matrix(y_test, best_clf.predict(X_test))
    plt.figure(figsize=(10, 7))
    sn.heatmap(cm, annot=True)
    plt.xlabel('Predicted')
    plt.ylabel('Truth')