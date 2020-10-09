from PIL import Image, ImageDraw
from sklearn import svm

import face_recognition
import pickle
import os

debug = True
extensions = ['png', 'jpg', 'jpeg']

# Training the SVC classifier
def train(training_dir='./training/'):
    global debug, extensions
    # The training data would be all the face encodings from all the known images and the labels are their names
    encodings = []
    names = []

    # Training directory
    train_dir = os.listdir(training_dir)

    # Loop through each person in the training directory
    for person in train_dir:
        pix = os.listdir(training_dir + person)

        # Loop through each training image for the current person
        for person_img in pix:
            # Get the face encodings for the face in each image file
            face = face_recognition.load_image_file(training_dir + person + "/" + person_img)
            face_bounding_boxes = face_recognition.face_locations(face)

            #If training image contains exactly one face
            if len(face_bounding_boxes) == 1:
                face_enc = face_recognition.face_encodings(face)[0]
                # Add face encoding for current image with corresponding label (name) to the training data
                encodings.append(face_enc)
                names.append(person)
            else:
                if debug: print(person + "/" + person_img + " was skipped and can't be used for training")

    # Create and train the SVC classifier
    knn_clf = svm.SVC(gamma='scale')
    knn_clf.fit(encodings,names)

    knn_clf = train()

    return knn_clf

if debug: print('TRAINING')
knn_clf = train()
if debug: print('SAVING knn classifier')
pickle_file = file('knnclf.data', 'w')
pickle.dump(knn_clf, pickle_file)
pickle_file.close()
if debug: print('SAVED knn classifier')