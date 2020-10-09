from PIL import Image, ImageDraw
from sklearn import neighbors

import face_recognition
import pickle
import os

debug = True
extensions = ['png', 'jpg', 'jpeg']

def train(training_dir='./training/', n_neighbors=None, knn_algo='ball_tree'):
    X = []
    y = []

    # Loop through each person in the training set
    for person in os.listdir(training_dir):
        if debug: print("== "+person+" ==")
        if not os.path.isdir(os.path.join(training_dir, person)):
            continue

        # Loop through each training image for the current person
        for training_image in os.listdir(os.path.join(training_dir, person)):
            if debug: print("> "+training_image)
            image = face_recognition.load_image_file(os.path.join(training_dir, person, training_image))
            face_bounding_boxes = face_recognition.face_locations(image)

            if len(face_bounding_boxes) != 1:
                # If there are no people (or too many people) in a training image, skip the image.
                if debug: print("Image {} not suitable for training: {}".format(training_image, "Didn't find a face" if len(face_bounding_boxes) < 1 else "Found more than one face"))
            else:
                # Add face encoding for current image to the training set
                X.append(face_recognition.face_encodings(image, known_face_locations=face_bounding_boxes)[0])
                y.append(person)

    # Determine how many neighbors to use for weighting in the KNN classifier
    if n_neighbors is None:
        n_neighbors = int(round(math.sqrt(len(X))))
        if debug: print("Chose n_neighbors automatically:", n_neighbors)

    # Create and train the KNN classifier
    knn_clf = neighbors.KNeighborsClassifier(n_neighbors=n_neighbors, algorithm=knn_algo, weights='distance')
    knn_clf.fit(X, y)

    return knn_clf

if debug: print('TRAINING')
knn_clf = train(training_dir='./training/', n_neighbors=2)
if debug: print('SAVING knn classifier')
pickle_file = open('knnclf.data', 'wb')
pickle.dump(knn_clf, pickle_file)
pickle_file.close()
if debug: print('SAVED knn classifier')