from PIL import Image, ImageDraw
from sklearn import svm

import face_recognition
import pickle
import os

debug = True

def identify_faces(input_image_path, knn_clf, distance_threshold=0.6):

    # test image
    input_image = face_recognition.load_image_file(input_image_path)

    # Find all the faces in the test image using the default HOG-based model
    found_face_locations = face_recognition.face_locations(input_image)

    # If no faces are found in the image, return an empty result.
    if len(found_face_locations) == 0:
        return []

    # Find encodings for faces in the test iamge
    faces_encodings = face_recognition.face_encodings(input_image, known_face_locations=found_face_locations)

    # Use the KNN model to find the best matches for the test face
    closest_distances = knn_clf.kneighbors(faces_encodings, n_neighbors=1)
    are_matches = [closest_distances[0][i][0] <= distance_threshold for i in range(len(found_face_locations))]

    return [(pred, loc) if rec else ("???", loc) for pred, loc, rec in zip(knn_clf.predict(faces_encodings), found_face_locations, are_matches)]

def improve_faces(input_image_path, faces_found, improvement_dir='./improvements', output_image=None):
    pil_image = Image.open(input_image_path)
    draw = ImageDraw.Draw(pil_image)

    for name, (top, right, bottom, left) in faces_found:
        improvement = Image.open(os.path.join(improvement_dir, name+'.png'))
        
        improvement = improvement.resize((bottom-top, right-left))
        
        pil_image.paste(improvement, (left, top))

    if output_image:
        pil_image.save(output_image)

    # Remove the drawing library from memory as per the Pillow docs
    del draw

    # Display the resulting image
    pil_image.show()

if debug: print('LOAD TRAINING')
knn_clf_file = open('knnclf.data', "rb")
knn_clf = pickle.load(knn_clf_file)

if debug: print('FACERECOGNITION')
faces = identify_faces('input_image.jpg', knn_clf)

if debug: print('RESULTS')
improve_faces(input_image_path='input_image.jpg', faces_found=faces, output_image='output_image.jpg')
