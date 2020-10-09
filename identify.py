from PIL import Image, ImageDraw
from sklearn import svm

import face_recognition
import pickle
import os

debug = True

def identify_faces(input_image_path, knn_clf):

    # test image
    input_image = face_recognition.load_image_file(input_image_path)

    # Find all the faces in the test image using the default HOG-based model
    found_face_locations = face_recognition.face_locations(input_image)

    # If no faces are found in the image, return an empty result.
    if len(X_face_locations) == 0:
        return []

    # Find encodings for faces in the test iamge
    faces_encodings = face_recognition.face_encodings(input_image, known_face_locations=found_face_locations)

    # Use the KNN model to find the best matches for the test face
    closest_distances = knn_clf.kneighbors(faces_encodings, n_neighbors=1)
    are_matches = [closest_distances[0][i][0] <= distance_threshold for i in range(len(X_face_locations))]

    return [(pred, loc) if rec else ("???", loc) for pred, loc, rec in zip(knn_clf.predict(faces_encodings), found_face_locations, are_matches)]

def show_recognized_faces(input_image_path, faces_found, output_image=None):
    pil_image = Image.open(input_image_path).convert("RGB")
    draw = ImageDraw.Draw(pil_image)

    for name, (top, right, bottom, left) in faces_found:
        # Draw a box around the face using the Pillow module
        draw.rectangle(((left, top), (right, bottom)), outline=(0, 0, 255))

        # There's a bug in Pillow where it blows up with non-UTF-8 text
        # when using the default bitmap font
        name = name.encode("UTF-8")

        # Draw a label with a name below the face
        text_width, text_height = draw.textsize(name)
        draw.rectangle(((left, bottom - text_height - 10), (right, bottom)), fill=(0, 0, 255), outline=(0, 0, 255))
        draw.text((left + 6, bottom - text_height - 5), name, fill=(255, 255, 255, 255))

    if output_image:
        im.save(output_image)

    # Remove the drawing library from memory as per the Pillow docs
    del draw

    # Display the resulting image
    pil_image.show()

if debug: print('TRAINING')

knn_clf_file = file('knnclf.data')
knn_clf = pickle.load(knn_clf_file)

if debug: print('FACERECOGNITION')
faces = identify_faces('input_image.jpg', knn_clf)
if debug: print('RESULTS')
show_recognized_faces('input_image.jpg', faces, 'output.jpg')
