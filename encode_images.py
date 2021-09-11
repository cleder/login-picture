"""Compute the face encodings for images."""
import pathlib
import pickle

import cv2
import face_recognition

DETECTION_METHOD = "cnn"

path = pathlib.Path.home() / "Pictures" / "login-capture"
jpg_files = path.glob("*.jpg")
for picture in jpg_files:
    fname = f"{picture}.{DETECTION_METHOD}-encoded.pickle"
    if pathlib.Path(fname).exists():
        continue
    print(picture)
    image = cv2.imread(str(picture))
    rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    # detect the (x, y)-coordinates of the bounding boxes
    # corresponding to each face in the input image
    boxes = face_recognition.face_locations(rgb, model=DETECTION_METHOD)
    if len(boxes) != 1:
        # skip this image when multiple face where detected
        continue
    encodings = face_recognition.face_encodings(rgb, boxes, model="large")
    with open(fname, "wb") as encoded:
        encoded.write(pickle.dumps(encodings[0]))
