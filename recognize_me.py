"""Recognize my face."""
import datetime
import getpass
import pathlib
import pickle

import cv2
import face_recognition

DETECTION_METHOD = "cnn"
path = pathlib.Path.home() / "Pictures" / "login-capture"


def load_encodings(path: pathlib.Path):
    encodings = []
    for encoding in path.glob(f"*.jpg.{DETECTION_METHOD}-encoded.pickle"):
        data = pickle.loads(open(str(encoding), "rb").read())
        encodings.append(data)
    return encodings


def filename() -> str:
    """
    Create a filename based on the username and current date and time.

    The file will be stored in the home directory `~/Pictures/login-capture/`.
    """
    username = getpass.getuser()
    now = datetime.datetime.now().strftime("%Y-%m-%d-%H-%M")

    path = pathlib.Path.home() / "Pictures" / "login-capture"
    path.mkdir(parents=True, exist_ok=True)

    return str(path / f"{username}-{now}.jpg")


data = load_encodings(path)
# define a video capture object
vid = cv2.VideoCapture(0)
# set the resolution to the maximum value
vid.set(cv2.CAP_PROP_FRAME_WIDTH, 10_000)
vid.set(cv2.CAP_PROP_FRAME_HEIGHT, 10_000)

# create an output window and set it to full screen
cv2.namedWindow("frame", cv2.WINDOW_NORMAL)
cv2.setWindowProperty("frame", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

datapath = (
    pathlib.Path(__file__).resolve().parents[0]
    / "data"
    / "haarcascade_frontalface_default.xml"
)
haar_face_cascade = cv2.CascadeClassifier(str(datapath))

w = vid.get(cv2.CAP_PROP_FRAME_WIDTH)
h = vid.get(cv2.CAP_PROP_FRAME_HEIGHT)
min_size = int(min(w, h) / 3)
while vid.isOpened():
    # Capture the video frame
    # by frame
    ret, frame = vid.read()
    flipped = cv2.flip(frame, 1)
    gray = cv2.cvtColor(flipped, cv2.COLOR_BGR2GRAY)
    detected = False

    faces = haar_face_cascade.detectMultiScale(
        gray, 1.1, 3, minSize=(min_size, min_size)
    )
    for x, y, w, h in faces:
        cv2.rectangle(flipped, (x, y), (x + w, y + h), (0, 255, 0), 3)
        text_size, _ = cv2.getTextSize(f"Face {w}x{h}", cv2.FONT_HERSHEY_SIMPLEX, 1, 2)
        cv2.rectangle(
            flipped,
            (x, y - text_size[1]),
            (x + text_size[0], y),
            (255, 255, 255),
            cv2.FILLED,
        )
        cv2.putText(
            flipped, f"Face {w}x{h}", (x, y), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2
        )
        detected = True

    # Display the resulting frame
    cv2.imshow("frame", flipped)

    key = cv2.waitKey(30)
    if not detected:
        continue

    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    boxes = face_recognition.face_locations(rgb, model=DETECTION_METHOD)
    encodings = face_recognition.face_encodings(rgb, boxes, model="large")
    # loop over the facial embeddings
    matches = []
    for encoding in encodings:
        # attempt to match each face in the input image to our known
        # encodings
        matches = face_recognition.compare_faces(
            known_face_encodings=data,
            face_encoding_to_check=encoding,
            tolerance=0.4,
        )
    if not any(matches):
        print("no match")
        continue
    # save the image once a face was recognized.
    num_matches = len([m for m in matches if m])
    print(f"{num_matches} matches")
    fname = filename()
    cv2.imwrite(fname, frame, [cv2.IMWRITE_JPEG_QUALITY, 100])
    # Write the encoding only when exactly one face was detected
    if len(boxes) == 1:
        with open(f"{fname}.{DETECTION_METHOD}-encoded.pickle", "wb") as encoded:
            encoded.write(pickle.dumps(encoding))
    break

# After the loop release the cap object
vid.release()
# Destroy all the windows
cv2.destroyAllWindows()
