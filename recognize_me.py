"""Recognize my face."""
import datetime
import getpass
import pathlib
import pickle  # noqa: S403
from functools import partial
from typing import Callable
from typing import List
from typing import Optional
from typing import Tuple

import cv2  # noqa: I900
import face_recognition
from numpy import ndarray  # noqa: I900

DETECTION_METHOD = "cnn"


def get_image_path() -> pathlib.Path:
    """
    Get the path of the directory where the images are stored.

    Default: ``~/Pictures/login-capture/``.
    """
    return pathlib.Path.home() / "Pictures" / "login-capture"


def load_encodings(path: pathlib.Path) -> List[ndarray]:
    """Load pickle files containing face encodings."""
    encodings = []
    for encoding in path.glob(f"*.jpg.{DETECTION_METHOD}-encoded.pickle"):
        with open(encoding, "rb") as f:
            encodings.append(pickle.load(f))  # noqa: S301
    return encodings


def get_filename() -> str:
    """
    Create a filename based on the username and current date and time.

    The file will be stored in the home directory `~/Pictures/login-capture/`.
    """
    username = getpass.getuser()
    now = datetime.datetime.now().strftime("%Y-%m-%d-%H-%M")
    path = get_image_path()
    path.mkdir(parents=True, exist_ok=True)
    return str(path / f"{username}-{now}.jpg")


def create_window() -> str:
    """Create an output window and set it to full screen."""
    window_name = "Recognize me"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.setWindowProperty(window_name, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
    cv2.setWindowProperty(window_name, cv2.WND_PROP_TOPMOST, 1)
    return window_name


def get_camera_capture(ratio: Optional[float] = 3) -> Tuple[cv2.VideoCapture, int]:
    """Define a video capture object."""
    vid = cv2.VideoCapture(0)
    # set the resolution to the maximum value
    vid.set(cv2.CAP_PROP_FRAME_WIDTH, 10_000)
    vid.set(cv2.CAP_PROP_FRAME_HEIGHT, 10_000)
    w = vid.get(cv2.CAP_PROP_FRAME_WIDTH)
    h = vid.get(cv2.CAP_PROP_FRAME_HEIGHT)
    return vid, int(min(w, h) / ratio)


def get_face_detector(min_size: int) -> Callable[[ndarray], ndarray]:
    """Create a face detection classifier and return a detection function."""
    datapath = (  # noqa: ECE001
        pathlib.Path(__file__).resolve().parents[0]
        / "data"
        / "haarcascade_frontalface_default.xml"
    )
    haar_face_cascade = cv2.CascadeClassifier(str(datapath))
    return partial(
        haar_face_cascade.detectMultiScale,
        scaleFactor=1.1,
        minNeighbors=3,
        minSize=(min_size, min_size),
    )


def detect_face(
    flipped: ndarray,
    detector: Callable[[ndarray], ndarray],
) -> Tuple[bool, ndarray]:
    """Detect faces in an image and draw a bounding box around them."""
    gray = cv2.cvtColor(flipped, cv2.COLOR_BGR2GRAY)
    detected = False
    faces = detector(gray)
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
            flipped,
            f"Face {w}x{h}",
            (x, y),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0, 0, 0),
            2,
        )
        detected = True

    return detected, flipped


def recognize_face(data: List[ndarray], frame: ndarray) -> Tuple[int, List[ndarray]]:
    """
    Recognize a face in an image.

    Return the number of matches and the encodings.
    """
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    boxes = face_recognition.face_locations(rgb, model=DETECTION_METHOD)
    encodings = face_recognition.face_encodings(rgb, boxes, model="large")
    # loop over the facial embeddings
    matches = []
    for encoding in encodings:
        # attempt to match each face in the input image to our known
        # encodings
        # tolerance: How much distance between faces to consider it a match.
        # Lower is more strict. 0.6 is typical best performance.
        matches = face_recognition.compare_faces(
            known_face_encodings=data,
            face_encoding_to_check=encoding,
            tolerance=0.4,
        )
    return len([m for m in matches if m]), encodings


def save_image_and_encoding(frame: ndarray, encodings: List[ndarray]) -> None:
    """Save an image and its encoding to disk."""
    filename = get_filename()
    cv2.imwrite(filename, frame, [cv2.IMWRITE_JPEG_QUALITY, 100])
    # Write the encoding only when exactly one face was detected
    if len(encodings) == 1:
        with open(f"{filename}.{DETECTION_METHOD}-encoded.pickle", "wb") as encoded:
            pickle.dump(encodings[0], encoded)


data = load_encodings(get_image_path())
vid, min_size = get_camera_capture()
window_name = create_window()
face_detector = get_face_detector(min_size)

while vid.isOpened():
    # Capture the video frame by frame
    ret, frame = vid.read()
    if not ret:
        continue
    # Flip the frame so that it is the mirror view
    flipped = cv2.flip(frame, 1)
    # detect faces in the flipped frame
    detected, flipped = detect_face(flipped, face_detector)
    # Display the resulting frame
    cv2.imshow(window_name, flipped)
    key = cv2.waitKey(30)
    if not detected:
        continue
    matches, encodings = recognize_face(data, frame)
    if not matches:
        print("No face recognized")
        continue
    # save the image once a face was recognized.
    save_image_and_encoding(frame, encodings)
    print(f"{matches} matches")
    # release the camera, this will stop the video capture and end the loop
    vid.release()

# Destroy all the windows
cv2.destroyAllWindows()
