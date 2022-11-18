"""Recognize my face."""
import datetime
import getpass
import pathlib
import pickle  # noqa: S403
import time
from concurrent.futures import FIRST_COMPLETED
from concurrent.futures import Future
from concurrent.futures import ProcessPoolExecutor
from concurrent.futures import wait
from functools import partial
from multiprocessing import cpu_count
from typing import Callable
from typing import List
from typing import Optional
from typing import Tuple

import cv2
import face_recognition
import typer
from numpy import ndarray

DETECTION_METHOD = "cnn"


def get_image_path() -> pathlib.Path:
    """
    Get the path of the directory where the images are stored.

    Default: ``~/Pictures/login-capture/``.
    """
    return pathlib.Path.home() / "Pictures" / "login-capture"


def load_encodings(path: pathlib.Path) -> Tuple[ndarray, ...]:
    """Load pickle files containing face encodings."""
    encodings = []
    for encoding in path.glob(f"*.jpg.{DETECTION_METHOD}-encoded.pickle"):
        with open(encoding, "rb") as f:
            encodings.append(pickle.load(f))  # noqa: S301
    return tuple(encodings)


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


def get_camera_capture(
    camera: int = 0,
    ratio: Optional[float] = 3,
) -> Tuple[cv2.VideoCapture, int]:
    """Define a video capture object."""
    vid = cv2.VideoCapture(camera)
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
            img=flipped,
            pt1=(x, y - text_size[1]),
            pt2=(x + text_size[0], y),
            color=(255, 255, 255),
            thickness=cv2.FILLED,
        )
        cv2.putText(
            img=flipped,
            text=f"Face {w}x{h}",
            org=(x, y),
            fontFace=cv2.FONT_HERSHEY_SIMPLEX,
            fontScale=1,
            color=(0, 0, 0),
            thickness=2,
        )
        detected = True

    return detected, flipped


def recognize_face(
    data: Tuple[ndarray, ...],
    frame: ndarray,
) -> Tuple[int, List[ndarray], ndarray]:
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
        # attempt to match each face in the input image to our known encodings
        # tolerance: How much distance between faces to consider it a match.
        # Lower is more strict. 0.6 is typical best performance.
        matches = face_recognition.compare_faces(
            known_face_encodings=data,
            face_encoding_to_check=encoding,
            tolerance=0.4,
        )
    return len([m for m in matches if m]), encodings, frame


def run_recognition(
    executor: ProcessPoolExecutor,
    futures: List[Future[ndarray]],
    data: Tuple[ndarray, ...],
    frame: ndarray,
) -> Tuple[int, List[ndarray], ndarray]:
    """Run face recognition in a separate process."""
    if len(futures) < cpu_count():
        # Start a new recognition task
        futures.append(executor.submit(recognize_face, data, frame))
    # Check if any of the tasks are done
    done, not_done = wait(futures, timeout=0.01, return_when=FIRST_COMPLETED)
    recognized = False
    for future in done:
        matches, encodings, origin_frame = future.result()
        if matches:
            recognized = True
        futures.remove(future)
    if not recognized:
        return 0, [], frame
    for future in not_done:
        future.cancel()
    print(f"Found {matches} matches")
    return matches, encodings, origin_frame


def save_image_and_encoding(frame: ndarray, encodings: List[ndarray]) -> None:
    """Save an image and its encoding to disk."""
    filename = get_filename()
    cv2.imwrite(filename, frame, [cv2.IMWRITE_JPEG_QUALITY, 100])
    # Write the encoding only when exactly one face was detected
    if len(encodings) == 1:
        with open(f"{filename}.{DETECTION_METHOD}-encoded.pickle", "wb") as encoded:
            pickle.dump(encodings[0], encoded)


def capture_and_display(
    vid: cv2.VideoCapture,
    window_name: str,
    detector: Callable[[ndarray], ndarray],
    frame_count: int,
) -> Tuple[bool, int, ndarray]:
    """Capture a frame from the camera and display it."""
    # Capture the video frame by frame
    ret, frame = vid.read()
    if not ret:
        time.sleep(0.1)
        return False, frame_count, []
    frame_count += 1
    frame.flags.writeable = False
    # Flip the frame so that it is the mirror view
    flipped = cv2.flip(frame, 1)
    # detect faces in the flipped frame
    detected, flipped = detect_face(flipped, detector)
    # Display the resulting frame
    cv2.imshow(window_name, flipped)
    cv2.waitKey(30)
    if frame_count < 10:
        # Skip the first 10 frames to allow the camera to adjust
        detected = False
    return detected, frame_count, frame


def main(camera: int = 0) -> None:
    """Run the main program."""
    data = load_encodings(get_image_path())
    vid, min_size = get_camera_capture(camera=camera)
    window_name = create_window()
    face_detector = get_face_detector(min_size)
    frame_count = 0
    futures: List[Future[ndarray]] = []
    with ProcessPoolExecutor(max_workers=max(cpu_count() // 2, 1)) as executor:
        while vid.isOpened():
            detected, frame_count, frame = capture_and_display(
                vid=vid,
                window_name=window_name,
                detector=face_detector,
                frame_count=frame_count,
            )
            if not detected:
                continue
            matches, encodings, origin_frame = run_recognition(
                executor,
                futures,
                data,
                frame,
            )
            if not matches:
                continue
            # save the image once a face was recognized.
            save_image_and_encoding(origin_frame, encodings)
            # release the camera, this will stop the video capture and end the loop
            vid.release()

    cv2.destroyAllWindows()


if __name__ == "__main__":
    typer.run(main)
