"""Capture an Image from webcam when you press any key."""
import datetime
import getpass
import pathlib

import cv2


def filename():
    """
    Create a filename based on the username and current date and time.

    The file will be stored in the home directory `~/Pictures/login-capture/`.
    """
    username = getpass.getuser()
    now = datetime.datetime.now().strftime("%Y-%m-%d-%H-%M")

    path = pathlib.Path.home() / "Pictures" / "login-capture"
    path.mkdir(parents=True, exist_ok=True)

    return str(path / f"{username}-{now}.jpg")


# define a video capture object
vid = cv2.VideoCapture(0)
# set the resolution to the maximum value
vid.set(cv2.CAP_PROP_FRAME_WIDTH, 10_000)
vid.set(cv2.CAP_PROP_FRAME_HEIGHT, 10_000)

# create an output window and set it to full screen
cv2.namedWindow("frame", cv2.WINDOW_NORMAL)
cv2.setWindowProperty("frame", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

datapath = pathlib.Path(__file__).resolve().parents[0] / 'data' / 'haarcascade_frontalface_default.xml'
haar_face_cascade = cv2.CascadeClassifier(str(datapath))

w = vid.get(cv2.CAP_PROP_FRAME_WIDTH)
h = vid.get(cv2.CAP_PROP_FRAME_HEIGHT)
min_size = int(min(w, h) / 3)
while True:
    # Capture the video frame
    # by frame
    ret, frame = vid.read()
    flipped = cv2.flip(frame, 1)
    gray = cv2.cvtColor(flipped, cv2.COLOR_BGR2GRAY)
    detected = False

    faces = haar_face_cascade.detectMultiScale(gray, 1.1, 3, minSize=(min_size, min_size))
    for x, y, w, h in faces:
        cv2.rectangle(flipped, (x, y), (x + w, y + h), (0, 255, 0), 3)
        text_size, _ = cv2.getTextSize(f'Face {w}x{h}', cv2.FONT_HERSHEY_SIMPLEX, 1, 2)
        cv2.rectangle(flipped, (x, y - text_size[1]), (x + text_size[0], y), (255, 255, 255), cv2.FILLED)
        cv2.putText(flipped, f'Face {w}x{h}', (x, y), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)
        detected = True

    # Display the resulting frame
    cv2.imshow("frame", flipped)

    key = cv2.waitKey(30)
    if not detected:
        continue
    # press any key to exit
    cv2.imwrite(filename(), frame, [cv2.IMWRITE_JPEG_QUALITY, 100])
    break

# After the loop release the cap object
vid.release()
# Destroy all the windows
cv2.destroyAllWindows()
