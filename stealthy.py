"""Capture an Image from webcam without use interaction."""
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

    return str(path / f"{username}-{now}.png")


# define a video capture object
vid = cv2.VideoCapture(0)
# set the resolution to the maximum value
vid.set(cv2.CAP_PROP_FRAME_WIDTH, 10_000)
vid.set(cv2.CAP_PROP_FRAME_HEIGHT, 10_000)
while True:
    success, frame = vid.read()
    if success:
        cv2.imwrite(filename(), frame)
        break

# After the loop release the cap object
vid.release()
# Destroy all the windows
cv2.destroyAllWindows()
