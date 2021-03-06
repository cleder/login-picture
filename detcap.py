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

while True:

    # Capture the video frame
    # by frame
    ret, frame = vid.read()

    # Display the resulting frame
    cv2.imshow("frame", frame)

    # press any key to exit
    if cv2.waitKey(50) > 0:
        cv2.imwrite(filename(), frame, [cv2.IMWRITE_JPEG_QUALITY, 100])
        break


# After the loop release the cap object
vid.release()
# Destroy all the windows
cv2.destroyAllWindows()
