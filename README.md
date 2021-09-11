A python script that takes a picture from you webcam when it detects a face.

# Installation

Execute the installation script with `./install.sh`.
This will create a virtual environment and install the requirements.
It will also create an executable bash script `capture-login.sh` in your home directory.
You can run this as a [startup script](https://www.howtogeek.com/686952/how-to-manage-startup-programs-on-ubuntu-linux/) to take your photo whenever you login.

# Face recognition

`sudo apt-get install build-essential cmake libopenblas-dev liblapack-dev libx11-dev libgtk-3-dev`

# Run from commandline
Activate the virtualenv with `source venv/bin/activate`.
Run the script with `python detcap.py`.
This will open a fullscreen window and take a picture when a face is detected. The picture will be stored in `~/Pictures/login-capture/` with the username and current date and time as the name.

# Background

I used [howdy](https://github.com/boltgolt/howdy) to have a facial rcognition login.
With this tool you can capture a picture on every login which can be used to train the howdy models.
It may also be fun to [create a video how you look over time](https://www.youtube.com/watch?v=wAIZ36GI4p8)
