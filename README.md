A python script that takes a picture from you webcam when you press a key

# Installation

Execute the installation script with `./install.sh`.
This will create a virtual environment and install the requirements.
It will also create an executable bash script `capture-login.sh` in your home directory.
You can run this as a [startup script](https://www.howtogeek.com/686952/how-to-manage-startup-programs-on-ubuntu-linux/) to take you photo whenever you login.

# Run from commandline
Activate the virtualenv with `source venv/bin/activate`.
Run the script with `python detcap.py`.
This will open a fullscreen window and take a picture when you press any key. The picture will be stored in `~/Pictures/login-capture/` with the username and current date and time as the name.
