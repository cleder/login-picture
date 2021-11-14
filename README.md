A python script that takes a picture from you webcam when it detects a face.

# Installation

Execute the installation script with `./install.sh`.
This will create a virtual environment and install the requirements.
It will also create an executable bash script `capture-login.sh` in your home directory.
You can run this as a [startup script](https://www.howtogeek.com/686952/how-to-manage-startup-programs-on-ubuntu-linux/) to take your photo whenever you login.

# Face detection

The script `capture-login.sh` will terminate and write a picture of you into the directory `~/Pictures/login-capture` once it detected
a face looking into the webcam.

# Run from commandline
Activate the virtualenv with `source venv/bin/activate`.
Run the script with `python detcap.py`.
This will open a fullscreen window and take a picture when a face is detected. The picture will be stored in `~/Pictures/login-capture/` with the username and current date and time as the name.

# Face recognition

## Required libraries for ubuntu

`sudo apt-get install build-essential cmake libopenblas-dev liblapack-dev libx11-dev libgtk-3-dev`

## Training

Once you have captured a decent amount of pictures you can compute face embedding vectors for your images with `python encode_images.py`.
This will write an encoding file next to the original captured image of yor face with the extension `.cnn-encoded.pickle`.
No embeddings will be written for pictures where more than one face was detected.

## Recognize your face on login

Test the recognizer with `python recognize_me.py` and see how it performs.
It will save the recognized image along with the embedding once your face was recognized.
You may want to fine tune the `tolerance` parameter in `recognize_me.py`.

Run `recognize-login.sh` on startup, the window will close once your face has been recognized.
The picture will be stored with its encoding in `~/Pictures/login-capture/` and will be used for future recognitions.


# Background

I used [howdy](https://github.com/boltgolt/howdy) to have a facial rcognition login.
With this tool you can capture a picture on every login which can be used to train the howdy models.
It may also be fun to [create a video how you look over time](https://www.youtube.com/watch?v=wAIZ36GI4p8)

## Create time lapse of your pictures

I recommend to use [Face-Alignment](https://github.com/SajjadAemmi/Face-Alignment) as a preprocessor.

To resize your images you can use the `convert` command of imagemagick:
```
find /path/to/input/ -iname '*.jpg' -exec convert \{} -verbose -colorspace Gray -set filename:base "%[basename]" -resize 256\> "path/to/output/%[filename:base].jpg" \;
```

From the resized images you can create an animated GIF or a video.
