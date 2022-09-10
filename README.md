 python script that takes a picture from you webcam when it detects or recognizes a face.
![Timelapse](https://raw.githubusercontent.com/cleder/login-picture/main/timelapse.gif)

# Installation

## Linux

Execute the installation script with `./install.sh`.
This will create a virtual environment and install the requirements.
It will also create an executable bash script `capture-login.sh` in your home directory.
You can run this as a [startup script](https://www.howtogeek.com/686952/how-to-manage-startup-programs-on-ubuntu-linux/) to take your photo whenever you login.

## Windows

Install Python (3.10) from the Windows store.
Open Power Shell and run `pip install opencv-python`
Create a shortcut on your Desktop, enter `python.exe C:\path\to\directory\detcap.py` as the command to run.
You can run this shortcut [on startup](https://superuser.com/questions/954950/run-a-script-on-start-up-on-windows-10) to take a photo when you login.


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

I used [howdy](https://github.com/boltgolt/howdy) to have a facial recognition login.
With this tool you can capture a picture on every login which can be used to train the howdy models.
It may also be fun to [create a video how you look over time](https://www.youtube.com/watch?v=wAIZ36GI4p8)

## Create time lapse of your pictures

I used [Face-Alignment](https://github.com/SajjadAemmi/Face-Alignment) as a preprocessor.

###  Animated GIF with imagemagic
The images were resized and converted to gray-scale with the `convert` command of imagemagick:
```
find /path/to/input/ -iname '*.jpg' -exec convert \{} -verbose -colorspace Gray -set filename:base "%[basename]" -resize 256\> "/path/to/output/%[filename:base].jpg" \;
```

Afterwards you can create an animated GIF in the `/path/to/output/` directory with:
```
 convert -delay 15 -loop 0 *.jpg timelapse.gif
```

### Video with ffmpeg
Alternatively you can create a video with ffmpeg:

```
ffmpeg -framerate 5 -pattern_type glob -i '*.jpg' -c:v libx264 -r 30 -vf format=gray,scale=256:256 output.mp4
```
<video src="https://github.com/cleder/login-picture/blob/develop/output.mp4?raw=true" width=256>
