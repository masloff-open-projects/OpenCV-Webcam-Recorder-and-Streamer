# OpenCV CCTV

A small software to create CCTV for single camera. Completely created on __Python 3.6__, can detect faces, people and movements, stream video. Recorded videos have an info plate on top of the screen, which you can change from the configuration variable. The system can recognize movement and record the moving object as it moves, or it can record video continuously.  The code also supports custom hooks. The software was tested on Raspberry Pi

<img align="center" width="100%" src="https://i.ibb.co/6FGZRv2/Screenshot-20200714-002543.png">

### How to set up a video server

##### How to isntall

Copy the repository to your Linux system. 

```
git clone https://github.com/iRTEX-Creative/OpenCV-Webcam-Recorder-and-Streamer.git
```

Browse to the repository folder.

```
cd OpenCV-Webcam-Recorder-and-Streamer
```

Install Python libraries

```
pip3 install cv2
pip3 install numpy
pip3 install flask

```

Start ```main.py```

```
python3 main.py
```

### Hooks
The system sends signals at every activity by hooking. You can hang your functions on these hooks.

```
def execute (e=None):
    # The body of your function
    pass

Hooks.set('Hook', lambda E: execute(E))
```

#### Table of filters

| Filter | Arg | Description|
| ------ | ------ |  ------ |
|on_frame_record | frame | | 
|on_frame_text | text  | |
|on_frame_motion_detect_record | frame | |
|on_socket_frame | base64.b64encode(frame) | |
|on_socket_frame_encoded | frame | |
|on_config | Configuration object | |
|on_reserve_videofile | Configuration object | |
 

#### Table of web requests

| Hook | Arg | Description|
| ------ | ------ |  ------ |
| on_face_detect | frame with face | Executing when the camera detect the face. |
| on_eye_detect | frame with eye | Executing when the camera detect the eye. |
| on_body_detect | frame with body | Executing when the camera detect the body. |
| on_body_upper_detect | frame with body | Executing when the camera detect the upper body. |
| on_body_lower_detect | frame with body | Executing when the camera detect the lower. |
| on_motion_detect | frame | Executing when the camera detect movement. |


#### Table of hooks

| Hook | Arg | Description|
| ------ | ------ |  ------ |
| on_face_detect | frame with face | Executing when the camera detect the face. |
| on_eye_detect | frame with eye | Executing when the camera detect the eye. |
| on_body_detect | frame with body | Executing when the camera detect the body. |
| on_body_upper_detect | frame with body | Executing when the camera detect the upper body. |
| on_body_lower_detect | frame with body | Executing when the camera detect the lower. |
| on_start_webserver | None | Executing when web server is ready started. |
| on_wait_camera | number camera in for | Executing when system wait camera. To be inside the iteration of the cycle, not outside it. |
| on_init | True | Executing when system started. |
| on_release | None | Executing when OpenCV 'cap' and video record release. |
| on_frame_send_to_server | JPG frame | Executing when web server send frame. |
| on_motion_detect | frame | Executing when the camera detect movement. |
| on_save_video | frame | Executing when the system write frame in video. To be inside the iteration of the cycle, not outside it. |
| on_before_write_frame | frame | Executing when the frame is ready to be recorded, but not yet recorded. |
| on_setup_cfg | Configuration object | Executing when configs ready to use. |
| on_frame_start | Time start frame | |
| on_videofile_created | File path | |
| on_reserve_videofile_created | File path | |
| on_reserve_videofile_not_created | File path | |
| on_exit | True | Executing when system exiting. |

### How to set up remote viewing

To use the camera remotely (in the local network), you must in the configuration file in the line "web_stream": false, change false to true.
Then reboot the server. In the same repository is a folder clients, it has a file ```view.py```, open it on the computer from which you want to view the camera.  You will be asked to enter the IP camera (you can get it in the control panel of your router, or via ```nmap```), port (5555 by default). If all data is entered correctly, the camera will open in a new window!

Either you can connect to the broadcast via a regular browser or VLC. As a streaming address, use ```http://IP-YOUR-PC:5000/video```. 

#### Table of server URLs
| URL | Description|
| ------ | ------ |
| tcp://0.0.0.0:5555 | Primary video stream. You can connect to it through the tuner client in the ``clients`` folder |
| http://0.0.0.0:5000/video | Secondary video stream Flask. You can connect to it from a VLC or browser. It works with only one client at a time, the second will not be able to connect if the first is watching the broadcast |
| http://0.0.0.0:5000/records | Page for downloading video recordings |
---
![Git size](https://img.shields.io/github/languages/code-size/iRTEX-Creative/OpenCV-Webcam-Recorder-and-Streamer)
![GitHub All Releases](https://img.shields.io/github/downloads/iRTEX-Creative/OpenCV-Webcam-Recorder-and-Streamer/total)
