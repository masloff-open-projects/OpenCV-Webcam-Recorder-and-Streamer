# Software for creating a video server
# for home video surveillance.
#
# Do not use this for commercial purposes
#
# version: 1.0.0
# author: iRTEX Creative

from libs.hooks import hooks
import datetime
import numpy as np
import sys
import cv2
import os
import time
import base64
import zmq
import json

def main(argv):

    # Try open web server
    if cfg['web_stream'] == True:
        context = zmq.Context()
        footage_socket = context.socket(zmq.PUB)
        footage_socket.connect('tcp://localhost:5555')

        Hooks.call('on_start_webserver', None)

    # Create cap
    cap = cv2.VideoCapture(0)

    # Mount camera
    while (not cap.isOpened()):
        time.sleep(1)
        for i in range(0, 3):

            Hooks.call('on_wait_camera', i)

            cap = cv2.VideoCapture(i)

            # Main script
            if cap.isOpened():

                video = cv2.VideoWriter(cfg['record_file_mask'].format(
                    time=(datetime.datetime.now()).strftime(cfg['time_mask']),
                    isotme=(datetime.datetime.now()).isoformat(),
                ), cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'), cfg['recorder_fps'], (int(cap.get(3)), int(cap.get(4))))

                # Create mask object
                _fgbg = cv2.createBackgroundSubtractorMOG2(300, 400, True)

                # Set path
                cascade_path_face = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
                cascade_path_eye = cv2.data.haarcascades + 'haarcascade_eye.xml'
                cascade_path_body = cv2.data.haarcascades + 'haarcascade_fullbody.xml'
                cascade_path_upper = cv2.data.haarcascades + 'haarcascade_upperbody.xml'
                cascade_path_lower = cv2.data.haarcascades + 'haarcascade_lowerbody.xml'

                # Init cascade
                face_cascade = cv2.CascadeClassifier(cascade_path_face)
                eye_cascade = cv2.CascadeClassifier(cascade_path_eye)
                body_cascade = cv2.CascadeClassifier(cascade_path_body)
                upper_cascade = cv2.CascadeClassifier(cascade_path_upper)
                lower_cascade = cv2.CascadeClassifier(cascade_path_lower)

                while True:

                    ret, frame = cap.read()

                    if ret:

                        _frame_text = "{value}\n".format(value=cfg['text_on_frame'])

                        _gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

                        _min_frame = cv2.resize(frame, (0, 0), fx=0.50, fy=0.50)

                        # Get the foreground mask
                        _fgmask = _fgbg.apply(_min_frame)

                        # Count all the non zero pixels within the mask
                        _count = np.count_nonzero(_fgmask)

                        if int(_count) > int(cfg['pixel_update_for_detect']):

                            if cfg['show_detect_motion_on_frame'] == True:
                                _frame_text += "Detect motion! Update pixels: {value}\n".format(value=str(_count))

                        if cfg['show_time_on_frame'] == True:

                            d = datetime.datetime.now()
                            _frame_text += "{value}\n".format(value=d.strftime(cfg['time_mask']))

                        if cfg['show_fps_on_frame'] == True:

                            _frame_text += "{value}\n".format(value=cfg['fps_mask'].format(value=str(cap.get(cv2.CAP_PROP_FPS))))
                            # _frame_text += "{value}\n".format(value=)

                        if cfg['detect_body'] == True:

                            bodies = body_cascade.detectMultiScale(_gray_frame, 1.3, 5)
                            upper = upper_cascade.detectMultiScale(_gray_frame, 1.3, 5)
                            lower = lower_cascade.detectMultiScale(_gray_frame, 1.3, 5)

                            for (x, y, w, h) in bodies:

                                cv2.rectangle(frame, (x, y), (x + w, y + h), (121, 184, 61), 2)
                                Hooks.call('on_body_detect', frame[y:y + h, x:x + w])

                            for (x, y, w, h) in upper:

                                cv2.rectangle(frame, (x, y), (x + w, y + h), (121, 184, 61), 2)
                                Hooks.call('on_body_upper_detect', frame[y:y + h, x:x + w])

                            for (x, y, w, h) in lower:

                                cv2.rectangle(frame, (x, y), (x + w, y + h), (121, 184, 61), 2)
                                Hooks.call('on_body_lower_detect', frame[y:y + h, x:x + w])

                        if cfg['detect_face'] == True:

                            faces = face_cascade.detectMultiScale(
                                _gray_frame,
                                scaleFactor=1.2,
                                minNeighbors=5,
                                minSize=(8, 8)
                            )

                            for (x, y, w, h) in faces:

                                cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
                                Hooks.call('on_face_detect', frame[y:y + h, x:x + w])

                                if cfg['detect_eye'] == True:

                                    eye = eye_cascade.detectMultiScale(
                                        frame[y:y + h, x:x + w],
                                        scaleFactor=1.2,
                                        minNeighbors=5,
                                        minSize=(8, 8)
                                    )

                                    for (x_, y_, w_, h_) in eye:

                                        cv2.rectangle(frame, (x_, y_), (x_ + w_, y_ + h_), (255, 177, 26), 8)
                                        Hooks.call('on_eye_detect', frame[y_:y_ + h_, x_:x_ + w_])

                        # Put text
                        y0, dy = 13 * 2, 13 * 2
                        for i, line in enumerate(_frame_text.split('\n')):
                            y = y0 + i * dy
                            cv2.putText(frame, line, (13, y), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255))

                        Hooks.call('on_before_write_frame', frame)

                        # Save video
                        if (cfg['save_video']) == True:
                            video.write(frame)
                            Hooks.call('on_save_video', frame)

                        # Motion detect
                        else:
                            if (cfg['save_video_on_movies'] == True):
                                if int(_count) > int(cfg['pixel_update_for_detect']):
                                    video.write(frame)
                                    Hooks.call('on_motion_detect', frame)

                        # Send picture to server
                        if cfg['web_stream'] == True:
                            encoded, buffer = cv2.imencode('.jpg', frame)
                            jpg_as_text = base64.b64encode(buffer)
                            footage_socket.send(jpg_as_text)

                            Hooks.call('on_frame_send_to_server', jpg_as_text)

                        # Show picture on PC
                        if cfg['show_stream'] == True:
                            cv2.imshow("_min_frame_", frame)

                        key = cv2.waitKey(10)

                        if key == 27:
                            Hooks.call('on_exit', True)
                            break

                    else: break

                cv2.destroyAllWindows()

                cap.release()
                video.release()

                Hooks.call('on_release', None)

if __name__ == '__main__':

    # Create env
    Hooks = hooks()

    # First hook
    Hooks.call('on_init', True)

    # Set config
    with open('config.json') as config_file:
        cfg = json.load(config_file)

    Hooks.call('on_setup_cfg', cfg)

    # Hooks.set('on_face_detect', lambda E: print('Face detect'))

    main(sys.argv)
