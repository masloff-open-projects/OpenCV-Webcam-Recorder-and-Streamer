# Software for creating a video server
# for home video surveillance.
#
# version: 1.0.0
# author: iRTEX Creative

from flask import Flask, render_template, Response
from libs.hooks import hooks
from libs.log import log
from user_hooks import _user_hooks as uh
from PIL import Image

import datetime
import numpy as np
import sys
import cv2
import os
import time
import base64
import zmq
import json
import glob
import threading

def main(argv):

    # Log
    Log.info('OK', 'System startup')

    # Try open web server
    if cfg['web_stream'] == True:

        Log.info('WAIT', 'Try open socket server')

        context = zmq.Context()
        footage_socket = context.socket(zmq.PUB)
        footage_socket.connect(cfg['web_ip'])

        Log.info('OK', 'Socket server is started')
        Log.info('OK', 'Server: {ip}'.format(ip=str(cfg['web_ip'])))

        Hooks.call('on_start_webserver', None)

    # Create cap
    cap = cv2.VideoCapture(0)

    # Mount camera
    Log.info('WAIT', 'Wait camera')

    while (True):

        for i in range(0, 3):

            Hooks.call('on_wait_camera', i)

            cap = cv2.VideoCapture(i)

            # Main script
            if cap.isOpened():

                Log.info('OK', 'Camera is connected')

                if (cfg['fourcc'] == 'MJPG'):
                    fourcc = cv2.VideoWriter_fourcc(*'MJPG')

                elif (cfg['fourcc'] == 'MP4V'):
                    fourcc = cv2.VideoWriter_fourcc(*'mp4v')

                else:
                    fourcc = cv2.VideoWriter_fourcc(*'XVID')


                video = cv2.VideoWriter(cfg['record_file_mask'].format(
                    time=(datetime.datetime.now()).strftime(cfg['time_mask']),
                    isotme=(datetime.datetime.now()).isoformat(),
                    count=str(len(glob.glob('{path}/*'.format(
                        path=os.path.dirname(os.path.abspath(cfg['record_file_mask']))
                    )))),
                    d=(datetime.datetime.now()).strftime("%d"),
                    m=(datetime.datetime.now()).strftime("%m"),
                    Y=(datetime.datetime.now()).strftime("%Y"),
                    H=(datetime.datetime.now()).strftime("%H"),
                    M=(datetime.datetime.now()).strftime("%M"),
                    S=(datetime.datetime.now()).strftime("%S"),
                ), fourcc, cfg['recorder_fps'], (int(cap.get(3)), int(cap.get(4))))

                # Create mask object
                _fgbg = cv2.createBackgroundSubtractorMOG2(300, 400, True)

                # Set path
                cascade_path_face = cv2.data.haarcascades + 'haarcascade_frontalface_alt2.xml'
                cascade_path_eye = cv2.data.haarcascades + 'haarcascade_eye_tree_eyeglasses.xml'
                cascade_path_body = cv2.data.haarcascades + 'haarcascade_fullbody.xml'
                cascade_path_upper = cv2.data.haarcascades + 'haarcascade_upperbody.xml'
                cascade_path_lower = cv2.data.haarcascades + 'haarcascade_lowerbody.xml'

                # Init cascade
                face_cascade = cv2.CascadeClassifier(cascade_path_face)
                eye_cascade = cv2.CascadeClassifier(cascade_path_eye)
                body_cascade = cv2.CascadeClassifier(cascade_path_body)
                upper_cascade = cv2.CascadeClassifier(cascade_path_upper)
                lower_cascade = cv2.CascadeClassifier(cascade_path_lower)

                # Try open Flask server
                def app ():

                    if cfg["flask_server"] == True:

                        app = Flask(__name__)

                        def __stream__():

                            Log.info('STREAM', 'Start stream')

                            context = zmq.Context()
                            footage_socket = context.socket(zmq.SUB)

                            Log.info('STREAM', 'Connect to socket server: {ip}'.format(ip=cfg['web_ip']))

                            try:

                                footage_socket.bind(cfg['web_ip'])

                                footage_socket.setsockopt_string(zmq.SUBSCRIBE, np.unicode(''))

                                Log.info('STREAM', 'Connected success')

                                footage_socket.setsockopt_string(zmq.SUBSCRIBE, np.unicode(''))

                                while True:

                                    frame = footage_socket.recv_string()
                                    img = base64.b64decode(frame)

                                    yield (b'--frame\r\n'
                                           b'Content-Type: image/jpeg\r\n\r\n' + img + b'\r\n\r\n')

                                footage_socket.close()

                            except: pass

                        @app.route('/video')
                        def __main__():
                            return Response(__stream__(), mimetype='multipart/x-mixed-replace; boundary=frame')

                        if __name__ == '__main__':
                            app.run(host='0.0.0.0', port=int(cfg['flask_port']))

                threading.Thread(target=app).start()

                while True:

                    ret, frame = cap.read()

                    if ret:

                        _frame_text = "{value}\n".format(value=cfg['text_on_frame'])

                        #_gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                        if (not cfg['video_zip'] == False):
                            frame = cv2.resize(frame, (0, 0), fx=float(cfg['video_zip']), fy=float(cfg['video_zip']))

                        # Get the foreground mask
                        _fgmask = _fgbg.apply(frame)

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

                            bodies = body_cascade.detectMultiScale(frame, 1.3, 5)
                            upper = upper_cascade.detectMultiScale(frame, 1.3, 5)
                            lower = lower_cascade.detectMultiScale(frame, 1.3, 5)

                            for (x, y, w, h) in bodies:

                                cv2.rectangle(frame, (x, y), (x + w, y + h), cfg['detect_body_boxcolor'], 1)

                                if cfg['detect_text_labels'] == True:
                                    cv2.putText(frame, 'Body', (x, y - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.4,
                                                cfg['detect_text_labels_color'])

                                Hooks.call('on_body_detect', frame[y:y + h, x:x + w])

                            for (x, y, w, h) in upper:

                                cv2.rectangle(frame, (x, y), (x + w, y + h), cfg['detect_body_boxcolor'], 1)

                                if cfg['detect_text_labels'] == True:
                                    cv2.putText(frame, 'Body Upper', (x, y - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.4,
                                                cfg['detect_text_labels_color'])

                                Hooks.call('on_body_upper_detect', frame[y:y + h, x:x + w])

                            for (x, y, w, h) in lower:

                                cv2.rectangle(frame, (x, y), (x + w, y + h), cfg['detect_body_boxcolor'], 1)

                                if cfg['detect_text_labels'] == True:
                                    cv2.putText(frame, 'Body Lower', (x, y - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.4,
                                                cfg['detect_text_labels_color'])

                                Hooks.call('on_body_lower_detect', frame[y:y + h, x:x + w])

                        if cfg['detect_face'] == True:

                            faces = face_cascade.detectMultiScale(
                                frame,
                                scaleFactor=1.2,
                                minNeighbors=5,
                                minSize=(8, 8)
                            )

                            for (x, y, w, h) in faces:

                                cv2.rectangle(frame, (x, y), (x + w, y + h), cfg['detect_face_boxcolor'], 1)

                                if cfg['detect_text_labels'] == True:
                                    cv2.putText(frame, 'Face', (x, y-8), cv2.FONT_HERSHEY_SIMPLEX, 0.4, cfg['detect_text_labels_color'])

                                Hooks.call('on_face_detect', frame[y:y + h, x:x + w])

                                if cfg['detect_eye'] == True:

                                    eye = eye_cascade.detectMultiScale(frame)

                                    for (x_, y_, w_, h_) in eye:

                                        cv2.rectangle(frame, (x_, y_), (x_ + w_, y_ + h_), cfg['detect_body_boxcolor'], 1)

                                        if cfg['detect_text_labels'] == True:
                                            cv2.putText(frame, 'Eye', (x_, y_ - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.4,
                                                        cfg['detect_text_labels_color'])

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
                            cv2.imshow("Video", frame)

                        key = cv2.waitKey(10)

                        if key == 27:
                            Hooks.call('on_exit', True)
                            break

                    else: break

                cv2.destroyAllWindows()

                cap.release()
                video.release()

                Hooks.call('on_release', None)

        time.sleep(1)

if __name__ == '__main__':

    # Create env
    Hooks = hooks()
    Log = log()

    # Init plugins
    uh(Hooks)

    # First hook
    Hooks.call('on_init', True)

    # Set config
    with open('config.json') as config_file:
        cfg = json.load(config_file)

    Hooks.call('on_setup_cfg', cfg)

    # Hooks.set('on_face_detect', lambda E: print('Face detect'))

    main(sys.argv)
