# Software for creating a video server
# for home video surveillance.
#
# version: 1.0.0
# author: iRTEX Creative

from flask import Flask, render_template, Response
from imutils.object_detection import non_max_suppression
from libs.hooks import hooks
from libs.log import log
from libs.data import data
from libs.requests_ import requests_
from libs.filters import filters
from user_hooks import _user_hooks as uh
from random import randint

import imutils
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

    # Setup masks
    class mask:

        def __init__(self, data):
            self.data = data

        def file(self, string):

            return string.format(
                time=(datetime.datetime.now()).strftime(cfg['time_mask']),
                isotme=(datetime.datetime.now()).isoformat(),
                count=str(len(glob.glob('{path}/*'.format(
                    path=os.path.dirname(os.path.abspath('static/' + cfg['record_file_mask']))
                )))),
                d=(datetime.datetime.now()).strftime("%d"),
                m=(datetime.datetime.now()).strftime("%m"),
                Y=(datetime.datetime.now()).strftime("%Y"),
                H=(datetime.datetime.now()).strftime("%H"),
                M=(datetime.datetime.now()).strftime("%M"),
                S=(datetime.datetime.now()).strftime("%S"),
            )

        def fps(self, string):

            return string.format(
                max=str(self.data['max']),
                current=str(self.data['current'])
            )

        def server(self, string):

            return string.format(
                local='0.0.0.0',
                random_port=str(randint(49152,65535))
            )

    # Try open web server
    if cfg['web_stream'] == True:

        Log.info('WAIT', 'Try open socket server')

        context = zmq.Context()
        footage_socket = context.socket(zmq.PUB)
        footage_socket.connect(cfg['web_ip'])

        Log.info('OK', 'Socket server is started')
        Log.info('OK', 'Server: {ip}'.format(ip=str(mask({}).server(cfg['web_ip']))))

        Hooks.call('on_start_webserver', None)

    # Try open Flask server
    if cfg['flask_server'] == True:

        def app():

            app = Flask(__name__)

            def __stream__():

                global ready_frame

                Log.info('STREAM', 'Start stream')

                while True:

                    try:

                        ret, jpeg = cv2.imencode('.jpg', ready_frame)

                        yield (b'--frame\r\n'
                               b'Content-Type: image/jpeg\r\n\r\n' + jpeg.tobytes() + b'\r\n\r\n')

                    except:
                        pass

            @app.route('/video')
            def __main__():
                return Response(__stream__(), mimetype='multipart/x-mixed-replace; boundary=frame')

            @app.route('/records')
            def __records__():
                return render_template('records.html', videos=os.listdir('./static/'))

            if __name__ == '__main__':
                app.run(host=str(mask({}).server(cfg['flask_ip'])), port=int(mask({}).server(str(cfg['flask_port']))))

        threading.Thread(target=app).start()

    # Start window thread
    cv2.startWindowThread()

    # Start detector
    def detector():
        pass

    threading.Thread(target=detector).start()

    # NET
    if cfg['net_detect']['enabled'] == True:

        net = {
            "net": cv2.dnn.readNetFromCaffe('models/MobileNetSSD_deploy.prototxt.txt', 'models/MobileNetSSD_deploy.caffemodel'),
            "CLASSES": cfg['net_detect']['classes']
        }

    # Mount camera
    Log.info('WAIT', 'Wait camera')

    while (True):

        for i in range(0, 3):

            Hooks.call('on_wait_camera', i)

            if (cfg['stream'] == False):

                if (cfg['video'] == False): cap = cv2.VideoCapture(i)
                else: cv2.VideoCapture(cfg['video'])

            else: cap = cv2.VideoCapture(cfg['stream'])

            # Main script
            if cap.isOpened():

                Log.info('OK', 'Camera is connected')

                if (cfg['detect_people'] == True):
                    hog = cv2.HOGDescriptor()
                    hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())

                if (cfg['fourcc'] == 'MJPG'):
                    fourcc = cv2.VideoWriter_fourcc('M','J','P','G')

                elif (cfg['fourcc'] == 'FMP4'):
                    fourcc = cv2.VideoWriter_fourcc('F','M','P','4')

                elif (cfg['fourcc'] == 'MP4V'):
                    fourcc = cv2.VideoWriter_fourcc(*'mp4v')

                elif (cfg['fourcc'] == 'DIVX'):
                    fourcc = cv2.VideoWriter_fourcc(*'DIVX')

                elif (cfg['fourcc'] == 'iYUV'):
                    fourcc = cv2.VideoWriter_fourcc('i', 'Y', 'U', 'V')

                else:
                    fourcc = cv2.VideoWriter_fourcc(*'XVID')


                _video = cv2.VideoWriter(
                    (os.path.dirname(os.path.realpath(__file__))) + '/static/' + mask({}).file(cfg['record_file_mask']),
                    fourcc,
                    cfg['recorder_fps'],
                    (int(cap.get(3)), int(cap.get(4)))
                )

                if _video and os.path.isfile((os.path.dirname(os.path.realpath(__file__))) + '/static/' + mask({}).file(cfg['record_file_mask'])):

                    Log.info('OK', 'Video file has been created')
                    Hooks.call('on_videofile_created', (os.path.dirname(os.path.realpath(__file__))) + '/static/' + mask({}).file(cfg['record_file_mask']))

                else:

                    Log.info('ERROR', 'Folder for recording video is not available! Trying to record a video in an open folder')

                    _video = cv2.VideoWriter(
                        mask({}).file(cfg['record_file_reserve_mask']),
                        fourcc,
                        cfg['recorder_fps'],
                        (int(cap.get(3)), int(cap.get(4)))
                    )

                    if _video and os.path.isfile(mask({}).file(cfg['record_file_reserve_mask'])):

                        Log.info('OK', 'Video file has been created')
                        Hooks.call('on_reserve_videofile_created', mask({}).file(cfg['record_file_reserve_mask']))

                    else:

                        if (Filters.get('on_reserve_videofile')):

                            Log.info('INFO', 'Filter found for backup file. Try to use it as a name')

                            _video = cv2.VideoWriter(
                                mask({}).file(Filters.call('on_reserve_videofile', cfg['record_file_reserve_mask'])),
                                fourcc,
                                cfg['recorder_fps'],
                                (int(cap.get(3)), int(cap.get(4)))
                            )

                            if _video and os.path.isfile(mask({}).file(Filters.call('on_reserve_videofile', cfg['record_file_reserve_mask']))):

                                Log.info('OK', 'Video file has been created')
                                Hooks.call('on_reserve_videofile_created', mask({}).file(cfg['record_file_reserve_mask']))

                            else:

                                Log.info('ERROR', 'Video file was not created.')
                                Hooks.call('on_reserve_videofile_not_created', mask({}).file(cfg['record_file_reserve_mask']))

                        else:

                            Log.info('ERROR', 'Video file was not created.')
                            Hooks.call('on_reserve_videofile_not_created', mask({}).file(cfg['record_file_reserve_mask']))


                # Create mask object
                if cfg['motion_detect'] == True:
                    _fgbg = cv2.createBackgroundSubtractorMOG2(False)

                # Set path
                cascade_path_face = cv2.data.haarcascades + 'haarcascade_frontalface_alt2.xml'
                cascade_path_eye = cv2.data.haarcascades + 'haarcascade_eye_tree_eyeglasses.xml'
                cascade_path_body = cv2.data.haarcascades + 'haarcascade_fullbody.xml'
                cascade_path_upper = cv2.data.haarcascades + 'haarcascade_upperbody.xml'
                cascade_path_lower = cv2.data.haarcascades + 'haarcascade_lowerbody.xml'
                cascade_path_cars = 'models/haarcascade_cars3.xml'

                # Init cascade
                face_cascade = cv2.CascadeClassifier(cascade_path_face)
                eye_cascade = cv2.CascadeClassifier(cascade_path_eye)
                body_cascade = cv2.CascadeClassifier(cascade_path_body)
                upper_cascade = cv2.CascadeClassifier(cascade_path_upper)
                lower_cascade = cv2.CascadeClassifier(cascade_path_lower)
                cars_cascade = cv2.CascadeClassifier(cascade_path_cars)

                while True:

                    global ready_frame

                    # Time frame start
                    _time_frame_start = time.time()
                    Hooks.call('on_frame_start', _time_frame_start)

                    ret, frame = cap.read()

                    if ret:

                        # ZIP Frame
                        if (not cfg['video_zip'] == False):

                            if (cfg['video_zip'] == 'HD'):
                                frame = cv2.resize(frame, (int(1080), int(720)), interpolation=cv2.INTER_NEAREST)

                            elif (type(cfg['video_zip']) == type(0.1)):
                                frame = cv2.resize(frame, (0, 0), fx=cfg['video_zip'], fy=cfg['video_zip'])

                            else:
                                frame = cv2.resize(frame, (int(cfg['video_zip'][0]), int(cfg['video_zip'][1])), interpolation=cv2.INTER_NEAREST)

                        # Create children frames
                        frame_gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)

                        (_h, _w) = frame.shape[:2]
        
        
                        _frame_text = "{value}\n".format(value=cfg['text_on_frame'])

                        # Get the foreground mask
                        if cfg['motion_detect'] == True:
                            _fgmask = _fgbg.apply(frame_gray)

                            # Count all the non zero pixels within the mask
                            _count = np.count_nonzero(_fgmask)

                            if int(_count) > int(cfg['pixel_update_for_detect']):

                                if cfg['show_detect_motion_on_frame'] == True:
                                    _frame_text += "Detect motion! Update pixels: {value}\n".format(value=str(_count))

                        if cfg['show_time_on_frame'] == True:

                            d = datetime.datetime.now()
                            _frame_text += "{value}\n".format(value=d.strftime(cfg['time_mask']))

                        if cfg['detect_body'] == True or type(cfg['detect_body']) == type({}):

                            if (type(cfg['detect_body']) == type(True)) or (cfg['detect_body']['full'] == True): bodies = body_cascade.detectMultiScale(frame_gray, 1.3, 5)
                            if (type(cfg['detect_body']) == type(True)) or (cfg['detect_body']['upper'] == True): upper = upper_cascade.detectMultiScale(frame_gray, 1.3, 5)
                            if (type(cfg['detect_body']) == type(True)) or (cfg['detect_body']['lower'] == True): lower = lower_cascade.detectMultiScale(frame_gray, 1.3, 5)

                            if (type(cfg['detect_body']) == type({}) and cfg['detect_body']['full'] == True) or (type(cfg['detect_body']) == type(True)):

                                for (x, y, w, h) in bodies:

                                    cv2.rectangle(frame, (x, y), (x + w, y + h), cfg['detect_body_boxcolor'], 1)

                                    if cfg['detect_text_labels'] == True:
                                        cv2.putText(frame, 'Body', (x, y - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.4, cfg['detect_text_labels_color'])

                                    if (cfg['zoom_body']) == True:
                                        frame = frame[y:y + h, x:x + w]

                                    Hooks.call('on_body_detect', frame[y:y + h, x:x + w])
                                    Request.call('on_body_detect', frame[y:y + h, x:x + w])

                            if (type(cfg['detect_body']) == type({}) and cfg['detect_body']['upper'] == True) or (type(cfg['detect_body']) == type(True)):

                                for (x, y, w, h) in upper:

                                    cv2.rectangle(frame, (x, y), (x + w, y + h), cfg['detect_body_boxcolor'], 1)

                                    if cfg['detect_text_labels'] == True:
                                        cv2.putText(frame, 'Body Upper', (x, y - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.4,
                                                    cfg['detect_text_labels_color'])

                                    Hooks.call('on_body_upper_detect', frame[y:y + h, x:x + w])
                                    Request.call('on_body_upper_detect', frame[y:y + h, x:x + w])

                            if (type(cfg['detect_body']) == type({}) and cfg['detect_body']['lower'] == True) or (type(cfg['detect_body']) == type(True)):

                                for (x, y, w, h) in lower:

                                    cv2.rectangle(frame, (x, y), (x + w, y + h), cfg['detect_body_boxcolor'], 1)

                                    if cfg['detect_text_labels'] == True:
                                        cv2.putText(frame, 'Body Lower', (x, y - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.4,
                                                    cfg['detect_text_labels_color'])

                                    Hooks.call('on_body_lower_detect', frame[y:y + h, x:x + w])
                                    Request.call('on_body_lower_detect', frame[y:y + h, x:x + w])

                        if cfg['detect_people'] == True:

                            (rects, weights) = hog.detectMultiScale(frame_gray)

                            rects = np.array([[x, y, x + w, y + h] for (x, y, w, h) in rects])
                            pick = non_max_suppression(rects, probs=None, overlapThresh=0.65)

                            for (xA, yA, xB, yB) in pick:

                                cv2.rectangle(frame, (xA, yA), (xB, yB), cfg['detect_people_boxcolor'], 1)

                                if cfg['detect_text_labels'] == True:
                                    cv2.putText(frame, 'Human', (xA, yA - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.4, cfg['detect_text_labels_color'])

                                if cfg['zoom_people'] == True:
                                    if len(pick) == 1:
                                        frame = frame[yA:yA + yB, xA:xA + xB]

                        if cfg['detect_face'] == True:

                            faces = face_cascade.detectMultiScale(frame_gray)

                            for (x, y, w, h) in faces:

                                cv2.rectangle(frame, (x, y), (x + w, y + h), cfg['detect_face_boxcolor'], 1)

                                if cfg['detect_text_labels'] == True:
                                    cv2.putText(frame, 'Face', (x, y - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.4,
                                                cfg['detect_text_labels_color'])

                                Hooks.call('on_face_detect', frame[y:y + h, x:x + w])
                                Request.call('on_face_detect', frame[y:y + h, x:x + w])

                                if cfg['detect_eye'] == True:

                                    eye = eye_cascade.detectMultiScale(frame_gray)

                                    for (x_, y_, w_, h_) in eye:

                                        cv2.rectangle(frame, (x_, y_), (x_ + w_, y_ + h_),
                                                      cfg['detect_body_boxcolor'], 1)

                                        if cfg['detect_text_labels'] == True:
                                            cv2.putText(frame, 'Eye', (x_, y_ - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.4,
                                                        cfg['detect_text_labels_color'])

                                        Hooks.call('on_eye_detect', frame[y_:y_ + h_, x_:x_ + w_])
                                        Request.call('on_eye_detect', frame[y_:y_ + h_, x_:x_ + w_])

                        if cfg['detect_car'] == True:

                            cars = cars_cascade.detectMultiScale(frame_gray)

                            for (x, y, w, h) in cars:

                                cv2.rectangle(frame, (x, y), (x + w, y + h), cfg['detect_car_boxcolor'], 1)

                                if cfg['detect_text_labels'] == True:
                                    cv2.putText(frame, 'Car', (x, y - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.4, cfg['detect_text_labels_color'])

                                Hooks.call('on_car_detect', frame[y:y + h, x:x + w])
                                Request.call('on_car_detect', frame[y:y + h, x:x + w])

                        if cfg['net_detect']['enabled'] == True:

                            blob = cv2.dnn.blobFromImage(cv2.resize(frame, (300, 300)),
                                                         0.007843, (300, 300), 127.5)
                            net['net'].setInput(blob)
                            detections = net['net'].forward()

                            for i in np.arange(0, detections.shape[2]):
                                confidence = detections[0, 0, i, 2]
                                if confidence > 0:
                                    idx = int(detections[0, 0, i, 1])
                                    box = detections[0, 0, i, 3:7] * np.array([_w, _h, _w, _h])
                                    (startX, startY, endX, endY) = box.astype("int")
                                    label = "{}: {:.2f}%".format(net['CLASSES'][idx], confidence * 100)
                                    cv2.rectangle(frame, (startX, startY), (endX, endY), cfg['net_detect']['boxcolor'], 1)
                                    y = startY - 15 if startY - 15 > 15 else startY + 15
                                    cv2.putText(frame, label, (startX, y), cv2.FONT_HERSHEY_SIMPLEX, 0.4, cfg['net_detect']['boxcolor'], 1)

                                    Hooks.call("on_net_{}_detect".format(net['CLASSES'][idx]), frame[startY:startY + _h, startX:startX + _w])

                        if cfg['show_fps_on_frame'] == True:

                            fps = 60 / (time.time() - _time_frame_start)

                            _frame_text += "{value}\n".format(value=mask({
                                "max": str(cap.get(cv2.CAP_PROP_FPS)),
                                "current": str(round(fps, 1))
                            }).fps(cfg['fps_mask']))

                            # _frame_text += "{value}\n".format(value=)

                        # Put text
                        y0, dy = 13 * 2, 13 * 2
                        for i, line in enumerate(Filters.call('on_frame_text', _frame_text).split('\n')):
                            y = y0 + i * dy
                            cv2.putText(frame, line, (13, y), cv2.FONT_HERSHEY_SIMPLEX, cfg['text_on_frame_size'], cfg['text_on_frame_color'])

                        Hooks.call('on_before_write_frame', frame)

                        # Save video
                        if (cfg['save_video']) == True:
                            _video.write(Filters.call('on_frame_record', frame))
                            Hooks.call('on_save_video', frame)

                        # Motion detect
                        else:
                            if (cfg['save_video_on_movies'] == True) and cfg['motion_detect'] == True:
                                if int(_count) > int(cfg['pixel_update_for_detect']):
                                    _video.write(Filters.call('on_frame_motion_detect_record', frame))
                                    Hooks.call('on_motion_detect', frame)
                                    Hooks.call('on_save_video', frame)
                                    Request.call('on_motion_detect', frame)

                        # Send picture to server
                        if cfg['web_stream'] == True:

                            encoded, buffer = cv2.imencode('.jpg', Filters.call('on_socket_frame', frame))

                            jpg_as_text = base64.b64encode(buffer)
                            footage_socket.send(Filters.call('on_socket_frame_encoded', jpg_as_text))

                            Hooks.call('on_frame_send_to_server', jpg_as_text)

                        # Setup ready frame
                        ready_frame = frame

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
                _video.release()

                Hooks.call('on_release', None)

        # Send noise to video server
        im = np.empty((720, 1080), np.uint8)
        ready_frame = cv2.randn(im, (0), (99))


if __name__ == '__main__':

    # Create env
    global ready_frame
    ready_frame = False
    Hooks = hooks()
    Log = log()
    Filters = filters()
    Request = requests_()

    # Init plugins
    uh(cv2, Hooks, Filters)

    # First hook
    Hooks.call('on_init', True)

    # Set config
    with open('config.json') as config_file:
        cfg = Filters.call('on_config', json.load(config_file))

    Hooks.call('on_setup_cfg', cfg)

    main(sys.argv)
