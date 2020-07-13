import cv2
import zmq
import json
import base64
import numpy as np

context = zmq.Context()
footage_socket = context.socket(zmq.SUB)

try:

    with open('_config.json') as config_file:
        cfg = json.load(config_file)

    footage_socket.bind('tcp://{ip}:{port}'.format(
        ip=str(cfg['ip']),
        port=str(cfg['port'])
    ))

except:

    ip = input('IP your camera [* = localhost]: ')
    port = input('Port your camera [default: 5555]: ')

    footage_socket.bind('tcp://{ip}:{port}'.format(
        ip=str(ip),
        port=str(port)
    ))

finally:

    footage_socket.setsockopt_string(zmq.SUBSCRIBE, np.unicode(''))

    while True:
        try:

            frame = footage_socket.recv_string()
            img = base64.b64decode(frame)
            npimg = np.fromstring(img, dtype=np.uint8)

            source = cv2.imdecode(npimg, 1)

            cv2.imshow("Stream", source)
            cv2.waitKey(1)

        except KeyboardInterrupt:
            cv2.destroyAllWindows()
            break
