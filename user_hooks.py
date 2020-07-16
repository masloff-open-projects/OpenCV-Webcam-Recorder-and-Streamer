# This file is intended for your custom hooks.
# You can write all the necessary hooks to the _user_hooks function.
# It is called immediately after the delivery and announcement
# of all the necessary objects and before the system is connected to the camera

def _user_hooks(cv2=None, Hooks=None, Filters=None):

    # An example of writing Hello world text on a frame that will be written to a file
    #
    # def bigtext(frame):
    #     cv2.putText(frame, 'Hello world!', (50, 140), cv2.FONT_HERSHEY_SIMPLEX, 4, (255, 255, 255), 8)
    #     return frame
    #
    # Filters.set('on_frame_record', bigtext)



    # Add a new line to the frame text.
    # def append_text(text):
    #     return "{text}" \
    #            "New line".format(text=text)
    #
    # Filters.set('on_frame_text', append_text)



    # Save detected face
    # def save_detected_face(frame):
    #     cv2.imwrite('static/face.jpg', frame)
    #
    # Hooks.set('on_face_detect', save_detected_face)



    # Change the configuration directly in the program without changing config.json
    # def change_cfg(config):
    #     config['detect_face'] = True;
    #     return config
    #
    # Filters.set('on_config', change_cfg)


    # Create the third highest priority file to attempt recording
    # def on_reserve_videofile(file):
    #     return ""

    # Filters.set('on_reserve_videofile', on_reserve_videofile)

    pass

