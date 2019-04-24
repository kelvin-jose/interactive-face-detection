from __future__ import print_function
import sys
import os
import cv2
import time
import logging as log
import numpy as np
from imutils.video import WebcamVideoStream
from openvino.inference_engine import IENetwork, IEPlugin

def load_xml_bin(model_xml, model_bin):

    # Plugin initialization for specified device and load extensions library if specified
    plugin = IEPlugin(device='CPU')
    plugin.add_cpu_extension('lib/libcpu_extension_sse4.so')
    # Read IR
    log.info("Reading IR...")
    net = IENetwork(model=model_xml, weights=model_bin)

    if plugin.device == "CPU":
        supported_layers = plugin.get_supported_layers(net)
        not_supported_layers = [l for l in net.layers.keys() if l not in supported_layers]
        if len(not_supported_layers) != 0:
            sys.exit(1)

    input_blob = next(iter(net.inputs))
    out_blob = next(iter(net.outputs))
    log.info("Loading IR to the plugin...")
    exec_net = plugin.load(network=net, num_requests=2)
    # Read and pre-process input image
    n, c, h, w = net.inputs[input_blob].shape
    del net
    return exec_net, n, c, h, w, input_blob, out_blob

def main():

    cur_request_id = 0

    face_detection_net, n, c, h, w, input_blob, out_blob = load_xml_bin('include/face-detection-adas-0001.xml', 'include/face-detection-adas-0001.bin')
    age_gender_net, a_n, a_c, a_h, a_w, a_input_blob, a_out_blob = load_xml_bin(
        'include/age-gender-recognition-retail-0013.xml', 'include/age-gender-recognition-retail-0013.bin')
    emotion_net, e_n, e_c, e_h, e_w, e_input_blob, e_out_blob = load_xml_bin(
        'include/emotions-recognition-retail-0003.xml', 'include/emotions-recognition-retail-0003.bin')
    stream = WebcamVideoStream(src=0).start()

    initial_w = 640
    initial_h = 480

    while True:
        frame = stream.read()
        if frame is None:
            break
        else:
            # ------ face detection
            in_frame = cv2.resize(frame, (w, h))
            in_frame = in_frame.transpose((2, 0, 1))  # Change data layout from HWC to CHW
            in_frame = in_frame.reshape((n, c, h, w))
            face_detection_net.start_async(request_id=cur_request_id, inputs={input_blob: in_frame})
            if face_detection_net.requests[cur_request_id].wait(-1) == 0:
                face_detection_res = face_detection_net.requests[cur_request_id].outputs[out_blob]
                detected_faces = []
                for face_loc in face_detection_res[0][0]:
                    if face_loc[2] > 0.5:
                        xmin = abs(int(face_loc[3] * initial_w))
                        ymin = abs(int(face_loc[4] * initial_h))
                        xmax = abs(int(face_loc[5] * initial_w))
                        ymax = abs(int(face_loc[6] * initial_h))
                        detected_faces.append([ymin, ymax, xmin, xmax])

            for face in detected_faces:

                # ------ age - gender
                age_gender_frame = frame[face[0]:face[1], face[2]:face[3]]
                age_in_frame = cv2.resize(age_gender_frame, (a_w, a_h))
                age_in_frame = age_in_frame.transpose((2, 0, 1))  # Change data layout from HWC to CHW
                age_in_frame = age_in_frame.reshape((a_n, a_c, a_h, a_w))
                age_gender_net.start_async(request_id=cur_request_id, inputs={a_input_blob: age_in_frame})
                if age_gender_net.requests[cur_request_id].wait(-1) == 0:
                    age_gender_res = age_gender_net.requests[cur_request_id].outputs
                    gender = ['female', 'male'][np.argmax(age_gender_res['prob'])]
                    age = int(age_gender_res['age_conv3'][0][0][0] * 100)

                # ----- emotion
                age_gender_frame = frame[face[0]:face[1], face[2]:face[3]]
                emotion_in_frame = cv2.resize(age_gender_frame, (e_w, e_h))
                emotion_in_frame = emotion_in_frame.transpose((2, 0, 1))
                emotion_in_frame = emotion_in_frame.reshape(e_n, e_c, e_h, e_w)
                emotion_net.start_async(request_id=cur_request_id, inputs={e_input_blob: emotion_in_frame})
                if emotion_net.requests[cur_request_id].wait(-1) == 0:
                    emotion_res = emotion_net.requests[cur_request_id].outputs[e_out_blob]
                    emotion = ['neutral', 'happy', 'sad', 'surprise', 'anger'][np.argmax(emotion_res)]

                cv2.putText(frame, gender + ', ' + str(age), (face[1], face[0]), cv2.FONT_HERSHEY_DUPLEX, 1.0, (255, 255, 255), 1)
                cv2.putText(frame, emotion, (face[1], face[0] + 20), cv2.FONT_HERSHEY_DUPLEX, 1.0,
                            (255, 255, 255), 1)
                cv2.rectangle(frame, (face[2], face[0]), (face[3], face[1]), (255, 255, 255), 1)
        cv2.imshow('Video', frame)
        # Hit 'q' on the keyboard to quit!
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

if __name__ == '__main__':
    main()
