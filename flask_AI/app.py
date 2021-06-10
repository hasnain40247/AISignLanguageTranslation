from flask import Flask, render_template, Response
import cv2
import os
import pyttsx3
import numpy as np
import time
from object_detection.utils import label_map_util
from object_detection.utils import visualization_utils as viz_utils
from object_detection.builders import model_builder
from object_detection.utils import config_util
from object_detection.protos import pipeline_pb2
from google.protobuf import text_format
import tensorflow as tf
from flask_executor import Executor
app = Flask(__name__)
executor = Executor(app)

# -----------------------------------------------------PHRASES MODEL PROCESSING--------------------------------------------------------------
category_index = {1: {'id': 1, 'name': 'Goodbye'},
                  2: {'id': 2, 'name': 'Hello'},
                  3: {'id': 3, 'name': 'Love'},
                  4: {'id': 4, 'name': 'No'},
                  5: {'id': 5, 'name': 'Please'},
                  6: {'id': 6, 'name': 'Sad'},
                  7: {'id': 7, 'name': 'Sorry'},
                  8: {'id': 8, 'name': 'Thanks'},
                  9: {'id': 9, 'name': 'Welcome'},
                  10: {'id': 10, 'name': 'Where'},
                  11: {'id': 11, 'name': 'Yes'}}
category = []
cat = []
category_final = []

category_index2 = {1: {'id': 1, 'name': 'A'},
                   2: {'id': 2, 'name': 'B'},
                   3: {'id': 3, 'name': 'C'},
                   4: {'id': 4, 'name': 'D'},
                   5: {'id': 5, 'name': 'E'},
                   6: {'id': 6, 'name': 'F'},
                   7: {'id': 7, 'name': 'G'},
                   8: {'id': 8, 'name': 'H'},
                   9: {'id': 9, 'name': 'I'},
                   10: {'id': 10, 'name': 'J'},
                   11: {'id': 11, 'name': 'K'},
                   12: {'id': 12, 'name': 'L'},
                   13: {'id': 13, 'name': 'M'},
                   14: {'id': 14, 'name': 'N'},
                   15: {'id': 15, 'name': 'O'},
                   16: {'id': 16, 'name': 'P'},
                   17: {'id': 17, 'name': 'Q'},
                   18: {'id': 18, 'name': 'R'},
                   19: {'id': 19, 'name': 'S'},
                   20: {'id': 20, 'name': 'T'},
                   21: {'id': 21, 'name': 'U'},
                   22: {'id': 22, 'name': 'V'},
                   23: {'id': 23, 'name': 'W'},
                   24: {'id': 24, 'name': 'X'},
                   25: {'id': 25, 'name': 'Y'}}
category2 = []
cat2 = []
category_final2 = []
camera = cv2.VideoCapture(0)
width = int(camera.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(camera.get(cv2.CAP_PROP_FRAME_HEIGHT))


def text_to_speech(para):
    time.sleep(2)
    print(para)
    engine = pyttsx3.init()
    voices = engine.getProperty('voices')
    engine.setProperty("rate", 178)
    engine.setProperty('voice', voices[1].id)
    engine.say(para)
    engine.runAndWait()

    # time.sleep(5)
    # li = list(set(category_final))

    # # voice_dict = {'Male': 0, 'Female': 1}
    # # code = voice_dict[gender]

    # # engine = pyttsx3.init()

    # # # Setting up voice rate
    # # engine.setProperty('rate', 125)

    # # # Setting up volume level  between 0 and 1
    # # engine.setProperty('volume', 0.8)

    # # # Change voices: 0 for male and 1 for female
    # # voices = engine.getProperty('voices')
    # # engine.setProperty('voice', voices[code].id)

    # for word in li:
    #     engine.say(word)
    #     engine.runAndWait()


def modelBuilder():
    config_file = 'C:/Users/hnsik/Pyth/CSE1014/Phrases/Tensorflow/workspace/models/my_ssd_mobnet/pipeline.config'
    check = 'C:/Users/hnsik/Pyth/CSE1014/Phrases/Tensorflow/workspace/models/my_ssd_mobnet'
    configs = config_util.get_configs_from_pipeline_file(config_file)
    detection_model = model_builder.build(
        model_config=configs['model'], is_training=False)

# Restore checkpoint
    ckpt = tf.compat.v2.train.Checkpoint(model=detection_model)
    ckpt.restore(os.path.join(check, 'ckpt-6')).expect_partial()
    return detection_model


detection_model = modelBuilder()


def detect_fn(image):

    image, shapes = detection_model.preprocess(image)
    prediction_dict = detection_model.predict(image, shapes)
    detections = detection_model.postprocess(prediction_dict, shapes)
    return detections


def gen_frames():
    print("INSIDE GEN FRAMES")

    while True:

        ret, frame = camera.read()
        print("RETS:", ret)
        print("FRAMES:", frame)
        if not ret:
            break
        else:
            print("INSIDE Else")

            image_np = np.array(frame)
            print("Image_np", image_np)

            input_tensor = tf.convert_to_tensor(
                np.expand_dims(image_np, 0), dtype=tf.float32)
            print("INPUT_TENSOR", input_tensor)
            detections = detect_fn(input_tensor)
            num_detections = int(detections.pop('num_detections'))
            print("Num_Detections:", num_detections)
            detections = {key: value[0, :num_detections].numpy()
                          for key, value in detections.items()}
            detections['num_detections'] = num_detections
            detections['detection_classes'] = detections['detection_classes'].astype(
                np.int64)
            print("detections['num_detections']", detections['num_detections'])
            print("detections['detection_classes']",
                  detections['detection_classes'])

            label_id_offset = 1
            image_np_with_detections = image_np.copy()

            viz_utils.visualize_boxes_and_labels_on_image_array(
                image_np_with_detections,
                detections['detection_boxes'],
                detections['detection_classes']+label_id_offset,
                detections['detection_scores'],
                category_index,
                use_normalized_coordinates=True,
                max_boxes_to_draw=5,
                min_score_thresh=.5,
                agnostic_mode=False)

            category.append([category_index.get(value) for index, value in enumerate(
                detections['detection_classes']+label_id_offset) if detections['detection_scores'][index] > 0.5])
            cat = list(filter(lambda x: x, category))
            # new_cat = []
            # for elem in cat:
            #     if elem not in new_cat:
            #         new_cat.append(elem)
            # cat = new_cat
            print(cat)
            for elem in cat:
                for k in elem:
                    if k is not None:

                        category_final.append(k['name'])

            cat = list(set(category_final))
            print(cat)

            # cv2.imshow('object detection',  cv2.resize(
            #     image_np_with_detections, (800, 600)))
            ret, buffer = cv2.imencode('.jpg',  image_np_with_detections)
            image_np_with_detections = buffer.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + image_np_with_detections + b'\r\n')
# -----------------------------------------------------PHRASES MODEL PROCESSING--------------------------------------------------------------


# -----------------------------------------------------ALPHABETS MODEL PROCESSING--------------------------------------------------------------
def modelBuilder2():
    config_file = 'C:/Users/hnsik/Pyth/CSE1014/Alphabets/Tensorflow/workspace/models/my_ssd_mobnet/pipeline.config'
    check = 'C:/Users/hnsik/Pyth/CSE1014/Alphabets/Tensorflow/workspace/models/my_ssd_mobnet'
    configs = config_util.get_configs_from_pipeline_file(config_file)
    detection_model = model_builder.build(
        model_config=configs['model'], is_training=False)
    ckpt = tf.compat.v2.train.Checkpoint(model=detection_model)
    ckpt.restore(os.path.join(check, 'ckpt-6')).expect_partial()
    return detection_model


detection_model2 = modelBuilder2()


def detect_fn2(image):

    image, shapes = detection_model2.preprocess(image)
    prediction_dict = detection_model2.predict(image, shapes)
    detections = detection_model2.postprocess(prediction_dict, shapes)
    return detections


def gen_frames2():
    print("INSIDE GEN FRAMES")

    while True:

        ret, frame = camera.read()
        print("RETS:", ret)
        print("FRAMES:", frame)
        if not ret:
            break
        else:
            print("INSIDE Else")

            image_np = np.array(frame)
            print("Image_np", image_np)

            input_tensor = tf.convert_to_tensor(
                np.expand_dims(image_np, 0), dtype=tf.float32)
            print("INPUT_TENSOR", input_tensor)
            detections = detect_fn2(input_tensor)
            num_detections = int(detections.pop('num_detections'))
            print("Num_Detections:", num_detections)
            detections = {key: value[0, :num_detections].numpy()
                          for key, value in detections.items()}
            detections['num_detections'] = num_detections
            detections['detection_classes'] = detections['detection_classes'].astype(
                np.int64)
            print("detections['num_detections']", detections['num_detections'])

            label_id_offset = 1
            print("detections['detection_classes']",
                  detections['detection_classes']+label_id_offset,)
            image_np_with_detections = image_np.copy()

            viz_utils.visualize_boxes_and_labels_on_image_array(
                image_np_with_detections,
                detections['detection_boxes'],
                detections['detection_classes']+label_id_offset,
                detections['detection_scores'],
                category_index2,
                use_normalized_coordinates=True,
                max_boxes_to_draw=5,
                min_score_thresh=.5,
                agnostic_mode=False)

            category2.append([category_index2.get(value) for index, value in enumerate(
                detections['detection_classes']+label_id_offset) if detections['detection_scores'][index] > 0.5])
            cat2 = list(filter(lambda x: x, category2))
            print(cat2)
            for elem in cat2:
                for k in elem:
                    if k is not None:

                        category_final2.append(k['name'])

            cat2 = list(set(category_final2))
            print(cat2)

            ret, buffer = cv2.imencode('.jpg',  image_np_with_detections)
            image_np_with_detections = buffer.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + image_np_with_detections + b'\r\n')

# --------------------------------------------------------------------------ROUTES---------------------------------------------------------------


@app.route('/video_feed')
def video_feed():
    return Response(gen_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')


@app.route('/video_feed2')
def video_feed2():
    return Response(gen_frames2(), mimetype='multipart/x-mixed-replace; boundary=frame')


@app.route('/')
def index():

    return render_template('main.html')


@app.route("/alp")
def hmus():
    return render_template("index.html")


@app.route("/phrases")
def hmu():

    return render_template("home.html")


@app.route("/tts")
def hu():
    executor.submit(text_to_speech(list(set(category_final))))
    return render_template("phrases.html", categ=list(set(category_final)))


@app.route("/atts")
def ahu():
    executor.submit(text_to_speech(list(set(category_final2))))
    return render_template("alphs.html", teg=list(set(category_final2)))


if __name__ == '__main__':
    app.run(debug=True)


# --------------------------------------------------------------------------ROUTES---------------------------------------------------------------
