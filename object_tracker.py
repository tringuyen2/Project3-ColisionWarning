from tools import generate_detections as gdet
from deep_sort.tracker import Tracker
from deep_sort.detection import Detection
from deep_sort import preprocessing, nn_matching
from tensorflow.compat.v1 import InteractiveSession
from tensorflow.compat.v1 import ConfigProto
import matplotlib.pyplot as plt
import numpy as np
import cv2
from PIL import Image
from core.config import cfg
from tensorflow.python.saved_model import tag_constants
from core.yolov4 import filter_boxes
import core.utils as utils
from absl.flags import FLAGS
from absl import app, flags, logging
import tensorflow as tf
import time
import os
from collections import deque
import math

# comment out below line to enable tensorflow logging outputs
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
physical_devices = tf.config.experimental.list_physical_devices('GPU')
if len(physical_devices) > 0:
    tf.config.experimental.set_memory_growth(physical_devices[0], True)
# deep sort imports
flags.DEFINE_string('framework', 'tf', '(tf, tflite, trt')
flags.DEFINE_string('weights', './checkpoints/yolov4-tiny-416',
                    'path to weights file')
flags.DEFINE_integer('size', 416, 'resize images to')
flags.DEFINE_boolean('tiny', True, 'yolo or yolo-tiny')
flags.DEFINE_string('model', 'yolov4', 'yolov3 or yolov4')
flags.DEFINE_string('video', './data/video/test4.mp4',
                    'path to input video or set to 0 for webcam')
flags.DEFINE_string('output', None, 'path to output video')
flags.DEFINE_string('output_format', 'XVID',
                    'codec used in VideoWriter when saving video to file')
flags.DEFINE_float('iou', 0.45, 'iou threshold')
flags.DEFINE_float('score', 0.50, 'score threshold')
flags.DEFINE_boolean('dont_show', False, 'dont show video output')
flags.DEFINE_boolean('info', False, 'show detailed info of tracked objects')
flags.DEFINE_boolean('count', False, 'count objects being tracked on screen')




def main(_argv):
    with open('focalLength.txt') as f:
        focal_length = float(f.read())

    real_car_height = 1.6
    real_truck_height = 3.5
    real_bus_height = 3.2
    real_motorbike_height = 1.2

    pts = [deque(maxlen=30) for _ in range(9999)]

    distances = [deque(maxlen=30) for _ in range(9999)]


    # Definition of the parameters
    max_cosine_distance = 0.4
    nn_budget = None
    nms_max_overlap = 1.0

    # initialize deep sort
    model_filename = 'model_data/mars-small128.pb'
    encoder = gdet.create_box_encoder(model_filename, batch_size=1)
    # calculate cosine distance metric
    metric = nn_matching.NearestNeighborDistanceMetric(
        "cosine", max_cosine_distance, nn_budget)
    # initialize tracker
    tracker = Tracker(metric)

    # load configuration for object detector
    config = ConfigProto()
    config.gpu_options.allow_growth = True
    session = InteractiveSession(config=config)
    STRIDES, ANCHORS, NUM_CLASS, XYSCALE = utils.load_config(FLAGS)
    input_size = FLAGS.size
    video_path = FLAGS.video

    # load tflite model if flag is set
    if FLAGS.framework == 'tflite':
        interpreter = tf.lite.Interpreter(model_path=FLAGS.weights)
        interpreter.allocate_tensors()
        input_details = interpreter.get_input_details()
        output_details = interpreter.get_output_details()
        print(input_details)
        print(output_details)
    # otherwise load standard tensorflow saved model
    else:
        saved_model_loaded = tf.saved_model.load(
            FLAGS.weights, tags=[tag_constants.SERVING])
        infer = saved_model_loaded.signatures['serving_default']

    # begin video capture
    try:
        vid = cv2.VideoCapture(int(video_path))
    except:
        vid = cv2.VideoCapture(video_path)

    out = None

    # L???y fps ????? t??nh v???n t???c
    (major_ver, minor_ver, subminor_ver) = (cv2.__version__).split('.')
    if int(major_ver) < 3:
        fps = vid.get(cv2.cv.CV_CAP_PROP_FPS)
        print(
            "Frames per second using video.get(cv2.cv.CV_CAP_PROP_FPS): {0}".format(fps))
    else:
        fps = vid.get(cv2.CAP_PROP_FPS)
        print(
            "Frames per second using video.get(cv2.CAP_PROP_FPS) : {0}".format(fps))

    # get video ready to save locally if flag is set
    if FLAGS.output:
        # by default VideoCapture returns float instead of int
        width = int(vid.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(vid.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = int(vid.get(cv2.CAP_PROP_FPS))
        codec = cv2.VideoWriter_fourcc(*FLAGS.output_format)
        out = cv2.VideoWriter(FLAGS.output, codec, fps, (width, height))

    frame_num = 0
    # while video is running
    while True:
        return_value, frame = vid.read()
        if return_value:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            image = Image.fromarray(frame)
        else:
            print('Video has ended or failed, try a different video format!')
            break
        frame_num += 1
        print('Frame #: ', frame_num)
        frame_size = frame.shape[:2]

        # Chu???n h??a ???nh
        image_data = cv2.resize(frame, (input_size, input_size))
        image_data = image_data / 255.
        image_data = image_data[np.newaxis, ...].astype(np.float32)
        # Th???i gian th???i ??i???m hi???n t???i
        start_time = time.time()

        # run detections on tflite if flag is set
        if FLAGS.framework == 'tflite':
            interpreter.set_tensor(input_details[0]['index'], image_data)
            interpreter.invoke()
            pred = [interpreter.get_tensor(
                output_details[i]['index']) for i in range(len(output_details))]
            # run detections using yolov3 if flag is set
            if FLAGS.model == 'yolov3' and FLAGS.tiny == True:
                boxes, pred_conf = filter_boxes(pred[1], pred[0], score_threshold=0.25,
                                                input_shape=tf.constant([input_size, input_size]))
            else:
                boxes, pred_conf = filter_boxes(pred[0], pred[1], score_threshold=0.25,
                                                input_shape=tf.constant([input_size, input_size]))
        else:
            batch_data = tf.constant(image_data)
            pred_bbox = infer(batch_data)
            for key, value in pred_bbox.items():
                boxes = value[:, :, 0:4]
                pred_conf = value[:, :, 4:]

        boxes, scores, classes, valid_detections = tf.image.combined_non_max_suppression(
            boxes=tf.reshape(boxes, (tf.shape(boxes)[0], -1, 1, 4)),
            scores=tf.reshape(
                pred_conf, (tf.shape(pred_conf)[0], -1, tf.shape(pred_conf)[-1])),
            max_output_size_per_class=50,
            max_total_size=50,
            iou_threshold=FLAGS.iou,
            score_threshold=FLAGS.score
        )

        # convert data to numpy arrays and slice out unused elements
        num_objects = valid_detections.numpy()[0]
        bboxes = boxes.numpy()[0]
        bboxes = bboxes[0:int(num_objects)]
        scores = scores.numpy()[0]
        scores = scores[0:int(num_objects)]
        classes = classes.numpy()[0]
        classes = classes[0:int(num_objects)]

        # format bounding boxes from normalized ymin, xmin, ymax, xmax ---> xmin, ymin, width, height
        original_h, original_w, _ = frame.shape
        bboxes = utils.format_boxes(bboxes, original_h, original_w)

        # store all predictions in one parameter for simplicity when calling functions
        pred_bbox = [bboxes, scores, classes, num_objects]

        # read in all class names from config
        class_names = utils.read_class_names(cfg.YOLO.CLASSES)

        # by default allow all classes in .names file
        # allowed_classes = list(class_names.values())

        # custom allowed classes (uncomment line below to customize tracker for only people)
        allowed_classes = ['car', 'truck', 'bus', 'motorbike']

        # loop through objects and use class index to get class name, allow only classes in allowed_classes list
        names = []
        deleted_indx = []
        for i in range(num_objects):
            class_indx = int(classes[i])
            class_name = class_names[class_indx]
            if class_name not in allowed_classes:
                deleted_indx.append(i)
            else:
                names.append(class_name)
        names = np.array(names)
        count = len(names)
        if FLAGS.count:
            cv2.putText(frame, "Objects being tracked: {}".format(
                count), (5, 35), cv2.FONT_HERSHEY_COMPLEX_SMALL, 2, (0, 255, 0), 2)
            print("Objects being tracked: {}".format(count))
        # delete detections that are not in allowed_classes
        bboxes = np.delete(bboxes, deleted_indx, axis=0)
        scores = np.delete(scores, deleted_indx, axis=0)

        # encode yolo detections and feed to tracker
        features = encoder(frame, bboxes)
        detections = [Detection(bbox, score, class_name, feature) for bbox,
                      score, class_name, feature in zip(bboxes, scores, names, features)]

        # initialize color map
        cmap = plt.get_cmap('tab20b')
        colors = [cmap(i)[:3] for i in np.linspace(0, 1, 20)]

        # run non-maxima supression
        boxs = np.array([d.tlwh for d in detections])
        scores = np.array([d.confidence for d in detections])
        classes = np.array([d.class_name for d in detections])
        indices = preprocessing.non_max_suppression(
            boxs, classes, nms_max_overlap, scores)
        detections = [detections[i] for i in indices]

        # Draw front area
        cv2.line(frame, (int(original_w/3), original_h),
                 (int(original_w/2), int(2*original_h/3)), (255, 64, 64), 1)
        cv2.line(frame, (int(2*original_w/3), original_h),
                 (int(original_w/2), int(2*original_h/3)), (255, 64, 64), 1)
        tan_front_area = (original_h/3) / (original_w/6)
        # print(f"tan_front_area = {tan_front_area}")

        print(f"Time detect: {time.time() - start_time}")
        # Call the tracker
        tracker.predict()
        tracker.update(detections)

        warn = 0
        # update tracks
        for track in tracker.tracks:
            if not track.is_confirmed() or track.time_since_update > 1:
                continue
            bbox = track.to_tlbr()
            class_name = track.get_class()

            color = colors[int(track.track_id) % len(colors)]
            color = [i * 255 for i in color]

            # Draw motion path
            center = (int(((bbox[0])+(bbox[2]))/2),
                      int(((bbox[1])+(bbox[3]))/2))

            pts[track.track_id].append(center)
            cv2.circle(frame,  (center), 1, color, 5)
            for j in range(1, len(pts[track.track_id])):
                if pts[track.track_id][j - 1] is None or pts[track.track_id][j] is None:
                    continue
                thickness = int(np.sqrt(64 / float(j + 1)) * 2)
                cv2.line(frame, (pts[track.track_id][j-1]),
                         (pts[track.track_id][j]), (color), thickness)

            # Caculate and draw distance
            if (class_name == 'car'):
                distance = (real_car_height * focal_length * original_h) / (bbox[3] - bbox[1])
            elif (class_name == 'truck'):
                distance = (real_truck_height * focal_length * original_h) / (bbox[3] - bbox[1])
            elif (class_name == 'bus'):
                distance = (real_bus_height * focal_length * original_h) / (bbox[3] - bbox[1])
            elif (class_name == 'motorbike'):
                distance = (real_motorbike_height * focal_length * original_h) / (bbox[3] - bbox[1])

            cv2.line(frame, (int(original_w/2), original_h),
                     (int((bbox[0]+bbox[2])/2), int((bbox[1]+bbox[3])/2)), color, 2)
            cv2.putText(frame, str(round(distance, 1)) + "m", (int(
                (bbox[0]+bbox[2])/2)-30, int((bbox[1]+bbox[3])/2)-10), 0, 0.75, (255, 255, 255), 2)
            # print(f"{bbox[2] - bbox[0]}, {bbox[3] - bbox[1]}")

            # Caculate velocity
            distances[track.track_id].append(distance)
            a = distances[track.track_id][len(distances[track.track_id])-1]
            b = distances[track.track_id][0]
            a_point = pts[track.track_id][len(distances[track.track_id])-1]
            b_point = pts[track.track_id][0]
            a_angle = math.atan(
                (original_h-a_point[0])/max(abs(a_point[1]-original_w/2), 0.0001))
            b_angle = math.atan(
                (original_h-b_point[0])/max(abs(b_point[1]-original_w/2), 0.0001))
            cos_gap_angle = math.cos(abs(a_angle-b_angle))
            gap_distance = math.sqrt(a**2 + b**2 - 2 * a * b * cos_gap_angle)
            if a < b:
                gap_distance = -gap_distance


            # Ki???m tra h?????ng c???a xe kh??c xem c?? nguy hi???m kh??ng, = 1 n???u nguy hi???m, = 0 ng?????c l???i
            if ((a_point[0] > b_point[0]) and (a_point[0] < original_w/2)) or ((a_point[0] < b_point[0]) and (a_point[0] > original_w/2)):
                tan = (b_point[1] - a_point[1]) / (b_point[0] - a_point[0])
                if tan < 0: tan = -tan
                # cv2.putText(frame, str(round(tan, 2)), (int(
                #         (bbox[0]+bbox[2])/2)-30, int((bbox[1]+bbox[3])/2)-30), 0, 0.75, (255, 255, 255), 2)
                if (a_point[1] > b_point[1]):
                    if tan < 0.7:
                        orientation = 1                     
                    else:
                        orientation = 0
                        
                else:               
                    if tan < 0.2:
                        orientation = 1
                    else:
                        orientation = 0
            else:
                orientation = 0
            
    

            # x = 0
            # y = 0 #  
            tan_center=0
            area_emergency = 0
            # Ki???m tra xe c?? ??? v??ng nguy hi???m kh??ng, = 1 n???u trong v??ng, = 2 n???u ch??? ch???m, = 0 n???u kh??ng ch???m
            if (center[0] < original_w/3) or (center[0] > 2*original_w/3) or (bbox[3] < 2*original_h/3):# Chia th??nh 4 v??ng tr??i, gi???a, ph???i, v??ng tr??n tr???i n???u l?? tr??i ho???c ph???i ho???c tr??n tr???i th?? an to??n-> x??t ph???n ??? gi???a
                area_emergency = 0
                # x = 1
            else: # X??t v??ng ??? gi???a
                # T??nh g??c v??? tr?? c???a ??i???m trung t??m
                if (center[0] < (original_w/2)) and (center[0] > (original_w/3)): 
                    tan_center = (original_h - center[1]) / (center[0] - (original_w/3)) 
                    # x = 2.1
                    # Ki???m tra ??i???m trung t??m c?? n???m trong v??ng nguy hi???m kh??ng
                    if tan_center < tan_front_area:
                        area_emergency = 1 # N???m trong v??ng nguy hi???m
                        # y = 2
                    else:
                        # N???u kh??ng th?? ki???m tra bounding box c?? ch???m v??ng nguy hi???m kh??ng
                        
                        tan_box = (original_h - bbox[3]) / (bbox[2] - original_w/3) 
                        # x = 3.1
                        
                        if tan_box < tan_front_area:
                            area_emergency = 2
                            # y = 3.1
                        else:
                            area_emergency = 0
                            # y = 3.2
                elif (center[0] > (original_w/2)) and (center[0] < (2*original_w/3)):
                    tan_center = (original_h - center[1]) / ((2*original_w/3) - center[0]) 
                    # x = 2.2
                    # Ki???m tra ??i???m trung t??m c?? n???m trong v??ng nguy hi???m kh??ng
                    if tan_center < tan_front_area:
                        area_emergency = 1 # N???m trong v??ng nguy hi???m
                        # y = 2
                
                    else:
                        # N???u kh??ng th?? ki???m tra bounding box c?? ch???m v??ng nguy hi???m kh??ng
                       
                        tan_box = (original_h - bbox[3]) / ((2*original_w/3) - bbox[0]) 
                        # x = 3.2
                        if tan_box < tan_front_area:
                            area_emergency = 2
                            # y = 3.1
                        else:
                            area_emergency = 0
                            # y = 3.2
            

            # V??? m??i t??n cho xe c?? h?????ng nguy hi???m
            if distance < 10 and orientation == 1 and area_emergency != 1:
                if center[0] < (original_w/2):
                    cv2.arrowedLine(frame, (int(bbox[0]), int(bbox[1]-10)), (int((bbox[0]+bbox[2])/2), int(bbox[1]-10)), (255, 215, 0), 3)
                else:
                    cv2.arrowedLine(frame, (int(bbox[2]), int(bbox[1]-10)), (int((bbox[0]+bbox[2])/2), int(bbox[1]-10)), (255, 215, 0), 3)




            # draw bbox on screen
            if ((area_emergency == 1) and (distance < 10)) or (distance < 5 and orientation == 1):
                color = (255, 0, 0)
                warn = 1
            elif (area_emergency == 1 and distance > 10) or (area_emergency == 2 and distance < 7) or (distance < 3):
                color = (255, 215, 0)
                if warn == 0:
                    warn = 2
            else:
                color = (0, 0, 255)
            # color = colors[int(track.track_id) % len(colors)]
            # color = [i * 255 for i in color]
            cv2.rectangle(frame, (int(bbox[0]), int(
                bbox[1])), (int(bbox[2]), int(bbox[3])), color, 2)
            # cv2.rectangle(frame, (int(bbox[0]), int(bbox[1]-30)), (int(bbox[0])+(
            #     len(class_name)+len(str(track.track_id)))*17, int(bbox[1])), color, -1)
            # cv2.putText(frame, class_name + "-" + str(track.track_id) + "(" + str(round(gap_distance/(30/fps), 1)) + "m/s)",
            #             (int(bbox[0]), int(bbox[1]-10)), 0, 0.75, (255, 255, 255), 2)
            velocity = round(gap_distance/(len(distances[track.track_id])/fps), 1)
            # cv2.putText(frame, str(area_emergency) + "," + str(orientation),
            #             (int(bbox[0]), int(bbox[1]-10)), 0, 0.75, (255, 255, 255), 2)

            # if enable info flag then print details about each track
            if FLAGS.info:
                print("Tracker ID: {}, Class: {},  BBox Coords (xmin, ymin, xmax, ymax): {}".format(
                    str(track.track_id), class_name, (int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3]))))

        # calculate frames per second of running detections
        fps = 1.0 / (time.time() - start_time)
        cv2.putText(frame, str(round(fps,1)) + " fps", (50, 50), 0, 1, (255, 128, 0), 1)
        if warn == 1:
            cv2.putText(frame, "Emergency!!", (50, 100), 0, 1.5, (255, 0, 0), 3)
        elif warn == 2:
            cv2.putText(frame, "Warning!!", (50, 100), 0, 1.5, (255, 215, 0), 3)
        else:
            cv2.putText(frame, "Safe", (50, 100), 0, 1.5, (0, 0, 255), 3)


        print("FPS: %.2f" % fps)
        result = np.asarray(frame)
        result = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

        if not FLAGS.dont_show:
            cv2.imshow("Output Video", result)

        # if output flag is set, save video file
        if FLAGS.output:
            out.write(result)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    cv2.destroyAllWindows()


if __name__ == '__main__':
    try:
        app.run(main)
    except SystemExit:
        pass
