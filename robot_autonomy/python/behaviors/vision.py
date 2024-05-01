import cv2
import cv_bridge
import rclpy
from rclpy.duration import Duration
import py_trees
from sensor_msgs.msg import Image
import tensorflow as tf
from object_detection.utils import label_map_util
from object_detection.utils import visualization_utils as viz_utils
import numpy as np

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Define HSV color space thresholds
hsv_threshold_dict = {
    "red": ((160, 220, 0), (180, 255, 255)),
    "green": ((40, 220, 0), (90, 255, 255)),
    "blue": ((100, 220, 0), (150, 255, 255)),
}


class LookForObject(py_trees.behaviour.Behaviour):
    """
    Gets images from the robot and looks for object using
    simple HSV color space thresholding and blob detection.
    """

    def __init__(self, name, color, node, img_timeout=3.0, visualize=True):
        super(LookForObject, self).__init__(name)
        self.color = color
        self.node = node
        self.hsv_min = hsv_threshold_dict[color][0]
        self.hsv_max = hsv_threshold_dict[color][1]
        self.img_timeout = Duration(nanoseconds=img_timeout * 1e9)
        self.viz_window_name = "Image with Detections"
        self.visualize = visualize
        if self.visualize:
            plt.figure(1)
            plt.axis("off")
            plt.title(self.viz_window_name)
            plt.ion()

    def initialise(self):
        """Starts all the vision related objects"""
        self.bridge = cv_bridge.CvBridge()
        params = cv2.SimpleBlobDetector_Params()
        params.minArea = 100
        params.maxArea = 100000
        params.filterByArea = True
        params.filterByColor = False
        params.filterByInertia = False
        params.filterByConvexity = False
        params.thresholdStep = 50
        self.detector = cv2.SimpleBlobDetector_create(params)

        self.start_time = self.node.get_clock().now()
        self.latest_img_msg = None
        self.img_sub = self.node.create_subscription(
            Image, "/camera/image_raw", self.img_callback, 10
        )

    def update(self):
        """Looks for objects using the TensorFlow Object Detection API"""
        # Wait for an image message and handle failure case
        now = self.node.get_clock().now()
        if self.latest_img_msg is None:
            if now - self.start_time < self.img_timeout:
                return py_trees.common.Status.RUNNING
            else:
                self.logger.info("Image timeout exceeded")
                return py_trees.common.Status.FAILURE

        # Process the image using TensorFlow Object Detection API
        img = self.bridge.imgmsg_to_cv2(self.latest_img_msg, desired_encoding="bgr8")
        input_tensor = tf.convert_to_tensor(img)
        input_tensor = input_tensor[tf.newaxis, ...]

        detections = self.detection_model(input_tensor)
        num_detections = int(detections.pop('num_detections'))
        detections = {key: value[0, :num_detections].numpy()
                      for key, value in detections.items()}
        detections['num_detections'] = num_detections

        # Visualize, if enabled
        if self.visualize:
            labeled_img = img.copy()
            viz_utils.visualize_boxes_and_labels_on_image_array(
                labeled_img,
                detections['detection_boxes'],
                detections['detection_classes'].astype(np.int64),
                detections['detection_scores'],
                self.category_index,
                use_normalized_coordinates=True,
                max_boxes_to_draw=200,
                min_score_thresh=0.5,
                agnostic_mode=False)

            cv2.imshow(self.viz_window_name, labeled_img)
            cv2.waitKey(100)

        # Check if objects were detected
        if np.sum(detections['detection_scores'] > 0.5) == 0:
            self.logger.info("No objects detected")
            return py_trees.common.Status.FAILURE

        for i, score in enumerate(detections['detection_scores']):
            if score > 0.5:
                class_name = self.category_index[detections['detection_classes'][i]]['name']
                self.logger.info(f"Detected {class_name} with score {score:.2f}")

        return py_trees.common.Status.SUCCESS

    def terminate(self, new_status):
        self.logger.info(f"Terminated with status {new_status}")
        self.img_sub = None
        self.latest_img_msg = None

    def img_callback(self, msg):
        # self.logger.info("Image received")
        self.latest_img_msg = msg
