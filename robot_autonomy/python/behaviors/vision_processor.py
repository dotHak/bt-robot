import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import cv2
import numpy as np
import tensorflow as tf
from object_detection.utils import label_map_util
from object_detection.utils import visualization_utils as viz_utils

class VisionProcessor(Node):
    def __init__(self):
        super().__init__('vision_processor')
        self.bridge = CvBridge()
        self.detection_model = self.load_model('/path/to/model/frozen_inference_graph.pb')
        self.category_index = label_map_util.create_category_index_from_labelmap('/path/to/labelmap.pbtxt', use_display_name=True)

        self.image_sub = self.create_subscription(Image, '/camera/image_raw', self.image_callback, 10)

    def load_model(self, model_path):
        detection_model = tf.saved_model.load(model_path)
        return detection_model.signatures['serving_default']

    def image_callback(self, msg):
        img = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
        input_tensor = tf.convert_to_tensor(img)
        input_tensor = input_tensor[tf.newaxis, ...]

        detections = self.detection_model(input_tensor)
        num_detections = int(detections.pop('num_detections'))
        detections = {key: value[0, :num_detections].numpy()
                      for key, value in detections.items()}
        detections['num_detections'] = num_detections

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

        cv2.imshow('Object Detection', labeled_img)
        cv2.waitKey(1)

        for i, score in enumerate(detections['detection_scores']):
            if score > 0.5:
                class_name = self.category_index[detections['detection_classes'][i]]['name']
                self.get_logger().info(f"Detected {class_name} with score {score:.2f}")

def main(args=None):
    rclpy.init(args=args)
    vision_processor = VisionProcessor()
    rclpy.spin(vision_processor)
    vision_processor.destroy_node()
    rclpy.shutdown()