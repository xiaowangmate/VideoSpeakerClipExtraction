import os
import gc
import cv2
import numpy as np
import mediapipe as mp


class FullBodyDetector:
    def __init__(self):
        self.mpPose = mp.solutions.pose

    def has_body(self, img):
        pose = self.mpPose.Pose()
        if not isinstance(img, np.ndarray):
            img = cv2.imread(img)
        if img is not None:
            imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            results = pose.process(imgRGB)
            del img, imgRGB, pose
            gc.collect()
            if results.pose_landmarks:
                print("Has Body.")
                return True
            else:
                print("Has not Body.")
                return False
        else:
            del img, pose
            gc.collect()
            print("Has not Body.")
            return False

    def detect_full_body(self, img, threshold=0.3):
        is_full_body = False
        pose = self.mpPose.Pose()
        if not isinstance(img, np.ndarray):
            img = cv2.imread(img)
        if img is not None:
            imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            results = pose.process(imgRGB)
            del img, imgRGB
            gc.collect()
            if results.pose_landmarks:
                left_feet_keypoint = results.pose_landmarks.landmark[31].visibility
                right_feet_keypoint = results.pose_landmarks.landmark[32].visibility
                del results, pose
                gc.collect()
                if left_feet_keypoint > threshold or right_feet_keypoint > threshold:
                    print(f"Full body: {left_feet_keypoint}, {right_feet_keypoint}")
                    return True
                else:
                    print(f"No full body: {left_feet_keypoint}, {right_feet_keypoint}")
            else:
                print("No body")
                del results, pose
                gc.collect()
        else:
            print("No body")
            del img, pose
            gc.collect()
        return is_full_body


if __name__ == '__main__':
    bd = FullBodyDetector()

    base_path = "../input/img/test_body_detection"

    for dir, sub_dir, files in os.walk(base_path):
        for file in files:
            input_image = os.path.join(dir, file)
            bd.detect_full_body(input_image)
            print(file)
