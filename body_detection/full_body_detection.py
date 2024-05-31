import os
import cv2
import mediapipe as mp


class FullBodyDetector:
    def __init__(self):
        self.mpPose = mp.solutions.pose

    def detect_full_body(self, img, threshold=0.3):
        is_full_body = False
        pose = self.mpPose.Pose()
        if os.path.isfile(img):
            img = cv2.imread(img)
        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        results = pose.process(imgRGB)
        if results.pose_landmarks:
            left_feet_keypoint = results.pose_landmarks.landmark[31].visibility
            right_feet_keypoint = results.pose_landmarks.landmark[32].visibility
            if left_feet_keypoint > threshold or right_feet_keypoint > threshold:
                print(f"Full body: {left_feet_keypoint}, {right_feet_keypoint}")
                is_full_body = True
            else:
                print(f"No full body: {left_feet_keypoint}, {right_feet_keypoint}")
        else:
            print("No body")
        return is_full_body


if __name__ == '__main__':
    bd = FullBodyDetector()

    base_path = "../input/img/test_body_detection"

    for dir, sub_dir, files in os.walk(base_path):
        for file in files:
            input_image = os.path.join(dir, file)
            bd.detect_full_body(input_image)
            print(file)
