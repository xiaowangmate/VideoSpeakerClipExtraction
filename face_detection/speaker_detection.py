import os
import cv2
import logging
from tqdm import tqdm
import face_recognition
from moviepy.editor import VideoFileClip


class SpeakerDetector:
    def __init__(self):
        self.speaker_face = None
        dir_path = os.path.dirname(os.path.abspath(__file__))
        self.cascade_path = os.path.join(dir_path, 'haarcascade_frontalface_default.xml')

    def load_speaker_face(self, speaker_face_picture_path):
        picture_of_speaker = face_recognition.load_image_file(speaker_face_picture_path)
        self.speaker_face = face_recognition.face_encodings(picture_of_speaker)[0]

    def detect_and_load_speaker_face(self, video_path):
        logging.warning("Detect and load speaker face...")
        clip = VideoFileClip(video_path)
        fps = clip.fps
        detect_face = False
        total_frames = int(clip.duration * fps)
        for img in tqdm(clip.iter_frames(fps=fps), total=total_frames):
            if len(self.detect_frontal_face(img)) > 0:
                speaker_face = face_recognition.face_encodings(img)
                if speaker_face:
                    self.speaker_face = face_recognition.face_encodings(img)[0]
                    detect_face = True
                    break
        clip.close()
        if detect_face:
            logging.warning("Speaker's face detected and loaded successfully!")
        else:
            logging.warning("Speaker's face not detected.")

    def detect_frontal_face(self, img):
        face_cascade = cv2.CascadeClassifier(self.cascade_path)
        if os.path.isfile(img):
            img = cv2.imread(img)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.1, 4)
        return faces

    def contain_face(self, img):
        if os.path.isfile(img):
            img = face_recognition.load_image_file(img)
        face_locations = face_recognition.face_locations(img)
        return face_locations

    def is_speaker_in_image(self, img):
        if os.path.isfile(img):
            img = face_recognition.load_image_file(img)
        target_img_encoding = face_recognition.face_encodings(img)[0]
        results = face_recognition.compare_faces([self.speaker_face], target_img_encoding)
        return results[0]

    def find_speaker_in_image(self, img):
        if os.path.isfile(img):
            img = face_recognition.load_image_file(img)
        face_locations = face_recognition.face_locations(img)
        is_contained = False
        for (top_right_y, top_right_x, left_bottom_y, left_bottom_x) in face_locations:
            face = img[top_right_y - 50:left_bottom_y + 50, left_bottom_x - 50:top_right_x + 50]
            if len(face) > 0:
                face = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
                face_encoding = face_recognition.face_encodings(face)
                if face_encoding:
                    results = face_recognition.compare_faces([self.speaker_face], face_encoding[0])
                    is_contained = results[0]
        return is_contained


if __name__ == '__main__':
    sd = SpeakerDetector()

    base_path = "../input/img/test_face_detection"

    sd.load_speaker_face(f"{base_path}/caixukun.jpg")
    result = sd.find_speaker_in_image(f"{base_path}/cuixukun_wuyifang.jpg")
    print(result)
    if result:
        print("包含目标人物")
    else:
        print("不包含目标人物")

    # sd.detect_and_load_speaker_face(f"{base_path}/220788303-1-208.mp4")
