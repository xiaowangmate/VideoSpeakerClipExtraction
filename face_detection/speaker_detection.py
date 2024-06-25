import os
import gc
import cv2
import logging
import numpy as np
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

    def detect_and_load_speaker_face(self, video_path, num_frames=None):
        logging.warning("Detect and load speaker face...")
        clip = VideoFileClip(video_path)
        fps = clip.fps
        total_frames = int(clip.duration * fps)
        if num_frames is None or total_frames < num_frames:
            frame_interval = 1
        else:
            frame_interval = max(1, total_frames // num_frames)
        for i, img in enumerate(tqdm(clip.iter_frames(fps=fps), total=total_frames)):
            if i % frame_interval == 0:
                if len(self.detect_frontal_face(img)) > 0:
                    speaker_face = face_recognition.face_encodings(img)
                    if speaker_face:
                        self.speaker_face = face_recognition.face_encodings(img)[0]
                        clip.close()
                        del img, speaker_face
                        gc.collect()
                        logging.warning("Speaker's face detected and loaded successfully!")
                        return True
                    else:
                        del speaker_face
                        gc.collect()
                del img
                gc.collect()
        clip.close()
        logging.warning("Speaker's face not detected.")
        return False

    def detect_frontal_face(self, img):
        face_cascade = cv2.CascadeClassifier(self.cascade_path)
        if not isinstance(img, np.ndarray):
            img = cv2.imread(img)
        if img is not None:
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            faces = face_cascade.detectMultiScale(gray, 1.1, 4)
            del gray
            gc.collect()
        else:
            faces = None
        del face_cascade, img
        gc.collect()
        return faces

    def contain_face(self, img):
        if not isinstance(img, np.ndarray):
            img = face_recognition.load_image_file(img)
        face_locations = face_recognition.face_locations(img)
        del img
        gc.collect()
        return face_locations

    def is_speaker_in_image(self, img):
        if not isinstance(img, np.ndarray):
            img = face_recognition.load_image_file(img)
        target_img_encoding = face_recognition.face_encodings(img)[0]
        results = face_recognition.compare_faces([self.speaker_face], target_img_encoding)
        del img, target_img_encoding
        gc.collect()
        return results[0]

    # def find_speaker_in_image(self, img):
    #     if not isinstance(img, np.ndarray):
    #         img = face_recognition.load_image_file(img)
    #     face_locations = face_recognition.face_locations(img)
    #     for (top_right_y, top_right_x, left_bottom_y, left_bottom_x) in face_locations:
    #         face = img[top_right_y - 50:left_bottom_y + 50, left_bottom_x - 50:top_right_x + 50]
    #         if face is not None and face.shape[0] > 0 and face.shape[1] > 0:
    #             try:
    #                 face = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
    #                 face_encoding = face_recognition.face_encodings(face)
    #                 if face_encoding:
    #                     results = face_recognition.compare_faces([self.speaker_face], face_encoding[0])
    #                     if results[0]:
    #                         return True
    #             except Exception as e:
    #                 print(f"find speaker in image error: {str(e)}")
    #                 continue
    #     return False

    def find_speaker_in_image(self, img):
        try:
            if not isinstance(img, np.ndarray):
                img = face_recognition.load_image_file(img)
            face_locations = face_recognition.face_locations(img)
            for (top, right, bottom, left) in face_locations:
                top_margin = max(0, top - 50)
                bottom_margin = min(img.shape[0], bottom + 50)
                left_margin = max(0, left - 50)
                right_margin = min(img.shape[1], right + 50)
                face = img[top_margin:bottom_margin, left_margin:right_margin]
                if face is not None and face.shape[0] > 0 and face.shape[1] > 0:
                    try:
                        if face.shape[2] == 3:
                            face = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
                        face_encoding = face_recognition.face_encodings(face)
                        if face_encoding:
                            results = face_recognition.compare_faces([self.speaker_face], face_encoding[0])
                            if results[0]:
                                return True
                    except Exception as e:
                        print(f"find speaker in image error: {str(e)}")
                    finally:
                        del face
        except Exception as main_e:
            print(f"Error processing image: {str(main_e)}")
        finally:
            del img
        return False


if __name__ == '__main__':
    sd = SpeakerDetector()

    base_path = "../input/img/test_face_detection"

    # sd.load_speaker_face(f"{base_path}/caixukun.jpg")
    # result = sd.find_speaker_in_image(f"{base_path}/cuixukun_wuyifang.jpg")
    # print(result)
    # if result:
    #     print("包含目标人物")
    # else:
    #     print("不包含目标人物")

    print(sd.detect_and_load_speaker_face(
        r"C:\Users\86176\PycharmProjects\HumanSpeak\TikTok\tiktok_videos\7379018392067689774.mp4", 1000))
