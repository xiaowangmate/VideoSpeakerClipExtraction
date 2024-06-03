import os
import cv2
import json
import logging
import datetime
import requests
import imagehash
import scrapetube
from PIL import Image
from pytube import YouTube
from pydub import AudioSegment
from moviepy.editor import VideoFileClip
from pydub.silence import detect_nonsilent
from video_asr.video_speech_recognition import VideoASR
from face_detection.speaker_detection import SpeakerDetector
from body_detection.full_body_detection import FullBodyDetector
from llm.openai_api import openai_call
from llm.speaker_name_recognition_prompt import speaker_name_recognition_prompt

asr = VideoASR()
sd = SpeakerDetector()
bd = FullBodyDetector()


class VideoSpeakerClipExtractor:
    def __init__(self, output_base_dir):
        self.headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/117.0.0.0 Safari/537.36 Edg/117.0.2045.60",
            "Referer": "https://mattw.io/"
        }
        self.metadata = {
            "original_video_id": None,
            "video_title": None,
            "source_url": None,
            "clip_id": None,
            "filename": None,
            "start_time": None,
            "end_time": None,
            "duration_seconds": None,
            "speaking_duration": None,
            "transcript": None,
            "speaker_upper_body_visible": True,
            "topic_keywords": None,
            "speaker_full_body_visible": "half",
            "category": None,
            "speaker_name": None,
            "language": None
        }
        self.processed_video_list = self.video_filter()
        self.output_base_dir = output_base_dir

    def get_video_detail(self, video_id):
        request_url = f"https://www.googleapis.com/youtube/v3/videos?key=AIzaSyASTMQck-jttF8qy9rtEnt1HyEYw5AmhE8&quotaUser=fBWEAGx05v5OAXp569fBfk7bm7k8AQrB6crdJBmO&part=snippet%2Cstatistics%2CrecordingDetails%2Cstatus%2CliveStreamingDetails%2Clocalizations%2CcontentDetails%2CtopicDetails&id={video_id}&_=1697074338983"
        response = requests.get(request_url, headers=self.headers).json()
        try:
            topic_keywords = response["items"][0]["snippet"]["tags"]
        except:
            topic_keywords = []
        try:
            language = response["items"][0]["snippet"]["defaultLanguage"]
        except:
            language = None
        return topic_keywords, language

    def get_speaker_name(self, content):
        speaker_name = openai_call(speaker_name_recognition_prompt.format(content))
        return speaker_name

    def download_video(self, video_id, folder_path):
        url = "https://www.youtube.com/watch?v=" + video_id
        youtube = YouTube(url)
        video_streams = youtube.streams
        video = video_streams.get_highest_resolution()
        video_download_name = f"{video_id.replace('-', '_')}.mp4"
        os.makedirs(folder_path, exist_ok=True)
        video.download(folder_path, filename=video_download_name)
        video_download_path = f"{folder_path}/{video_download_name}"
        logging.warning(f"downloaded video: {video.default_filename}.")
        return video_download_path

    def video_filter(self):
        with open("record/processed_video.txt", mode="r", encoding="utf-8") as r:
            processed_video_list = r.read().split("\n")
            return processed_video_list

    def update_processed_video_list(self, video_id):
        with open("record/processed_video.txt", mode="a+", encoding="utf-8") as w:
            w.write(f"{video_id}\n")
            self.processed_video_list.append(video_id)

    def gen_video_data_by_keyword(self, keyword):
        videos = scrapetube.get_search(keyword)
        for video in videos:
            video_id = video["videoId"]
            if video_id not in self.processed_video_list:
                video_title = video["title"]["runs"][0]["text"]
                video_url = "https://www.youtube.com/watch?v=" + video_id
                topic_keywords, language = self.get_video_detail(video_id)

                self.metadata["original_video_id"] = video_id
                self.metadata["video_title"] = video_title
                self.metadata["source_url"] = video_url
                self.metadata["topic_keywords"] = topic_keywords
                self.metadata["category"] = keyword

                speaker_name = self.get_speaker_name(video_title)
                if speaker_name != "None":
                    self.metadata["speaker_name"] = speaker_name

                self.metadata["language"] = language

                print(
                    f"video_id: {video_id}, video_title: {video_title}， video_url： {video_url}， topic_keywords: {topic_keywords}, language: {language}")

                folder_path = f"{self.output_base_dir}/{video_id.replace('-', '_')}"
                try:
                    video_download_path = self.download_video(video_id, folder_path)

                    sd.detect_and_load_speaker_face(video_download_path)
                    self.video_clipping(
                        video_download_path,
                        folder_path,
                        min_duration=5,
                        faceRecSpeedMult=10,
                        cutoff=7
                    )
                except Exception as e:
                    logging.warning(f"Video: {video_id} download error: {str(e)}.")
                    self.init_metadata()
            else:
                logging.warning(f"Video: {video_id} already processed.")

    def process_target_video(self, video_download_path, video_id, keyword):
        video_url = "https://www.youtube.com/watch?v=" + video_id
        request_url = f"https://www.googleapis.com/youtube/v3/videos?key=AIzaSyASTMQck-jttF8qy9rtEnt1HyEYw5AmhE8&quotaUser=fBWEAGx05v5OAXp569fBfk7bm7k8AQrB6crdJBmO&part=snippet%2Cstatistics%2CrecordingDetails%2Cstatus%2CliveStreamingDetails%2Clocalizations%2CcontentDetails%2CtopicDetails&id={video_id}&_=1697074338983"
        response = requests.get(request_url, headers=self.headers).json()
        try:
            video_title = response["items"][0]["snippet"]["title"]
        except:
            video_title = response["items"][0]["snippet"]["title"]
        try:
            topic_keywords = response["items"][0]["snippet"]["tags"]
        except:
            topic_keywords = []
        try:
            language = response["items"][0]["snippet"]["defaultLanguage"]
        except:
            language = None

        self.metadata["original_video_id"] = video_id
        self.metadata["video_title"] = video_title
        self.metadata["source_url"] = video_url
        self.metadata["topic_keywords"] = topic_keywords
        self.metadata["category"] = keyword

        speaker_name = self.get_speaker_name(video_title)
        if speaker_name != "None":
            self.metadata["speaker_name"] = speaker_name

        self.metadata["language"] = language

        print(
            f"video_id: {video_id}, video_title: {video_title}， video_url： {video_url}， topic_keywords: {topic_keywords}, language: {language}")

        folder_path = f"{self.output_base_dir}/{video_id.replace('-', '_')}"
        os.makedirs(folder_path, exist_ok=True)

        sd.detect_and_load_speaker_face(video_download_path)
        self.video_clipping(
            video_download_path,
            folder_path,
            min_duration=5,
            faceRecSpeedMult=10,
            cutoff=7
        )

    def check_for_shot_transitions(self, img1, img2, cutoff=7):
        img1 = Image.fromarray(cv2.cvtColor(img1, cv2.COLOR_BGR2RGB))
        img2 = Image.fromarray(cv2.cvtColor(img2, cv2.COLOR_BGR2RGB))

        n0 = imagehash.average_hash(img1)
        n1 = imagehash.average_hash(img2)

        flag = True
        if n0 - n1 < cutoff:
            logging.warning('No shot transition.')
        else:
            flag = False
            logging.warning('Shot transition.')

        return flag

    def convert_seconds_to_time_format(self, seconds):
        time = datetime.timedelta(seconds=seconds)
        str_time = str(time)
        if '.' in str_time:
            str_time = str_time[:str_time.index('.') + 4]
        else:
            str_time += '.000'
        return str_time

    def video_clipping(self, video_path, output_dir, min_duration=5, faceRecSpeedMult=10, cutoff=7):
        clip = VideoFileClip(video_path)
        fps = clip.fps

        im0 = ""
        start_time = 0
        end_time = 0
        clip_index = 0
        for i, img in enumerate(clip.iter_frames(fps)):
            if i == 0:
                im0 = img
            time = i / fps
            logging.warning(f"Check for shot transitions, frame {i}: ")
            result = self.check_for_shot_transitions(im0, img, cutoff)
            if not result:
                end_time = (i - 1) / fps
                logging.warning(
                    f"Time range of the shot: {self.convert_seconds_to_time_format(start_time)} - {self.convert_seconds_to_time_format(end_time)}.")
                if start_time != end_time and end_time - start_time >= min_duration:
                    subclip = clip.subclip(start_time, end_time)
                    logging.warning(f"Cutting the shot, checking whether the speaker appears in the clip...")
                    full_body_count = half_body_count = 0
                    for frame in subclip.iter_frames(fps / faceRecSpeedMult):
                        if bd.detect_full_body(frame):
                            full_body_count += 1
                        else:
                            half_body_count += 1

                        if sd.find_speaker_in_image(frame):
                            logging.warning(f"Speaker in frame.")
                            clip_name = f'{self.metadata["original_video_id"].replace("-", "_")}_{clip_index}.mp4'
                            clip_path = f"{output_dir}/{clip_name}"
                            subclip.audio.write_audiofile(clip_path.replace(".mp4", ".mp3"))
                            subclip.write_videofile(clip_path)
                            logging.warning(f"Clip extraction from video: {clip_path}")
                            self.metadata["clip_id"] = clip_index
                            self.metadata["filename"] = clip_name
                            self.metadata["start_time"] = start_time
                            self.metadata["end_time"] = end_time
                            self.metadata["duration_seconds"] = VideoFileClip(clip_path).duration
                            self.metadata["speaking_duration"] = self.detect_speaking_duration(clip_path)
                            self.metadata["transcript"] = asr.speech_recognition(clip_path)

                            if full_body_count == 0 and half_body_count > 0:
                                self.metadata["speaker_full_body_visible"] = "half"
                            elif full_body_count > 0 and half_body_count > 0:
                                self.metadata["speaker_full_body_visible"] = "mix"
                            else:
                                self.metadata["speaker_full_body_visible"] = "full"

                            self.write_clip_metadata(clip_path)
                            clip_index += 1
                            break
                        else:
                            logging.warning(f"Speaker not in frame.")
                start_time = time
            im0 = img
        end_time = clip.duration
        if start_time != end_time and end_time - start_time >= min_duration:
            subclip = clip.subclip(start_time, end_time)
            logging.warning(f"Cutting the shot, checking whether the speaker appears in the clip...")
            for frame in subclip.iter_frames(fps / faceRecSpeedMult):
                if sd.find_speaker_in_image(frame):
                    logging.warning(f"Speaker in frame.")
                    clip_name = f'{self.metadata["original_video_id"].replace("-", "_")}_{clip_index}.mp4'
                    clip_path = f"{output_dir}/{clip_name}"
                    subclip.audio.write_audiofile(clip_path)
                    subclip.write_videofile(clip_path)
                    logging.warning(f"Clip extraction from video: {clip_path}")
                    self.metadata["clip_id"] = clip_index
                    self.metadata["filename"] = clip_name
                    self.metadata["start_time"] = start_time
                    self.metadata["end_time"] = end_time
                    self.metadata["duration_seconds"] = VideoFileClip(clip_path).duration
                    self.metadata["speaking_duration"] = self.detect_speaking_duration(clip_path)
                    self.metadata["transcript"] = asr.speech_recognition(clip_path)
                    self.write_clip_metadata(clip_path)
                    clip_index += 1
                    break
                else:
                    logging.warning(f"Speaker not in frame.")

        clip.close()
        self.update_processed_video_list(self.metadata["original_video_id"])
        self.init_metadata()
        logging.warning(f"Video shot cut complete.")

    def detect_different_shots(self, clip, fps, cutoff=7):
        im0 = ""
        different_shots = False
        for i, img in enumerate(clip.iter_frames(fps)):
            if i == 0:
                im0 = img
            result = self.check_for_shot_transitions(im0, img, cutoff)
            if not result:
                different_shots = True
                break
            im0 = img
        return different_shots

    def detect_speaking_duration(self, clip_path):
        clip = VideoFileClip(clip_path)
        audio_path = "temp/temp_clip_audio.wav"
        audio = clip.audio
        audio.write_audiofile(audio_path)
        sound = AudioSegment.from_file(audio_path, format="wav")
        nonsilent_ranges = detect_nonsilent(sound, min_silence_len=1000, silence_thresh=-32)
        talk_duration = sum([end - start for start, end in nonsilent_ranges]) / 1000.0
        return talk_duration

    def init_metadata(self):
        self.metadata = {
            "original_video_id": None,
            "video_title": None,
            "source_url": None,
            "clip_id": None,
            "filename": None,
            "start_time": None,
            "end_time": None,
            "duration_seconds": None,
            "speaking_duration": None,
            "transcript": None,
            "speaker_upper_body_visible": True,
            "topic_keywords": None,
            "speaker_full_body_visible": "half",
            "category": None,
            "speaker_name": None,
            "language": None
        }

    def write_clip_metadata(self, clip_path):
        with open(clip_path.replace(".mp4", ".json"), mode="w", encoding="utf-8") as w:
            w.write(json.dumps(self.metadata, ensure_ascii=False))
        logging.warning(f"Successfully saved video clip metadata.")


if __name__ == "__main__":
    extractor = VideoSpeakerClipExtractor("./output")
    extractor.gen_video_data_by_keyword("Stand-up comedy")
