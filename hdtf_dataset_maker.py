import os
from video_speaker_clip_extractor import VideoSpeakerClipExtractor

extractor = VideoSpeakerClipExtractor("output")

file_folder_path = "input/txt"

for dir1, dir2, files in os.walk(file_folder_path):
    for file in files:
        file_path = f"{file_folder_path}/{file}"
        print(file_path)
        with open(file_path, mode="r", encoding="utf-8") as r:
            video_url_list = r.read().split("\n")
            for video_url_info in video_url_list:
                if video_url_info:
                    video_id, video_url = video_url_info.split(" ")
                    print(f"video_url: {video_url}")
                    extractor.gen_video_data_by_video_url(video_url)
print("All video process completed.")
