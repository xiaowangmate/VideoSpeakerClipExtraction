import os
import subprocess
from typing import List
from ffmpy import FFmpeg
from urllib import parse
from pytube import YouTube


class HDTFCropDatasetMaker:
    def __init__(self, source_dir="input", output_dir="output"):
        self.source_dir = source_dir
        self.output_dir = output_dir
        self.cropped_list_path = "record/cropped_list.txt"
        self.cropped_list = self.read_cropped_list()

    def read_cropped_list(self):
        with open(self.cropped_list_path, 'r', encoding="utf-8") as f:
            return f.read().strip().split("\n")

    def update_cropped_list(self, folder_name):
        with open(self.cropped_list_path, 'a+', encoding="utf-8") as f:
            f.write(f"{folder_name}\n")
            self.cropped_list.append(folder_name)

    def read_file_as_space_separated_data(self, filepath):
        with open(filepath, 'r', encoding="utf-8") as f:
            lines = f.read().replace("\ufeff", "").splitlines()
            lines = [[v.strip() for v in l.strip().split(' ')] for l in lines]
            data = {l[0]: l[1:] for l in lines}
        return data

    def process_data(self):
        process_info_list = self.get_process_info_list()
        print(f"process_info_list: {process_info_list}")
        for process_info in process_info_list:
            folder_name = process_info["name"]

            if folder_name not in self.cropped_list:
                video_id = process_info["id"]
                intervals = process_info["intervals"]
                crops = process_info["crops"]
                resolution = process_info["resolution"]

                folder_path = os.path.join(self.output_dir, folder_name)
                print(f"video_id: {video_id}, folder_path: {folder_path}, resolution: {resolution}")
                try:
                    video_download_path = self.download_video(video_id, folder_path, resolution)

                    clip_idx = 0
                    if intervals:
                        for clip_interval, clip_crop in zip(intervals, crops):
                            start, end = clip_interval
                            clip_path = os.path.join(self.output_dir, folder_name, f"{video_id}_{clip_idx}" + '.mp4')
                            print(video_download_path, clip_path, start, end, clip_crop)
                            self.cut_and_crop_video(video_download_path, clip_path, start, end, clip_crop)
                            clip_idx += 1
                    else:
                        if crops:
                            for crop in crops:
                                clip_path = os.path.join(self.output_dir, folder_name, f"{video_id}_{clip_idx}" + '.mp4')
                                self.cut_and_crop_video(video_download_path, clip_path, None, None, crop)
                                clip_idx += 1
                    self.update_cropped_list(folder_name)
                    print(f"{folder_name} process success.")
                except Exception as e:
                    print(f"Video {video_id} error: {str(e)}.")
                    self.update_cropped_list(folder_name)
            else:
                print(f"{folder_name} already processed.")
        print("-" * 100)

    def get_process_info_list(self):
        process_info_list = []

        subsets = ["RD", "WDA", "WRA"]
        for subset in subsets:
            video_urls = self.read_file_as_space_separated_data(
                os.path.join(self.source_dir, f'{subset}_video_url.txt'))
            crops = self.read_file_as_space_separated_data(
                os.path.join(self.source_dir, f'{subset}_crop_wh.txt'))
            intervals = self.read_file_as_space_separated_data(
                os.path.join(self.source_dir, f'{subset}_annotion_time.txt'))
            resolutions = self.read_file_as_space_separated_data(
                os.path.join(self.source_dir, f'{subset}_resolution.txt'))

            for video_name, (video_url,) in video_urls.items():
                if f'{video_name}.mp4' not in resolutions.keys() or len(resolutions[f'{video_name}.mp4']) > 1:
                    resolution = None
                else:
                    resolution = resolutions[f'{video_name}.mp4'][0]

                if f'{video_name}.mp4' not in intervals.keys():
                    clips_intervals = None
                    clips_crops = None
                else:
                    all_clips_intervals = [x.split('-') for x in intervals[f'{video_name}.mp4']]
                    clips_crops = []
                    clips_intervals = []

                    for clip_idx, clip_interval in enumerate(all_clips_intervals):
                        clip_name = f'{video_name}_{clip_idx}.mp4'
                        if clip_name not in crops.keys():
                            clips_crop = None
                        else:
                            clips_crop = crops[clip_name]
                        clips_crops.append(clips_crop)
                        clips_intervals.append(clip_interval)

                process_info_list.append({
                    'name': f'{subset}_{video_name}',
                    'id': parse.parse_qs(parse.urlparse(video_url).query)['v'][0],
                    'intervals': clips_intervals,
                    'crops': clips_crops,
                    'resolution': resolution
                })
        return process_info_list

    def download_video(self, video_id, folder_path, resolution):
        video_name = video_id + ".mp4"
        video_download_path = os.path.join(folder_path, video_name)
        if not os.path.exists(video_download_path):
            audio_name = video_id + ".mp3"
            video_url = "https://www.youtube.com/watch?v=" + video_id
            youtube = YouTube(video_url, use_oauth=True, allow_oauth_cache=True)
            video_streams = youtube.streams
            if resolution:
                video = video_streams.filter(res=f"{resolution}p").first()
                if not video:
                    video = video_streams.get_highest_resolution()
            else:
                video = video_streams.get_highest_resolution()
            audio = video_streams.filter(only_audio=True).first()

            os.makedirs(folder_path, exist_ok=True)

            temp_video_path = os.path.join(folder_path, 'temp_' + video_name)
            temp_audio_path = os.path.join(folder_path, 'temp_' + audio_name)

            video.download(folder_path, filename="temp_" + video_name)
            audio.download(folder_path, filename="temp_" + audio_name)

            ff = FFmpeg(
                inputs={temp_video_path: None, temp_audio_path: None},
                outputs={video_download_path: '-c:v copy -c:a aac -strict experimental'}
            )
            ff.run()

            os.remove(temp_video_path)
            os.remove(temp_audio_path)
            print(f"Video: {video_id} download successful.")
        else:
            print(f"Video: {video_id} already download.")
        return video_download_path

    def cut_and_crop_video(self, raw_video_path, output_path, start, end, crop: List[int]):
        arg_list = [
            "ffmpeg", "-i", raw_video_path,
            "-strict", "-2",
            "-loglevel", "quiet",
            "-qscale", "0",
            "-y",
        ]

        if start and end:
            arg_list += ["-ss", str(start), "-to", str(end)]

        if crop:
            x, out_w, y, out_h = crop
            arg_list += ["-filter:v", f'"crop={out_w}:{out_h}:{x}:{y}"']

        arg_list += [output_path]

        command = ' '.join(arg_list)

        return_code = subprocess.call(command, shell=True)
        success = return_code == 0

        if not success:
            print('Command failed:', command)
        else:
            print('Command success:', command)

        return success


if __name__ == '__main__':
    maker = HDTFCropDatasetMaker(source_dir="input/txt")
    maker.process_data()
