import os
import glob
import json
from moviepy.editor import AudioFileClip, VideoFileClip

video_path = "data/video"
video_files = glob.glob(os.path.join(video_path, "*.mp4"))

for video_file in video_files:
    audio_clip = AudioFileClip(video_file, fps=16000)
    audio_clip.write_audiofile(video_files.replace(".mp4", ".wav").replace("video", "audio"))
    audio_clip.close()

scene_split = json.load(open('data/scene_detect/scene_split_new.json'))
for clip_id in list(scene_split.keys()):
    video_clip = VideoFileClip(f'data/video/{clip_id}.mp4', audio=False)
    for i, item in enumerate(scene_split[clip_id]):
        for j in range(8):
            filename = f'data/frame/{clip_id}_{i}_{j}.jpg'
            if not os.path.exists(filename):
                video_clip.save_frame(filename, t=(item['start_time'] + (j + 1) * (item['end_time'] - item['start_time']) / 9))
    video_clip.close()