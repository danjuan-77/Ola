import os
import json
from tqdm import tqdm
import tempfile
tempfile.tempdir = "/share/nlp/tuwenming/projects/HAVIB/tmp"

from pathlib import Path
from typing import List, Optional
from moviepy.editor import (
    AudioFileClip,
    concatenate_audioclips,
    ImageClip,
    concatenate_videoclips,
)
import argparse


def concat_audio(audio_paths: List[str]) -> str:
    """
    Concatenate multiple audio files into one WAV file.
    Returns the path to the temp WAV file.
    """
    clips = [AudioFileClip(p) for p in audio_paths]
    final = concatenate_audioclips(clips)
    tmp = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
    out_path = tmp.name
    final.write_audiofile(out_path, fps=16000, logger=None)
    return out_path

def images_to_video(image_paths: List[str], duration: float, fps: int = 1) -> str:
    """
    Turn a list of images into a silent video of total `duration` seconds.
    Each image is shown for `duration / len(image_paths)` seconds.
    Returns the path to the temp MP4 file.
    """
    single_dur = duration / len(image_paths)
    clips = [ImageClip(p).set_duration(single_dur) for p in image_paths]
    video = concatenate_videoclips(clips, method="compose")
    tmp = tempfile.NamedTemporaryFile(suffix=".mp4", delete=False)
    out_path = tmp.name
    video.write_videofile(out_path, fps=fps, codec="libx264", audio=False, logger=None)
    return out_path

def images_and_audio_to_video(image_paths: List[str], audio_paths: List[str], fps: int = 1) -> str:
    """
    Concatenate audio_paths into one audio, then build a video from image_paths
    that matches the audio duration, and merge them.
    Returns the path to the temp MP4 file.
    """
    # 1) build the concatenated audio
    audio_path = concat_audio(audio_paths)
    audio_clip = AudioFileClip(audio_path)
    # 2) build video from images matching audio duration
    duration = audio_clip.duration
    vid_path = images_to_video(image_paths, duration, fps=fps)
    # 3) attach audio to video
    video_clip = AudioFileClip(audio_path)  # re-open to avoid MoviePy caching issues
    from moviepy.editor import VideoFileClip
    base_vid = VideoFileClip(vid_path)
    final = base_vid.set_audio(audio_clip)
    tmp = tempfile.NamedTemporaryFile(suffix=".mp4", delete=False)
    out_path = tmp.name
    final.write_videofile(out_path, fps=fps, codec="libx264", logger=None)
    return out_path 
    
def get_real_path(task_path: str, src_path: str) -> str:
    """传入taskpath和一些文件的path，构造文件的真实path

    Args:
        task_path (str): task path
        src_path (str): 每个文件的path

    Returns:
        str: 文件的真实path
    """
    temp_path = os.path.join(task_path, src_path)
    return os.path.normpath(temp_path)


if __name__ == "__main__":
    """
        直接根据predict方法进行传参
        audio
        audio list -> concat audio
        image
        image list -> concat to be a video
        video
        video+audio -> use audio in video
        image list + audio -> to be a video and add audio
        image + audio list -> concat audio
        
    """
    # task_path = "/share/nlp/tuwenming/projects/HAVIB/data/levels/level_1/LAQA"
    # task_path = "/share/nlp/tuwenming/projects/HAVIB/data/levels/level_1/LIQA"
    parser = argparse.ArgumentParser(
        description="Run prediction over a dataset described by data.json"
    )
    parser.add_argument(
        "--task_path",
        type=str,
        required=True,
        help="Path to the task folder containing data.json and media files"
    )
    args = parser.parse_args()
    task_path = args.task_path
    task_name = f"L{task_path.rsplit('/', 1)[0][-1]}_{task_path.rsplit('/', 1)[-1]}"
    
    data_json_path = os.path.join(task_path, "data.json")
    with open(data_json_path, "r", encoding='utf-8') as f:
        raw_data = json.load(f)
    print(">>>Finished load raw data...")
    parsed_data = []
    for item in raw_data:
        inp = item.get('input', {})
        question = inp.get('question', {})
        entry = {
            'id': item.get('id'),
            'task': item.get('task'),
            'text': question.get('text'),
            'audio_list': inp.get('audio_list', None),
            'image_list': inp.get('image_list', None),
            'video': inp.get('video', None)
        }
        parsed_data.append(entry)

    print(">>>Finished parse raw data...")    
    
    predict_results = []
    
    for data in tqdm(parsed_data, desc=f"test evaluating {task_name}"):
        _id = data['id']
        task = data['task']
        text = data['text']
        audio_list = (
            [get_real_path(task_path, p) for p in data["audio_list"]]
            if data["audio_list"] else None
        )
        image_list = (
            [get_real_path(task_path, p) for p in data["image_list"]]
            if data["image_list"] else None
        )
        video = (
            get_real_path(task_path, data['video'])
            if data['video'] else None
        )
        
        # Case 1: only audio_list
        if audio_list and not image_list and not video:
            if len(audio_list) > 1:
                audio_path = concat_audio(audio_list)
                output = audio_path
            else:
                audio_path = audio_list[0]
                output = audio_path

        # Case 2: only one image
        if image_list and not audio_list and not video:
            image_path = image_list[0]
            output = image_path
            

        # Case 3: only video
        if video and not audio_list and not image_list:
            output = video

        # Case 4: video + audio_list -> tell predict to use audio in video
        if video and audio_list:
            output = video

        # Case 5: image_list + audio_list -> build video with audio
        if image_list and audio_list and not video:
            video_path = images_and_audio_to_video(image_list, audio_list, fps=1)
            output = video_path

        # Case 6: audio_list + video (treat like case 4)
        if audio_list and video:
            output = video

        print(f"output:>>>>>>>>{output}")