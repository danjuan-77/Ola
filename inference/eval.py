from email.mime import audio
import os
import tempfile
tempfile.tempdir = "/share/nlp/tuwenming/projects/HAVIB/tmp"
from pathlib import Path
os.environ['LOWRES_RESIZE'] = '384x32'
os.environ['HIGHRES_BASE'] = '0x32'
os.environ['VIDEO_RESIZE'] = "0x64"
os.environ['VIDEO_MAXRES'] = "480"
os.environ['VIDEO_MINRES'] = "288"
os.environ['MAXRES'] = '1536'
os.environ['MINRES'] = '0'
os.environ['FORCE_NO_DOWNSAMPLE'] = '1'
os.environ['LOAD_VISION_EARLY'] = '1'
os.environ['PAD2STRIDE'] = '1'

import gradio as gr
import torch
import re
from decord import VideoReader, cpu
from PIL import Image
import numpy as np
import transformers
import moviepy.editor as mp
from typing import Dict, Optional, Sequence, List
import librosa
import whisper
from ola.conversation import conv_templates, SeparatorStyle
from ola.model.builder import load_pretrained_model
from ola.datasets.preprocess import tokenizer_image_token, tokenizer_speech_image_token, tokenizer_speech_question_image_token, tokenizer_speech_token
from ola.mm_utils import KeywordsStoppingCriteria, process_anyres_video, process_anyres_highres_image
from ola.constants import IGNORE_INDEX, DEFAULT_IMAGE_TOKEN, IMAGE_TOKEN_INDEX, DEFAULT_SPEECH_TOKEN, SPEECH_TOKEN_INDEX
import json
from tqdm import tqdm
from typing import List, Optional
from moviepy.editor import (
    AudioFileClip,
    concatenate_audioclips,
    ImageClip,
    concatenate_videoclips,
)

# import argparse

# parser = argparse.ArgumentParser()
# parser.add_argument('--model_path', type=str, default='THUdyh/Ola-7b')
# parser.add_argument('--text', type=str, default=None)
# parser.add_argument('--audio_path', type=str, default=None)
# parser.add_argument('--image_path', type=str, default=None)
# parser.add_argument('--video_path', type=str, default=None)
# parser.add_argument("--use_audio_in_the_video", action="store_true", help="Use Speech in the video")

# args = parser.parse_args()

model_path = "/share/nlp/tuwenming/models/THUdyh/Ola-7b"
use_audio_in_the_video = True
tokenizer, model, image_processor, _ = load_pretrained_model(model_path, None)
model = model.to('cuda').eval()
model = model.bfloat16()

cur_dir = os.path.dirname(os.path.abspath(__file__))

def load_audio(audio_file_name):
    speech_wav, samplerate = librosa.load(audio_file_name, sr=16000)
    if len(speech_wav.shape) > 1:
        speech_wav = speech_wav[:, 0]
    speech_wav = speech_wav.astype(np.float32)
    CHUNK_LIM = 480000
    SAMPLE_RATE = 16000
    speechs = []
    speech_wavs = []

    if len(speech_wav) <= CHUNK_LIM:
        speech = whisper.pad_or_trim(speech_wav)
        speech_wav = whisper.pad_or_trim(speech_wav)
        speechs.append(speech)
        speech_wavs.append(torch.from_numpy(speech_wav).unsqueeze(0))
    else:
        for i in range(0, len(speech_wav), CHUNK_LIM):
            chunk = speech_wav[i : i + CHUNK_LIM]
            if len(chunk) < CHUNK_LIM:
                chunk = whisper.pad_or_trim(chunk)
            speechs.append(chunk)
            speech_wavs.append(torch.from_numpy(chunk).unsqueeze(0))
    mels = []
    for chunk in speechs:
        chunk = whisper.log_mel_spectrogram(chunk, n_mels=128).permute(1, 0).unsqueeze(0)
        mels.append(chunk)

    mels = torch.cat(mels, dim=0)
    speech_wavs = torch.cat(speech_wavs, dim=0)
    if mels.shape[0] > 25:
        mels = mels[:25]
        speech_wavs = speech_wavs[:25]

    speech_length = torch.LongTensor([mels.shape[1]] * mels.shape[0])
    speech_chunks = torch.LongTensor([mels.shape[0]])
    return mels, speech_length, speech_chunks, speech_wavs

def extract_audio(videos_file_path):
    my_clip = mp.VideoFileClip(videos_file_path)
    return my_clip.audio

# image_path = args.image_path
# audio_path = args.audio_path
# video_path = args.video_path
# text = args.text

def predict(model, text, video_path=None, image_path=None, audio_path=None, use_audio_in_the_video=False):
    USE_SPEECH=False
    
    if video_path is not None:
        modality = "video"
        visual = video_path
        assert image_path is None

    elif image_path is not None:
        visual = image_path
        modality = "image"
        assert video_path is None

    elif audio_path is not None:
        modality = "text"


    # If an external audio file is provided, always use it
    if audio_path:
        USE_SPEECH = True
    # If video input and the flag is set, use the video's audio
    elif video_path and use_audio_in_the_video:
        USE_SPEECH = True

    speechs = []
    speech_lengths = []
    speech_wavs = []
    speech_chunks = []
    if modality == "video":
        vr = VideoReader(visual, ctx=cpu(0))
        total_frame_num = len(vr)
        fps = round(vr.get_avg_fps())
        uniform_sampled_frames = np.linspace(0, total_frame_num - 1, 64, dtype=int)
        frame_idx = uniform_sampled_frames.tolist()
        spare_frames = vr.get_batch(frame_idx).asnumpy()
        video = [Image.fromarray(frame) for frame in spare_frames]
    elif modality == "image":
        image = [Image.open(visual)]
        image_sizes = [image[0].size]
    else:
        images = [torch.zeros(1, 3, 224, 224).to(dtype=torch.bfloat16, device='cuda', non_blocking=True)]
        images_highres = [torch.zeros(1, 3, 224, 224).to(dtype=torch.bfloat16, device='cuda', non_blocking=True)]
        image_sizes = [(224, 224)]


    if USE_SPEECH and audio_path:
        audio_path = audio_path
        speech, speech_length, speech_chunk, speech_wav = load_audio(audio_path)
        speechs.append(speech.bfloat16().to('cuda'))
        speech_lengths.append(speech_length.to('cuda'))
        speech_chunks.append(speech_chunk.to('cuda'))
        speech_wavs.append(speech_wav.to('cuda'))
        print('load audio')
    elif USE_SPEECH and not audio_path:
        # parse audio in the video
        audio = extract_audio(visual)
        audio.write_audiofile("./video_audio.wav")
        video_audio_path = './video_audio.wav'
        speech, speech_length, speech_chunk, speech_wav = load_audio(video_audio_path)
        speechs.append(speech.bfloat16().to('cuda'))
        speech_lengths.append(speech_length.to('cuda'))
        speech_chunks.append(speech_chunk.to('cuda'))
        speech_wavs.append(speech_wav.to('cuda'))
    else:
        speechs = [torch.zeros(1, 3000, 128).bfloat16().to('cuda')]
        speech_lengths = [torch.LongTensor([3000]).to('cuda')]
        speech_wavs = [torch.zeros([1, 480000]).to('cuda')]
        speech_chunks = [torch.LongTensor([1]).to('cuda')]

    conv_mode = "qwen_1_5"
    if text:
        qs = text
    else:
        qs = ''

    if USE_SPEECH and audio_path and image_path: # image + speech instruction
        qs = DEFAULT_IMAGE_TOKEN + "\n" + "User's question in speech: " + DEFAULT_SPEECH_TOKEN + '\n'
    elif USE_SPEECH and video_path: # video + audio
        qs = DEFAULT_SPEECH_TOKEN + DEFAULT_IMAGE_TOKEN + "\n" + qs
    elif USE_SPEECH and audio_path: # audio + text
        qs = DEFAULT_SPEECH_TOKEN + "\n" + qs
    elif image_path or video_path: # image / video
        qs = DEFAULT_IMAGE_TOKEN + "\n" + qs
    elif text: # text
        qs = qs

    conv = conv_templates[conv_mode].copy()
    conv.append_message(conv.roles[0], qs)
    conv.append_message(conv.roles[1], None)
    prompt = conv.get_prompt()
    if USE_SPEECH and audio_path and image_path: # image + speech instruction
        input_ids = tokenizer_speech_question_image_token(prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt").unsqueeze(0).to('cuda')
    elif USE_SPEECH and video_path: # video + audio
        input_ids = tokenizer_speech_image_token(prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt").unsqueeze(0).to('cuda')
    elif USE_SPEECH and audio_path: # audio + text
        input_ids = tokenizer_speech_token(prompt, tokenizer, SPEECH_TOKEN_INDEX, return_tensors="pt").unsqueeze(0).to('cuda')
    else:
        input_ids = tokenizer_image_token(prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt").unsqueeze(0).to('cuda')

    if modality == "video":
        video_processed = []
        for idx, frame in enumerate(video):
            image_processor.do_resize = False
            image_processor.do_center_crop = False
            frame = process_anyres_video(frame, image_processor)

            if frame_idx is not None and idx in frame_idx:
                video_processed.append(frame.unsqueeze(0))
            elif frame_idx is None:
                video_processed.append(frame.unsqueeze(0))
        
        if frame_idx is None:
            frame_idx = np.arange(0, len(video_processed), dtype=int).tolist()
        
        video_processed = torch.cat(video_processed, dim=0).bfloat16().to("cuda")
        video_processed = (video_processed, video_processed)

        video_data = (video_processed, (384, 384), "video")
    elif modality == "image":
        image_processor.do_resize = False
        image_processor.do_center_crop = False
        image_tensor, image_highres_tensor = [], []
        for visual in image:
            image_tensor_, image_highres_tensor_ = process_anyres_highres_image(visual, image_processor)
            image_tensor.append(image_tensor_)
            image_highres_tensor.append(image_highres_tensor_)
        if all(x.shape == image_tensor[0].shape for x in image_tensor):
            image_tensor = torch.stack(image_tensor, dim=0)
        if all(x.shape == image_highres_tensor[0].shape for x in image_highres_tensor):
            image_highres_tensor = torch.stack(image_highres_tensor, dim=0)
        if type(image_tensor) is list:
            image_tensor = [_image.bfloat16().to("cuda") for _image in image_tensor]
        else:
            image_tensor = image_tensor.bfloat16().to("cuda")
        if type(image_highres_tensor) is list:
            image_highres_tensor = [_image.bfloat16().to("cuda") for _image in image_highres_tensor]
        else:
            image_highres_tensor = image_highres_tensor.bfloat16().to("cuda")

    pad_token_ids = 151643

    attention_masks = input_ids.ne(pad_token_ids).long().to('cuda')
    stop_str = conv.sep if conv.sep_style != SeparatorStyle.TWO else conv.sep2
    keywords = [stop_str]
    stopping_criteria = KeywordsStoppingCriteria(keywords, tokenizer, input_ids)

    gen_kwargs = {}

    if "max_new_tokens" not in gen_kwargs:
        gen_kwargs["max_new_tokens"] = 1024
    if "temperature" not in gen_kwargs:
        gen_kwargs["temperature"] = 0.2
    if "top_p" not in gen_kwargs:
        gen_kwargs["top_p"] = None
    if "num_beams" not in gen_kwargs:
        gen_kwargs["num_beams"] = 1

    with torch.inference_mode():
        if modality == "video":
            output_ids = model.generate(
                inputs=input_ids,
                images=video_data[0][0],
                images_highres=video_data[0][1],
                modalities=video_data[2],
                speech=speechs,
                speech_lengths=speech_lengths,
                speech_chunks=speech_chunks,
                speech_wav=speech_wavs,
                attention_mask=attention_masks,
                use_cache=True,
                stopping_criteria=[stopping_criteria],
                do_sample=True if gen_kwargs["temperature"] > 0 else False,
                temperature=gen_kwargs["temperature"],
                top_p=gen_kwargs["top_p"],
                num_beams=gen_kwargs["num_beams"],
                max_new_tokens=gen_kwargs["max_new_tokens"],
            )
        elif modality == "image":
            output_ids = model.generate(
                inputs=input_ids,
                images=image_tensor,
                images_highres=image_highres_tensor,
                image_sizes=image_sizes,
                modalities=['image'],
                speech=speechs,
                speech_lengths=speech_lengths,
                speech_chunks=speech_chunks,
                speech_wav=speech_wavs,
                attention_mask=attention_masks,
                use_cache=True,
                stopping_criteria=[stopping_criteria],
                do_sample=True if gen_kwargs["temperature"] > 0 else False,
                temperature=gen_kwargs["temperature"],
                top_p=gen_kwargs["top_p"],
                num_beams=gen_kwargs["num_beams"],
                max_new_tokens=gen_kwargs["max_new_tokens"],
            )
        elif modality == "text":
            output_ids = model.generate(
                input_ids,
                images=images,
                images_highres=images_highres,
                image_sizes=image_sizes,
                modalities=['text'],
                speech=speechs,
                speech_lengths=speech_lengths,
                speech_chunks=speech_chunks,
                speech_wav=speech_wavs,
                attention_mask=attention_masks,
                use_cache=True,
                stopping_criteria=[stopping_criteria],
                do_sample=True if gen_kwargs["temperature"] > 0 else False,
                temperature=gen_kwargs["temperature"],
                top_p=gen_kwargs["top_p"],
                num_beams=gen_kwargs["num_beams"],
                max_new_tokens=gen_kwargs["max_new_tokens"],
                )

    outputs = tokenizer.batch_decode(output_ids, skip_special_tokens=True)[0]
    outputs = outputs.strip()
    if outputs.endswith(stop_str):
        outputs = outputs[:-len(stop_str)]
    outputs = outputs.strip()

    return outputs

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
    task_path = "/share/nlp/tuwenming/projects/HAVIB/data/levels/level_1/LIQA"
    task_name = f"L{task_path.rsplit('/', 1)[0][-1]}_{task_path.rsplit('/', 1)[-1]}"
    model_name = "ola"
    save_prediction_json = f'/share/nlp/tuwenming/projects/HAVIB/eval/user_outputs/{model_name}/tasks/{task_name}.json'
    os.makedirs(os.path.dirname(save_prediction_json), exist_ok=True)
    print('>>> save res to:', save_prediction_json)
    
    
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
    
    predictions = []
    
    for data in tqdm(parsed_data):
        _id = data['id']
        _task = data['task']
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
                output = predict(model, text, audio_path=audio_path)
            else:
                audio_path = audio_list[0]
                output = predict(model, text, audio_path=audio_path)

        # Case 2: only one image
        if image_list and not audio_list and not video:
            image_path = image_list[0]
            output = predict(model, text, image_path=image_path)
            

        # Case 3: only video
        if video and not audio_list and not image_list:
            output = predict(model, text, video_path=video)

        # Case 4: video + audio_list -> tell predict to use audio in video
        if video and audio_list:
            output = predict(model, text, video_path=video, use_audio_in_the_video=True)

        # Case 5: image_list + audio_list -> build video with audio
        if image_list and audio_list and not video:
            video_path = images_and_audio_to_video(image_list, audio_list, fps=1)
            output = predict(model, text, video_path=video_path, use_audio_in_the_video=True)

        # Case 6: audio_list + video (treat like case 4)
        if audio_list and video:
            output = predict(model, text, video_path=video, use_audio_in_the_video=True)

        # print(f"output:>>>>>>>>{output}")
        pred_record = {
            "task": _task,
            "id": _id,
            "predict": output,
        }
        predictions.append(pred_record)
        print('>>> ans=:', pred_record)
    
    
    with open(save_prediction_json, 'w', encoding='utf-8') as json_file:
        json.dump(predictions, json_file, ensure_ascii=False, indent=4)
    
    # vid_path = "./data/test.mp4"
    # text = "Describe what she say"
    # predict(model, text, vid_path)
    
    # predict(model, text, vid_path)
    
    # predict(model, text, vid_path)
    
    # predict(model, text, vid_path)
    
    # predict(model, text, vid_path)
    
    # predict(model, text, vid_path)
    