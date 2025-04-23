#!/bin/bash

# # Text & Image Understanding
# python3 inference/infer.py --model_path "/share/nlp/tuwenming/models/THUdyh/Ola-7b" \
#     --image_path "./data/image.png" \
#     --text "Describe this image."
# echo "Text & Image Understanding Finished"

# # Text & Video Understanding
# python3 inference/infer.py --model_path "/share/nlp/tuwenming/models/THUdyh/Ola-7b" \
#     --video_path "./data/test.mp4"\
#     --text "Describe this video."
# echo "Text & Video Understanding Finished"


# # Text & Audio Understanding
# python3 inference/infer.py --model_path "/share/nlp/tuwenming/models/THUdyh/Ola-7b" \
#     --audio_path "./data/sample.wav" \
#     --text "Describe this audio."
# echo "Text & Audio Understanding Finished"


# Audio & Image Understanding
python3 inference/infer.py --model_path "/share/nlp/tuwenming/models/THUdyh/Ola-7b" \
    --audio_path "./data/vision_qa_image.jpg"\
    --audio_path "./data/vision_qa_audio.wav"
echo "Audio & Image Understandin Finished"
