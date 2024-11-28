# Copyright (2024) Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
from PIL import Image
from typing import List
import torch
from transformers import WhisperFeatureExtractor
from transformers.models.llava import LlavaProcessor
import soundfile as sf
import librosa
import re
import numpy as np
from torch.nn.utils.rnn import pad_sequence
from .utils import sample_image, sample_video


class CustomImageProcessor:
    def __init__(self, processor) -> None:
        self.processor = processor

    def __call__(self, images: List[Image.Image], do_padding=False) -> torch.Tensor:
        if do_padding:
            images = [self.expand2square(
                img,
                tuple(int(x * 255) for x in self.processor.image_processor.image_mean)
            ) for img in images]
        else:
            images = [self.resize2square(img) for img in images]
        images_pixel = self.processor(text="", images=images, return_tensors="pt")['pixel_values']
        return images_pixel  # [num_images, 3, 336, 336]

    def expand2square(self, pil_img, background_color):
        width, height = pil_img.size
        if width == height:
            return pil_img
        elif width > height:
            result = Image.new(pil_img.mode, (width, width), background_color)
            result.paste(pil_img, (0, (width - height) // 2))
            return result
        else:
            result = Image.new(pil_img.mode, (height, height), background_color)
            result.paste(pil_img, ((height - width) // 2, 0))
            return result

    def resize2square(self, pil_img: Image.Image):
        width, height = pil_img.size
        pil_img = pil_img.resize((max(width, height), max(width, height)))
        return pil_img


class AudioProcessor:
    def __init__(self, whisper_path):
        self.wav_processor = WhisperFeatureExtractor.from_pretrained(whisper_path)
        self.target_sample_rate = 16000

    def read_audio(self, audio_path, start_time=None, end_time=None):
        audio, sr = sf.read(audio_path)
        if len(audio.shape) > 1: # stereo to mono
            audio = audio[:, 0]
        if start_time is not None or end_time is not None:
            start_sample = int(start_time * sr)
            end_sample = int(end_time * sr)
            audio = audio[start_sample:end_sample]
        return audio, sr
    
    def load_audio_file(self, audio_path, start_time=None, end_time=None):
        if isinstance(audio_path, str):
            audio_path = [audio_path]
            start_time = [start_time]
            end_time = [end_time]
        else:
            if start_time is None:
                start_time = [None] * len(audio_path)
            if end_time is None:
                end_time = [None] * len(audio_path)
        audio, sr = self.read_audio(audio_path[0], start_time[0], end_time[0])
        for idx in range(1, len(audio_path)):
            expand_audio, expand_sr = self.read_audio(audio_path[idx], start_time[idx], end_time[idx])
            assert sr==expand_sr, "audio sample rate is different!"
            sil = np.zeros(sr, dtype=float)
            audio = np.concatenate((audio, sil, expand_audio), axis=0)
        if len(audio) < sr:
            sil = np.zeros(sr - len(audio), dtype=float)
            audio = np.concatenate((audio, sil), axis=0)
        if sr != self.target_sample_rate:
            audio = librosa.resample(audio, orig_sr=sr, target_sr=self.target_sample_rate)
            sr = self.target_sample_rate
        audio_spectrogram = self.wav_processor(audio, sampling_rate=sr, return_tensors="pt")["input_features"].squeeze()
        return {
            "audio_spectrogram": audio_spectrogram,
            "audio_wav": audio,
        }
    
    def batch_collate(self, samples):
        audio_spectrogram = [s["audio_spectrogram"] for s in samples]
        audio_spectrogram = torch.stack(audio_spectrogram, dim=0)
        audio_wav = [torch.from_numpy(s["audio_wav"]) for s in samples]
        wav_length = torch.tensor([len(s["audio_wav"]) for s in samples])
        audio_wav = pad_sequence(audio_wav, batch_first=True, padding_value=0)
        audio_wav_mask = torch.arange(audio_wav.size(1)).unsqueeze(0) >= wav_length.unsqueeze(1)
        return {
            "audio_spectrogram": audio_spectrogram.half(),
            "audio_wav": audio_wav.half(),
            "audio_wav_mask": audio_wav_mask.half(),
        }
    
    def dummy_audio_input(self):
        sr = self.target_sample_rate
        audio = np.zeros(sr, dtype=float)
        audio_spectrogram = self.wav_processor(audio, sampling_rate=sr, return_tensors="pt")["input_features"].squeeze()
        return self.batch_collate([{
            "audio_spectrogram": audio_spectrogram,
            "audio_wav": audio,
        }])


class Processor(object):
    def __init__(
            self,
            model_name_or_path,
            config,
            max_n_frames=8,
            max_seq_len=None,
            add_sep=False,
            do_image_padding=False,
        ):
        self.max_n_frames = max_n_frames
        self.max_seq_len = max_seq_len,
        self.add_sep = add_sep
        self.do_image_padding = do_image_padding
        if not self.do_image_padding:
            print(f"### do_image_padding is set as False, images will be resized directly!")

        self.setup(model_name_or_path, config)
    
    def setup(self, model_name_or_path, config):
        sub_processor = LlavaProcessor.from_pretrained(
            model_name_or_path,
            padding_side='left',
            trust_remote_code=True,
        )
        self.processor = CustomImageProcessor(sub_processor)
        self.tokenizer = sub_processor.tokenizer
        # self.pad_collator = DataCollatorForSeq2Seq(self.tokenizer, padding='longest')
        self.sep_id = self.tokenizer.sep_token_id
        self.pad_id = self.tokenizer.pad_token_id
        self.eos_id = self.tokenizer.eos_token_id
        self.audio_processor = AudioProcessor(whisper_path=config.audio_config["whisper_path"])

        if self.sep_id is None:
            self.add_sep = False
        if not self.max_seq_len:
            self.max_seq_len = self.tokenizer.model_max_length

    def get_pixel_values(self, images):
        if images is not None and len(images) > 0:
            pixel_values = self.processor(images=images, do_padding=self.do_image_padding)
        else:
            pixel_values = None
        return pixel_values

    def get_text_inputs(self, text):
        prompt_ids = self.tokenizer.encode(text, add_special_tokens=True)
        prompt_ids = torch.tensor(prompt_ids, dtype=torch.long).unsqueeze(dim=0)
        return prompt_ids

    def __call__(self, data):
        # load image/video
        images = []
        if "video" in data:
            for video in data["video"]:
                if "video_file" in video:
                    images = sample_video(video["video_file"], video["n_frames"], video["start_time"], video["end_time"])
                elif "image_file" in video:
                    for i in video["image_file"]:
                        images += sample_image(i)
                else:
                    assert True == False, "video data must have either video_file or image_file"

        # load audio
        audios = []
        if "audio" in data:
            for audio in data["audio"]:
                audios.append(self.audio_processor.load_audio_file(
                    audio_path=audio['audio_file'],
                    start_time=audio['start_time'],
                    end_time=audio['end_time'],
                ))
        
        # process prompt
        prompt = data["text"]["prompt"]
        assert prompt.count("<image>") == len(images), "{} != {}, wrong images numbers".format(prompt.count("<image>"), len(images))
        assert prompt.count("<audio>") == len(audios), "{} != {}, wrong audios numbers".format(prompt.count("<audio>"), len(audios))
        mm_types_matches = re.compile(r'<image>|<audio>').findall(prompt)
        mm_types = [1 if match == '<image>' else 2 if match == '<audio>' else 0 for match in mm_types_matches]
        prompt = prompt.replace("<audio>", "<image>")

        inputs = {
            "input_ids": self.get_text_inputs(prompt),
            "pixel_values": self.get_pixel_values(images),
            "mm_types": mm_types
        }
        inputs.update(self.audio_processor.batch_collate(audios))
        return inputs
