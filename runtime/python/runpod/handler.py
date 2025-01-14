import runpod
import time
import os
import sys
import argparse
import logging
logging.getLogger('matplotlib').setLevel(logging.WARNING)
import uvicorn
import numpy as np
ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append('{}/../../..'.format(ROOT_DIR))
sys.path.append('{}/../../../third_party/Matcha-TTS'.format(ROOT_DIR))
from cosyvoice.cli.cosyvoice import CosyVoice, CosyVoice2
from cosyvoice.utils.file_utils import load_wav
import tempfile
import base64
from io import BytesIO
import torchaudio
import torch

def generate_data(model_output):
    for i in model_output:
        tts_audio = (i['tts_speech'].numpy() * (2 ** 15)).astype(np.int16).tobytes()
        yield tts_audio

def handler(event):
    input = event['input']
    mode = input.get('mode', 'sft')
    tts_text = input.get('tts_text', '')
    spk_id = input.get('spk_id', '')
    prompt_text = input.get('prompt_text', '')
    prompt_wav = input.get('prompt_wav', None)
    # get the type of the prompt_wav
    print(type(prompt_wav))
    instruct_text = input.get('instruct_text', '')

    try:
        if mode == 'sft':
            model_output = cosyvoice.inference_sft(tts_text, spk_id)
        elif mode == 'zero_shot':
            if not prompt_wav:
                raise ValueError("prompt_wav is required for zero_shot mode")
            # here prompt_wav is base64 encoded string, need to convert to file type
            prompt_wav = base64.b64decode(prompt_wav)
            
            prompt_wave_file = BytesIO(prompt_wav)
            prompt_speech_16k = load_wav(prompt_wave_file, 16000)
            model_output = cosyvoice.inference_zero_shot(tts_text, prompt_text, prompt_speech_16k)
        elif mode == 'cross_lingual':
            if not prompt_wav:
                raise ValueError("prompt_wav is required for cross_lingual mode")
            prompt_speech_16k = load_wav(prompt_wav, 16000)
            model_output = cosyvoice.inference_cross_lingual(tts_text, prompt_speech_16k)
        elif mode == 'instruct':
            model_output = cosyvoice.inference_instruct(tts_text, spk_id, instruct_text)
        else:
            raise ValueError(f"Invalid mode: {mode}")

        # Convert model output to bytes
        audio_chunks = []
        for output in model_output:
            audio_chunk = (output['tts_speech'].numpy() * (2 ** 15)).astype(np.int16).tobytes()
            audio_chunks.append(audio_chunk)

        # convert the entire audio_chunks to one base64 encoded string
        audio_data = b''.join(audio_chunks)
        audio_np = np.frombuffer(audio_data, dtype=np.int16)
        tts_speech = torch.from_numpy(audio_np).float().unsqueeze(0) / 32768.0  # Normalize to [-1, 1]

        # Save as MP3 in memory
        mp3_buffer = BytesIO()
        torchaudio.save(
            mp3_buffer,
            tts_speech,
            sample_rate=24000,  # Adjust sample rate to match your model's output
            format="mp3"
        )
        mp3_buffer.seek(0)
        mp3_data = base64.b64encode(mp3_buffer.read()).decode('utf-8')

        return {
            "status": "success",
            "audio": mp3_data
        }

    except Exception as e:
        return {
            "status": "error",
            "error": str(e)
        }

if __name__ == '__main__':
    cosyvoice = CosyVoice2('CosyVoice/pretrained_models/CosyVoice2-0.5B')
    runpod.serverless.start({'handler': handler})