# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

# Updated to account for UI changes from https://github.com/rkfg/audiocraft/blob/long/app.py
# also released under the MIT license.

import argparse
from concurrent.futures import ProcessPoolExecutor
import os
import subprocess as sp
from tempfile import NamedTemporaryFile
import time
import warnings

import torch
import gradio as gr

from audiocraft.data.audio_utils import convert_audio
from audiocraft.data.audio import audio_write
from audiocraft.models import MusicGen

from diffusers import StableDiffusionPipeline, EulerDiscreteScheduler
import torch
from scipy.io import wavfile


import random





from pinatapy import PinataPy

pinata = PinataPy('a03df7b80a0b9612cad4', '7a281586456175d7764d1ebc9b5b5d26c3a6f428128b9d5dd70f48ef22208c3d')

#TITLE = """<h4 align="center">  SynthNFT</h4>"""

TITLE="""
<div style="text-align: center; max-width: 500px; margin: 0 auto;">
        <div
        style="
            display: inline-flex;
            align-items: center;
            gap: 0.8rem;
            font-size: 1.75rem;
            margin-bottom: 10px;
        "
        >
        <h1 style="font-weight: 600; margin-bottom: 7px;">
            SynthNFT
        </h1>
        
        </div>
       <p> Democratising Art </p>
    </div>
"""

MODEL = None  # Last used model
save_name = None
IS_BATCHED = "facebook/MusicGen" in os.environ.get('SPACE_ID', '')
MAX_BATCH_SIZE = 12
BATCHED_DURATION = 15
INTERRUPTING = False
# We have to wrap subprocess call to clean a bit the log when using gr.make_waveform
_old_call = sp.call


def _call_nostderr(*args, **kwargs):
    # Avoid ffmpeg vomitting on the logs.
    kwargs['stderr'] = sp.DEVNULL
    kwargs['stdout'] = sp.DEVNULL
    _old_call(*args, **kwargs)


sp.call = _call_nostderr
# Preallocating the pool of processes.
pool = ProcessPoolExecutor(4)
pool.__enter__()


def interrupt():
    global INTERRUPTING
    INTERRUPTING = True


def make_waveform(*args, **kwargs):
    # Further remove some warnings.
    be = time.time()
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        out = gr.make_waveform(*args, **kwargs)
        print("Make a video took", time.time() - be)
        return out


def load_model(version='melody'):
    global MODEL
    print("Loading model", version)
    if MODEL is None or MODEL.name != version:
        MODEL = MusicGen.get_pretrained(version)

def load_stable_diffusion():
  global pipe
  model_id = "stabilityai/stable-diffusion-2"
  print("Loading model", model_id)
  scheduler = EulerDiscreteScheduler.from_pretrained(model_id, subfolder="scheduler")
  pipe = StableDiffusionPipeline.from_pretrained(model_id, scheduler=scheduler, revision="fp16", torch_dtype=torch.float16)
  pipe = pipe.to("cuda")


def _do_predictions(texts, melodies, duration, progress=False, **gen_kwargs):
    MODEL.set_generation_params(duration=duration, **gen_kwargs)
    print("new batch", len(texts), texts, [None if m is None else (m[0], m[1].shape) for m in melodies])
    be = time.time()
    processed_melodies = []
    target_sr = 32000
    target_ac = 1
    for melody in melodies:
        if melody is None:
            processed_melodies.append(None)
        else:
            sr, melody = melody[0], torch.from_numpy(melody[1]).to(MODEL.device).float().t()
            if melody.dim() == 1:
                melody = melody[None]
            melody = melody[..., :int(sr * duration)]
            melody = convert_audio(melody, sr, target_sr, target_ac)
            processed_melodies.append(melody)

    if any(m is not None for m in processed_melodies):
        outputs = MODEL.generate_with_chroma(
            descriptions=texts,
            melody_wavs=processed_melodies,
            melody_sample_rate=target_sr,
            progress=progress,
        )
    else:
        outputs = MODEL.generate(texts, progress=progress)

    outputs = outputs.detach().cpu().float()
    out_files = []
    for output in outputs:
        with NamedTemporaryFile("wb", suffix=".wav", delete=False) as file:
            audio_write(
                file.name, output, MODEL.sample_rate, strategy="loudness",
                loudness_headroom_db=16, loudness_compressor=True, add_suffix=False)
            out_files.append(pool.submit(make_waveform, file.name))
    res = [out_file.result() for out_file in out_files]
    print("batch finished", len(texts), time.time() - be)
    return res


def predict_batched(texts, melodies):
    max_text_length = 512
    texts = [text[:max_text_length] for text in texts]
    
    res = _do_predictions(texts, melodies, BATCHED_DURATION)
    return [res]
  
def txt2img(prompt):
  
  global save_name

  image = pipe(prompt, height=768, width=768, guidance_scale = 10).images[0]

  random_number = random.randint(1000000000, 9999999999)

  save_name = prompt +'_'+ str(random_number)+ '.png'
    
  image.save(save_name)

  return image



def predict_full(text, melody, duration, topk, topp, temperature, cfg_coef, progress=gr.Progress()):
    global INTERRUPTING
    INTERRUPTING = False

    global save_name

    if temperature < 0:
        raise gr.Error("Temperature must be >= 0.")
    if topk < 0:
        raise gr.Error("Topk must be non-negative.")
    if topp < 0:
        raise gr.Error("Topp must be non-negative.")

    topk = int(topk)
    #load_model(model)

    def _progress(generated, to_generate):
        progress((generated, to_generate))
        if INTERRUPTING:
            raise gr.Error("Interrupted.")
    MODEL.set_custom_progress_callback(_progress)

    outs = _do_predictions(
        [text], [melody], duration, progress=True,top_k=topk, top_p=topp, temperature=temperature, cfg_coef=cfg_coef)
    
    random_number = random.randint(1000000000, 9999999999)

    save_name = outs[0]



    return outs[0]

def push_pinata():
    global save_name
    metadata = pinata.pin_file_to_ipfs(save_name, '/')
    
    return metadata

def ui_full(launch_kwargs):

    with gr.Blocks() as interface:
        gr.HTML(TITLE)
        with gr.Tab("SynthMusic"):
            load_model('melody')
            with gr.Row():
                with gr.Column():
                    with gr.Row():
                        text = gr.Text(label="Input Text", interactive=True)
                        melody = gr.Audio(source="upload", type="numpy", label="Melody Condition (optional)", interactive=True)
                    with gr.Row():
                        submit = gr.Button("Submit")
                        _ = gr.Button("Interrupt").click(fn=interrupt, queue=False)
                    with gr.Row():
                        duration = gr.Slider(minimum=1, maximum=120, value=10, label="Duration", interactive=True)
                    with gr.Row():
                        topk = gr.Number(label="Feel", value=250, interactive=True)
                        topp = gr.Number(label="Speed", value=0, interactive=True)
                        temperature = gr.Number(label="Intensity", value=1.0, interactive=True)
                        cfg_coef = gr.Number(label="Classifier Free Guidance", value=3.0, interactive=True)
                with gr.Column():
                    output = gr.Video(label="Generated Music")
            submit.click(predict_full, inputs=[text, melody, duration, topk, topp, temperature, cfg_coef], outputs=[output])
            gr.Examples(
                fn=predict_full,
                examples=[
                    [
                        "An 80s driving pop song with heavy drums and synth pads in the background",
                        "./assets/bach.mp3"
                    ],
                    [
                        "A cheerful country song with acoustic guitars",
                        "./assets/bolero_ravel.mp3"
                    ],
                    [
                        "90s rock song with electric guitar and heavy drums",
                        None
                    ],
                    [
                        "a light and cheerly EDM track, with syncopated drums, aery pads, and strong emotions",
                        "./assets/bach.mp3"
                    ],
                    [
                        "lofi slow bpm electro chill with organic samples",
                        None
                    ],
                ],
                inputs=[text, melody],
                outputs=[output]
            )

            
        
        with gr.Tab("SynthImage"):
            load_stable_diffusion()
            text_input = gr.Textbox()
            image_output = gr.Image()
            image_button = gr.Button("Generate")
            image_button.click(txt2img, inputs=text_input, outputs=image_output)

        

        nft_button = gr.Button("NFT")
        nft_button.click(push_pinata)



        interface.queue().launch(**launch_kwargs)


if __name__ == "__main__":
    
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--listen',
        type=str,
        default='0.0.0.0' if 'SPACE_ID' in os.environ else '127.0.0.1',
        help='IP to listen on for connections to Gradio',
    )
    parser.add_argument(
        '--username', type=str, default='', help='Username for authentication'
    )
    parser.add_argument(
        '--password', type=str, default='', help='Password for authentication'
    )
    parser.add_argument(
        '--server_port',
        type=int,
        default=0,
        help='Port to run the server listener on',
    )
    parser.add_argument(
        '--inbrowser', action='store_true', help='Open in browser'
    )
    parser.add_argument(
        '--share', action='store_true', help='Share the gradio UI'
    )

    args = parser.parse_args()

    launch_kwargs = {}
    launch_kwargs['server_name'] = args.listen

    if args.username and args.password:
        launch_kwargs['auth'] = (args.username, args.password)
    if args.server_port:
        launch_kwargs['server_port'] = args.server_port
    if args.inbrowser:
        launch_kwargs['inbrowser'] = args.inbrowser
    if args.share:
        launch_kwargs['share'] = args.share

 
    ui_full(launch_kwargs)
