import numpy as np
import gradio as gr
import time
from tqdm import tqdm
import secrets

from datetime import datetime


def train(images: list, token: str):
    tqdmlog = f'/tmp/tqdm_{secrets.token_hex(nbytes=4)}.log'
    file = open(tqdmlog, 'w')
    for _ in tqdm(range(10), file=file):
        time.sleep(0.3)

        with open(tqdmlog, 'r') as f:
            tqdm_str = f.readlines()[-1]
            tqdm_str += str(datetime.now())
        yield tqdm_str


def generate(model_name: str, prompt: str):
    time.sleep(2)
    return 'https://cdn.esawebb.org/archives/images/screen/potm2209a.jpg'


with gr.Blocks() as demo:
    # gr.Markdown("Flip text or image files using this demo.")
    with gr.Tab('Train model'):
        input_images = gr.File(label='Train images', file_count='multiple', file_types=['image'])
        token_input = gr.Text(label='Text token')
        # text_output = gr.Text(label='Output')
        tqdm_md = gr.Markdown()
        text_button = gr.Button("Train")

    text_button.click(train, inputs=[input_images, token_input], outputs=tqdm_md, queue=True)

    with gr.Tab('Inference'):
        with gr.Row():
            with gr.Column():
                text_input = gr.Text(label='Prompt')
                model_input = gr.Dropdown(choices=os.listdir('models'), value='model_1', label='Model')
            image_output = gr.Image(label='Generated image')
        gen_button = gr.Button("Generate")

    gen_button.click(generate, inputs=[text_input, model_input], outputs=image_output)

demo.queue()
demo.launch(share=True)
