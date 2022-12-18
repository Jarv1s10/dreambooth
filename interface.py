import os

import gradio as gr
from PIL import Image
import torch
from diffusers import StableDiffusionPipeline, DDIMScheduler

from train_dreambooth import main


def create_model_params(model_name: str, instance_prompt: str, class_prompt: str):
    model_params = {
        "pretrained_model_name_or_path": 'CompVis/stable-diffusion-v1-4',

        "instance_data_dir": f"data/{model_name}/instance",
        "class_data_dir": f"data/{model_name}/class",
        "output_dir": f'drive/MyDrive/dreambooth/{model_name}',
        "logging_dir": f'logs/{model_name}',

        "instance_prompt": instance_prompt,
        "class_prompt": class_prompt,

        "revision": 'fp16',
        "tokenizer_name": None,

        "with_prior_preservation": True,
        "prior_loss_weight": 1.,

        "seed": 2402,
        "resolution": 512,
        "train_batch_size": 1,
        "train_text_encoder": False,
        "mixed_precision": "fp16",
        "use_8bit_adam": True,
        "gradient_accumulation_steps": 1,
        "learning_rate": 2e-6,
        "lr_scheduler": "constant",
        "lr_warmup_steps": 0,
        "num_class_images": 50,
        "sample_batch_size":  4,
        "max_train_steps":  1000,
        "save_interval":  10000,
        "center_crop": True,
        "num_train_epochs": 1000,
        "checkpointing_steps": 500,
        "resume_from_checkpoint": None,
        "gradient_checkpointing": True,
        "scale_lr": False,
        "adam_beta1": 0.9,
        "adam_beta2": 0.999,
        "adam_weight_decay": 1e-2,
        "adam_epsilon": 1e-8,
        "max_grad_norm": 1.,
        "push_to_hub": False,
        "hub_token": "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",
        "hub_model_id": None
    }

    return model_params


def train(model_name: str, images: list, instance_prompt: str, class_prompt: str):
    model_params = create_model_params(model_name, instance_prompt, class_prompt)

    os.makedirs(model_params['output_dir'], exist_ok=True)
    os.makedirs(model_params['instance_data_dir'], exist_ok=True)
    os.makedirs(model_params['class_data_dir'], exist_ok=True)
    os.makedirs(model_params['logging_dir'], exist_ok=True)

    for img in images:
        Image.open(img.name).save(f"{model_params['instance_data_dir']}/{os.path.split(img.name)[-1]}")

    main(model_params)


def inference(model_path, prompt, num_samples=4, height=512, width=512, num_inference_steps=50, guidance_scale=7.5):
    scheduler = DDIMScheduler(beta_start=0.00085, beta_end=0.012, beta_schedule="scaled_linear", clip_sample=False, set_alpha_to_one=False)
    pipe = StableDiffusionPipeline.from_pretrained('drive/MyDrive/dreambooth/' + model_path,
                                                   scheduler=scheduler, safety_checker=None, torch_dtype=torch.float16).to("cuda")

    with torch.autocast("cuda"), torch.inference_mode():
        return pipe(
            prompt, height=int(height), width=int(width),
            num_images_per_prompt=int(num_samples),
            num_inference_steps=int(num_inference_steps), guidance_scale=guidance_scale,
            generator=None
        ).images


def interface():
    with gr.Blocks() as demo:
        gr.Markdown("# Personalize your AI Image Generation")
        with gr.Tab('Train model'):
            with gr.Column():
                model_name = gr.Text(label='Model name', placeholder='e.g. `my_dreambooth_model`')
                input_images = gr.File(label='Train images', file_count='multiple', file_types=['image'])
                instance_prompt = gr.Text(label='Instance prompt', placeholder='e.g. `photo of a [V] person`')
                class_prompt = gr.Text(label='Class prompt', placeholder='e.g. `photo of a person`')
                text_button = gr.Button("Train")

        text_button.click(train, inputs=[model_name, input_images, instance_prompt, class_prompt], queue=True)

        with gr.Tab('Inference'):
            with gr.Row():
                with gr.Column():
                    text_input = gr.Text(label='Prompt')

                    os.makedirs('models', exist_ok=True)
                    model_input = gr.Dropdown(choices=os.listdir('drive/MyDrive/dreambooth/'), label='Model')
                    gen_button = gr.Button("Generate")
                images_output = gr.Gallery(label='Generated images')

        gen_button.click(inference, inputs=[model_input, text_input], outputs=images_output)

    demo.queue()
    demo.launch(share=True)


if __name__ == '__main__':
    interface()
