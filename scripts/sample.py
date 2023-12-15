import contextlib
import os
import datetime
import time
import sys
script_path = os.path.abspath(__file__)
sys.path.append(os.path.dirname(os.path.dirname(script_path)))
from absl import app, flags
from ml_collections import config_flags
from accelerate import Accelerator
from accelerate.utils import set_seed, ProjectConfiguration
from accelerate.logging import get_logger
from diffusers import StableDiffusionPipeline, DDIMScheduler
from diffusers.loaders import AttnProcsLayers
from diffusers.models.attention_processor import LoRAAttnProcessor
import numpy as np
import d3po_pytorch.prompts
import d3po_pytorch.rewards
from d3po_pytorch.diffusers_patch.pipeline_with_logprob import pipeline_with_logprob
import torch
from functools import partial
import tqdm
from PIL import Image
import json
import pickle
from scripts.utils import post_processing

tqdm = partial(tqdm.tqdm, dynamic_ncols=True)

now_time = datetime.datetime.now().strftime('%Y-%m-%d-%H-%M-%S')
save_dir = './data'
save_dir = os.path.join(save_dir, now_time)

FLAGS = flags.FLAGS
NUM_PER_PROMPT = 7
config_flags.DEFINE_config_file("config", "config/base.py", "Training configuration.")

logger = get_logger(__name__)


def main(_):
    # basic Accelerate and logging setup
    config = FLAGS.config

    unique_id = datetime.datetime.now().strftime("%Y.%m.%d_%H.%M.%S")
    if not config.run_name:
        config.run_name = unique_id
    else:
        config.run_name += "_" + unique_id

    if config.resume_from:
        print("loading model. Please Wait.")
        config.resume_from = os.path.normpath(os.path.expanduser(config.resume_from))
        if "checkpoint_" not in os.path.basename(config.resume_from):
            # get the most recent checkpoint in this directory
            checkpoints = list(filter(lambda x: "checkpoint_" in x, os.listdir(config.resume_from)))
            if len(checkpoints) == 0:
                raise ValueError(f"No checkpoints found in {config.resume_from}")
            config.resume_from = os.path.join(
                config.resume_from,
                sorted(checkpoints, key=lambda x: int(x.split("_")[-1]))[-1],
            )
        print("load successfully!")

    # number of timesteps within each trajectory to train on
    num_train_timesteps = int(config.sample.num_steps * config.train.timestep_fraction)

    accelerator_config = ProjectConfiguration(
        project_dir=os.path.join(config.logdir, config.run_name),
        automatic_checkpoint_naming=True,
        total_limit=config.num_checkpoint_limit,
    )

    accelerator = Accelerator(
        log_with="wandb",
        mixed_precision=config.mixed_precision,
        project_config=accelerator_config,
        # we always accumulate gradients across timesteps; we want config.train.gradient_accumulation_steps to be the
        # number of *samples* we accumulate across, so we need to multiply by the number of training timesteps to get
        # the total number of optimizer steps to accumulate across.
        gradient_accumulation_steps=config.train.gradient_accumulation_steps * num_train_timesteps,
    )

    # set a random seed
    ramdom_seed = np.random.randint(0,100000)
    set_seed(ramdom_seed, device_specific=True)

    # load scheduler, tokenizer and models.
    pipeline = StableDiffusionPipeline.from_pretrained(config.pretrained.model, torch_dtype=torch.float16)
    # freeze parameters of models to save more memory
    pipeline.vae.requires_grad_(False)
    pipeline.text_encoder.requires_grad_(False)
    pipeline.unet.requires_grad_(not config.use_lora)
    # disable safety checker
    pipeline.safety_checker = None
    # make the progress bar nicer
    pipeline.set_progress_bar_config(
        position=1,
        disable=not accelerator.is_local_main_process,
        leave=False,
        desc="Timestep",
        dynamic_ncols=True,
    )
    # switch to DDIM scheduler
    pipeline.scheduler = DDIMScheduler.from_config(pipeline.scheduler.config)
    total_image_num_per_gpu = config.sample.batch_size * config.sample.num_batches_per_epoch * NUM_PER_PROMPT
    global_idx = accelerator.process_index * total_image_num_per_gpu 
    local_idx = 0
    # For mixed precision training we cast all non-trainable weigths (vae, non-lora text_encoder and non-lora unet) to half-precision
    # as these weights are only used for inference, keeping weights in full precision is not required.
    inference_dtype = torch.float32
    if accelerator.mixed_precision == "fp16":
        inference_dtype = torch.float16
    elif accelerator.mixed_precision == "bf16":
        inference_dtype = torch.bfloat16

    # Move unet, vae and text_encoder to device and cast to inference_dtype
    pipeline.vae.to(accelerator.device, dtype=inference_dtype)
    pipeline.text_encoder.to(accelerator.device, dtype=inference_dtype)
    
    if config.use_lora:
        pipeline.unet.to(accelerator.device, dtype=inference_dtype)

    if config.use_lora:
        # Set correct lora layers
        lora_attn_procs = {}
        for name in pipeline.unet.attn_processors.keys():
            cross_attention_dim = (
                None if name.endswith("attn1.processor") else pipeline.unet.config.cross_attention_dim
            )
            if name.startswith("mid_block"):
                hidden_size = pipeline.unet.config.block_out_channels[-1]
            elif name.startswith("up_blocks"):
                block_id = int(name[len("up_blocks.")])
                hidden_size = list(reversed(pipeline.unet.config.block_out_channels))[block_id]
            elif name.startswith("down_blocks"):
                block_id = int(name[len("down_blocks.")])
                hidden_size = pipeline.unet.config.block_out_channels[block_id]

            lora_attn_procs[name] = LoRAAttnProcessor(hidden_size=hidden_size, cross_attention_dim=cross_attention_dim)
        for name in pipeline.unet.attn_processors.keys():
            cross_attention_dim = (
                None if name.endswith("attn1.processor") else pipeline.unet.config.cross_attention_dim
            )
            if name.startswith("mid_block"):
                hidden_size = pipeline.unet.config.block_out_channels[-1]
            elif name.startswith("up_blocks"):
                block_id = int(name[len("up_blocks.")])
                hidden_size = list(reversed(pipeline.unet.config.block_out_channels))[block_id]
            elif name.startswith("down_blocks"):
                block_id = int(name[len("down_blocks.")])
                hidden_size = pipeline.unet.config.block_out_channels[block_id]
        pipeline.unet.set_attn_processor(lora_attn_procs)
        trainable_layers = AttnProcsLayers(pipeline.unet.attn_processors)
    else:
        trainable_layers = pipeline.unet

    # Enable TF32 for faster training on Ampere GPUs,
    # cf https://pytorch.org/docs/stable/notes/cuda.html#tensorfloat-32-tf32-on-ampere-devices
    if config.allow_tf32:
        torch.backends.cuda.matmul.allow_tf32 = True
    if config.train.use_8bit_adam:
        try:
            import bitsandbytes as bnb
        except ImportError:
            raise ImportError(
                "Please install bitsandbytes to use 8-bit Adam. You can do so by running `pip install bitsandbytes`"
            )

        optimizer_cls = bnb.optim.AdamW8bit
    else:
        optimizer_cls = torch.optim.AdamW
    optimizer = optimizer_cls(
        trainable_layers.parameters(),
        lr=config.train.learning_rate,
        betas=(config.train.adam_beta1, config.train.adam_beta2),
        weight_decay=config.train.adam_weight_decay,
        eps=config.train.adam_epsilon,
    )
    # prepare prompt and reward fn
    prompt_fn = getattr(d3po_pytorch.prompts, config.prompt_fn)
    
    # generate negative prompt embeddings
    neg_prompt_embed = pipeline.text_encoder(
        pipeline.tokenizer(
            [""],
            return_tensors="pt",
            padding="max_length",
            truncation=True,
            max_length=pipeline.tokenizer.model_max_length,
        ).input_ids.to(accelerator.device)
    )[0]
    sample_neg_prompt_embeds = neg_prompt_embed.repeat(config.sample.batch_size, 1, 1)
    autocast = contextlib.nullcontext if config.use_lora else accelerator.autocast

    # Prepare everything with our `accelerator`.
    trainable_layers, optimizer = accelerator.prepare(trainable_layers, optimizer)

    if config.resume_from:
        logger.info(f"Resuming from {config.resume_from}")
        accelerator.load_state(config.resume_from)

    #################### SAMPLING ####################
    pipeline.unet.eval()
    samples = []
    total_prompts = []
    for i in tqdm(
        range(config.sample.num_batches_per_epoch),
        disable=not accelerator.is_local_main_process,
        position=0,
    ):
        # generate prompts
        prompts1, prompt_metadata = zip(
            *[prompt_fn(**config.prompt_fn_kwargs) for _ in range(config.sample.batch_size)]
        )
        # we set the prompts to be the same
        # prompts1 = ["1 hand"] * config.sample.batch_size 
        prompts7 = prompts6 = prompts5 = prompts4 = prompts3 = prompts2 = prompts1
        total_prompts.extend(prompts1)
        # encode prompts
        prompt_ids1 = pipeline.tokenizer(
            prompts1,
            return_tensors="pt",
            padding="max_length",
            truncation=True,
            max_length=pipeline.tokenizer.model_max_length,
        ).input_ids.to(accelerator.device)

        prompt_ids2 = pipeline.tokenizer(
            prompts2,
            return_tensors="pt",
            padding="max_length",
            truncation=True,
            max_length=pipeline.tokenizer.model_max_length,
        ).input_ids.to(accelerator.device)

        prompt_ids3 = pipeline.tokenizer(
            prompts3,
            return_tensors="pt",
            padding="max_length",
            truncation=True,
            max_length=pipeline.tokenizer.model_max_length,
        ).input_ids.to(accelerator.device)

        prompt_ids4 = pipeline.tokenizer(
            prompts4,
            return_tensors="pt",
            padding="max_length",
            truncation=True,
            max_length=pipeline.tokenizer.model_max_length,
        ).input_ids.to(accelerator.device)

        prompt_ids5 = pipeline.tokenizer(
            prompts5,
            return_tensors="pt",
            padding="max_length",
            truncation=True,
            max_length=pipeline.tokenizer.model_max_length,
        ).input_ids.to(accelerator.device)

        prompt_ids6 = pipeline.tokenizer(
            prompts6,
            return_tensors="pt",
            padding="max_length",
            truncation=True,
            max_length=pipeline.tokenizer.model_max_length,
        ).input_ids.to(accelerator.device)
        prompt_ids7 = pipeline.tokenizer(
            prompts7,
            return_tensors="pt",
            padding="max_length",
            truncation=True,
            max_length=pipeline.tokenizer.model_max_length,
        ).input_ids.to(accelerator.device)
        prompt_embeds1 = pipeline.text_encoder(prompt_ids1)[0]
        prompt_embeds2 = pipeline.text_encoder(prompt_ids2)[0]
        prompt_embeds3 = pipeline.text_encoder(prompt_ids3)[0]
        prompt_embeds4 = pipeline.text_encoder(prompt_ids4)[0]
        prompt_embeds5 = pipeline.text_encoder(prompt_ids5)[0]
        prompt_embeds6 = pipeline.text_encoder(prompt_ids6)[0]
        prompt_embeds7 = pipeline.text_encoder(prompt_ids7)[0]
        # sample
        with autocast():
            images1, _, latents1, _ = pipeline_with_logprob(
                pipeline,
                prompt_embeds=prompt_embeds1,
                negative_prompt_embeds=sample_neg_prompt_embeds,
                num_inference_steps=config.sample.num_steps,
                guidance_scale=config.sample.guidance_scale,
                eta=config.sample.eta,
                output_type="pt",
            )
            latents1 = torch.stack(latents1, dim=1)
            images1 = images1.cpu().detach()
            latents1 = latents1.cpu().detach()

            images2, _, latents2, _ = pipeline_with_logprob(
                pipeline,
                prompt_embeds=prompt_embeds2,
                negative_prompt_embeds=sample_neg_prompt_embeds,
                num_inference_steps=config.sample.num_steps,
                guidance_scale=config.sample.guidance_scale,
                eta=config.sample.eta,
                output_type="pt",
                latents = latents1[:,0,:,:,:]
            )
            latents2 = torch.stack(latents2, dim=1)
            images2 = images2.cpu().detach()
            latents2 = latents2.cpu().detach()

            images3, _, latents3, _ = pipeline_with_logprob(
                pipeline,
                prompt_embeds=prompt_embeds3,
                negative_prompt_embeds=sample_neg_prompt_embeds,
                num_inference_steps=config.sample.num_steps,
                guidance_scale=config.sample.guidance_scale,
                eta=config.sample.eta,
                output_type="pt",
                latents = latents1[:,0,:,:,:]
            )
            latents3 = torch.stack(latents3, dim=1)
            images3 = images3.cpu().detach()
            latents3 = latents3.cpu().detach()

            images4, _, latents4, _ = pipeline_with_logprob(
                pipeline,
                prompt_embeds=prompt_embeds4,
                negative_prompt_embeds=sample_neg_prompt_embeds,
                num_inference_steps=config.sample.num_steps,
                guidance_scale=config.sample.guidance_scale,
                eta=config.sample.eta,
                output_type="pt",
                latents = latents1[:,0,:,:,:]
            )
            latents4 = torch.stack(latents4, dim=1)
            images4 = images4.cpu().detach()
            latents4 = latents4.cpu().detach()

            images5, _, latents5, _ = pipeline_with_logprob(
                pipeline,
                prompt_embeds=prompt_embeds5,
                negative_prompt_embeds=sample_neg_prompt_embeds,
                num_inference_steps=config.sample.num_steps,
                guidance_scale=config.sample.guidance_scale,
                eta=config.sample.eta,
                output_type="pt",
                latents = latents1[:,0,:,:,:]
            )
            latents5 = torch.stack(latents5, dim=1)
            images5 = images5.cpu().detach()
            latents5 = latents5.cpu().detach()

            images6, _, latents6, _ = pipeline_with_logprob(
                pipeline,
                prompt_embeds=prompt_embeds6,
                negative_prompt_embeds=sample_neg_prompt_embeds,
                num_inference_steps=config.sample.num_steps,
                guidance_scale=config.sample.guidance_scale,
                eta=config.sample.eta,
                output_type="pt",
                latents = latents1[:,0,:,:,:]
            )
            latents6 = torch.stack(latents6, dim=1)
            images6 = images6.cpu().detach()
            latents6 = latents6.cpu().detach()
            images7, _, latents7, _ = pipeline_with_logprob(
                pipeline,
                prompt_embeds=prompt_embeds7,
                negative_prompt_embeds=sample_neg_prompt_embeds,
                num_inference_steps=config.sample.num_steps,
                guidance_scale=config.sample.guidance_scale,
                eta=config.sample.eta,
                output_type="pt",
                latents = latents1[:,0,:,:,:]
            )
            latents7 = torch.stack(latents7, dim=1)
            images7 = images7.cpu().detach()
            latents7 = latents7.cpu().detach()

        latents = torch.stack([latents1,latents2,latents3,latents4,latents5,latents6,latents7], dim=1)  # (batch_size, 2, num_steps + 1, 4, 64, 64)
        prompt_embeds = torch.stack([prompt_embeds1,prompt_embeds2,prompt_embeds3,prompt_embeds4,prompt_embeds5,prompt_embeds6,prompt_embeds7], dim=1)
        images = torch.stack([images1,images2,images3,images4,images5,images6,images7], dim=1)
        current_latents = latents[:, :, :-1]
        next_latents = latents[:, :, 1:]
        timesteps = pipeline.scheduler.timesteps.repeat(config.sample.batch_size, 1)  # (batch_size, num_steps)


        samples.append(
            {
                "prompt_embeds": prompt_embeds.cpu().detach(),
                "timesteps": timesteps.cpu().detach(),
                "latents": current_latents.cpu().detach(),  # each entry is the latent before timestep t
                "next_latents": next_latents.cpu().detach(),  # each entry is the latent after timestep t
                "images":images.cpu().detach(),
            }
        )
        os.makedirs(os.path.join(save_dir, "images/"), exist_ok=True)
        if (i+1)%config.sample.save_interval ==0 or i==(config.sample.num_batches_per_epoch-1):
            print(f'-----------{accelerator.process_index} save image start-----------')
            new_samples = {k: torch.cat([s[k] for s in samples]) for k in samples[0].keys()}
            images = new_samples['images'][local_idx:]
            for j, image in enumerate(images):
                for k in range(NUM_PER_PROMPT):
                    pil = Image.fromarray((image[k].cpu().numpy().transpose(1, 2, 0) * 255).astype(np.uint8))
                    pil.save(os.path.join(save_dir, f"images/{(NUM_PER_PROMPT*j+global_idx+k):05}.png"))
            global_idx += len(images)*NUM_PER_PROMPT
            local_idx += len(images)
            with open(os.path.join(save_dir, f'prompt{accelerator.process_index}.json'),'w') as f:
                json.dump(total_prompts, f)
            with open(os.path.join(save_dir, f'sample{accelerator.process_index}.pkl'), 'wb') as f:
                pickle.dump({"prompt_embeds": new_samples["prompt_embeds"], "timesteps": new_samples["timesteps"], "latents": new_samples["latents"], "next_latents": new_samples["next_latents"]}, f)
    with open(os.path.join(save_dir, f'{accelerator.process_index}.txt'), 'w') as f:
        f.write(f'{accelerator.process_index} done')
        print(f'GPU: {accelerator.device} done')
    if accelerator.is_main_process:
        while True:
            done = [True if os.path.exists(os.path.join(save_dir, f'{i}.txt')) else False for i in range(accelerator.num_processes)]
            if all(done):
                time.sleep(5)
                break
        print('---------start post processing---------')
        post_processing(save_dir, accelerator.num_processes)
if __name__ == "__main__":
    app.run(main)
