from collections import defaultdict
import contextlib
import os
import copy
import datetime
from concurrent import futures
import time
import sys
script_path = os.path.abspath(__file__)
sys.path.append(os.path.dirname(os.path.dirname(script_path)))
from absl import app, flags
from ml_collections import config_flags
from accelerate import Accelerator
from accelerate.utils import set_seed, ProjectConfiguration
from accelerate.logging import get_logger
from diffusers import StableDiffusionPipeline, DDIMScheduler, UNet2DConditionModel
from diffusers.loaders import AttnProcsLayers
from diffusers.models.attention_processor import LoRAAttnProcessor
import numpy as np
import d3po_pytorch.prompts
import d3po_pytorch.rewards
from d3po_pytorch.diffusers_patch.pipeline_with_logprob import pipeline_with_logprob
from d3po_pytorch.diffusers_patch.ddim_with_logprob import ddim_step_with_logprob
import torch
import wandb
from functools import partial
import tqdm
import tempfile
from PIL import Image
import copy
import numpy as np
tqdm = partial(tqdm.tqdm, dynamic_ncols=True)


FLAGS = flags.FLAGS
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
        gradient_accumulation_steps=config.train.gradient_accumulation_steps * num_train_timesteps,
    )
    if accelerator.is_main_process:
        accelerator.init_trackers(
            project_name="d3po-pytorch", config=config.to_dict(), init_kwargs={"wandb": {"name": config.run_name}}
        )
    logger.info(f"\n{config}")

    ramdom_seed = np.random.randint(0,100000)
    set_seed(ramdom_seed, device_specific=True)

    # load scheduler, tokenizer and models.
    pipeline = StableDiffusionPipeline.from_pretrained(config.pretrained.model, torch_dtype=torch.float16)
    if config.use_xformers:
        pipeline.enable_xformers_memory_efficient_attention()
    # freeze parameters of models to save more memory
    pipeline.vae.requires_grad_(False)
    pipeline.text_encoder.requires_grad_(False)
    pipeline.unet.requires_grad_(not config.use_lora)
    if not config.use_lora and config.train.activation_checkpointing:
        pipeline.unet.enable_gradient_checkpointing()
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
    pipeline.unet.to(accelerator.device, dtype=inference_dtype)
    ref =  copy.deepcopy(pipeline.unet)
    for param in ref.parameters():
        param.requires_grad = False
        
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
        pipeline.unet.set_attn_processor(lora_attn_procs)
        trainable_layers = AttnProcsLayers(pipeline.unet.attn_processors)
    else:
        trainable_layers = pipeline.unet

    # set up diffusers-friendly checkpoint saving with Accelerate

    def save_model_hook(models, weights, output_dir):
        assert len(models) == 1
        if config.use_lora and isinstance(models[0], AttnProcsLayers):
            pipeline.unet.save_attn_procs(output_dir)
        elif not config.use_lora and isinstance(models[0], UNet2DConditionModel):
            models[0].save_pretrained(os.path.join(output_dir, "unet"))
        else:
            raise ValueError(f"Unknown model type {type(models[0])}")
        weights.pop()  # ensures that accelerate doesn't try to handle saving of the model

    def load_model_hook(models, input_dir):
        assert len(models) == 1
        if config.use_lora and isinstance(models[0], AttnProcsLayers):
            tmp_unet = UNet2DConditionModel.from_pretrained(
                config.pretrained.model, revision=config.pretrained.revision, subfolder="unet"
            )
            tmp_unet.load_attn_procs(input_dir)
            models[0].load_state_dict(AttnProcsLayers(tmp_unet.attn_processors).state_dict())
            del tmp_unet
        elif not config.use_lora and isinstance(models[0], UNet2DConditionModel):
            load_model = UNet2DConditionModel.from_pretrained(input_dir, subfolder="unet")
            models[0].register_to_config(**load_model.config)
            models[0].load_state_dict(load_model.state_dict())
            del load_model
        else:
            raise ValueError(f"Unknown model type {type(models[0])}")
        models.pop()  # ensures that accelerate doesn't try to handle loading of the model

    # Support multi-dimensional comparison. Default demension is 1. You can add many rewards instead of only one to judge the preference of images.
    # For example: A: clipscore-30 blipscore-10 LAION aesthetic score-6.0 ; B: 20, 8, 5.0  then A is prefered than B
    # if C: 40, 4, 4.0 since C[0] = 40 > A[0] and C[1] < A[1], we do not think C is prefered than A or A is prefered than C 
    def compare(a, b):
        assert isinstance(a, torch.Tensor) and isinstance(b, torch.Tensor)
        if len(a.shape)==1:
            a = a[...,None]
            b = b[...,None]

        a_dominates = torch.logical_and(torch.all(a <= b, dim=1), torch.any(a < b, dim=1))
        b_dominates = torch.logical_and(torch.all(b <= a, dim=1), torch.any(b < a, dim=1))


        c = torch.zeros([a.shape[0],2],dtype=torch.float,device=a.device)

        c[a_dominates] = torch.tensor([-1., 1.],device=a.device)
        c[b_dominates] = torch.tensor([1., -1.],device=a.device)

        return c


    accelerator.register_save_state_pre_hook(save_model_hook)
    accelerator.register_load_state_pre_hook(load_model_hook)

    # Enable TF32 for faster training on Ampere GPUs,
    # cf https://pytorch.org/docs/stable/notes/cuda.html#tensorfloat-32-tf32-on-ampere-devices
    if config.allow_tf32:
        torch.backends.cuda.matmul.allow_tf32 = True

    # Initialize the optimizer
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
    reward_fn = getattr(d3po_pytorch.rewards, config.reward_fn)()

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
    train_neg_prompt_embeds = neg_prompt_embed.repeat(config.train.batch_size, 1, 1)
    # for some reason, autocast is necessary for non-lora training but for lora training it isn't necessary and it uses
    # more memory
    autocast = contextlib.nullcontext if config.use_lora else accelerator.autocast

    # Prepare everything with our `accelerator`.
    trainable_layers, optimizer = accelerator.prepare(trainable_layers, optimizer)

    # executor to perform callbacks asynchronously.
    executor = futures.ThreadPoolExecutor(max_workers=2)

    # Train!
    samples_per_epoch = config.sample.batch_size * accelerator.num_processes * config.sample.num_batches_per_epoch
    total_train_batch_size = (
        config.train.batch_size * accelerator.num_processes * config.train.gradient_accumulation_steps
    )

    logger.info("***** Running training *****")
    logger.info(f"  Num Epochs = {config.num_epochs}")
    logger.info(f"  Sample batch size per device = {config.sample.batch_size}")
    logger.info(f"  Train batch size per device = {config.train.batch_size}")
    logger.info(f"  Gradient Accumulation steps = {config.train.gradient_accumulation_steps}")
    logger.info("")
    logger.info(f"  Total number of samples per epoch = {samples_per_epoch}")
    logger.info(f"  Total train batch size (w. parallel, distributed & accumulation) = {total_train_batch_size}")
    logger.info(f"  Number of gradient updates per inner epoch = {samples_per_epoch // total_train_batch_size}")
    logger.info(f"  Number of inner epochs = {config.train.num_inner_epochs}")

    assert config.sample.batch_size >= config.train.batch_size
    assert config.sample.batch_size % config.train.batch_size == 0
    assert samples_per_epoch % total_train_batch_size == 0

    if config.resume_from:
        logger.info(f"Resuming from {config.resume_from}")
        accelerator.load_state(config.resume_from)
        first_epoch = int(config.resume_from.split("_")[-1]) + 1
    else:
        first_epoch = 0

    global_step = 0
    for epoch in range(first_epoch, config.num_epochs):
        #################### SAMPLING ####################
        pipeline.unet.eval()
        samples = []
        prompt_metadata = None
        for i in tqdm(
            range(config.sample.num_batches_per_epoch),
            desc=f"Epoch {epoch}: sampling",
            disable=not accelerator.is_local_main_process,
            position=0,
        ):
            # generate prompts
            prompts1, prompt_metadata = zip(
                *[prompt_fn(**config.prompt_fn_kwargs) for _ in range(config.sample.batch_size)]
            )
            prompts2 = prompts1
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
            prompt_embeds1 = pipeline.text_encoder(prompt_ids1)[0]
            prompt_embeds2 = pipeline.text_encoder(prompt_ids2)[0]

            # sample
            with autocast():
                images1, _, latents1, log_probs1 = pipeline_with_logprob(
                    pipeline,
                    prompt_embeds=prompt_embeds1,
                    negative_prompt_embeds=sample_neg_prompt_embeds,
                    num_inference_steps=config.sample.num_steps,
                    guidance_scale=config.sample.guidance_scale,
                    eta=config.sample.eta,
                    output_type="pt",
                )
                latents1 = torch.stack(latents1, dim=1)
                log_probs1 = torch.stack(log_probs1, dim=1)
                images2, _, latents2, log_probs2 = pipeline_with_logprob(
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
                log_probs2 = torch.stack(log_probs2, dim=1)

            latents = torch.stack([latents1,latents2], dim=1)  # (batch_size, 2, num_steps + 1, 4, 64, 64)
            log_probs = torch.stack([log_probs1,log_probs2], dim=1)  # (batch_size, num_steps, 1)
            prompt_embeds = torch.stack([prompt_embeds1,prompt_embeds2], dim=1)
            images = torch.stack([images1,images2], dim=1)
            current_latents = latents[:, :, :-1]
            next_latents = latents[:, :, 1:]
            timesteps = pipeline.scheduler.timesteps.repeat(config.sample.batch_size, 1)  # (batch_size, num_steps)

            # compute rewards asynchronously
            rewards1 = executor.submit(reward_fn, images1, prompts1, prompt_metadata).result()[0]
            # yield to to make sure reward computation starts
            rewards2 = executor.submit(reward_fn, images2, prompts2, prompt_metadata).result()[0]
            if isinstance(rewards1, np.ndarray):
                rewards = np.c_[rewards1, rewards2]
            else:
                rewards1 = rewards1.cpu().detach().numpy()
                rewards2 = rewards2.cpu().detach().numpy()
                rewards = np.c_[rewards1, rewards2]
            
            time.sleep(0)
            prompts1 = list(prompts1)
            samples.append(
                {
                    "prompt_embeds": prompt_embeds,
                    "prompts": prompts1,
                    "timesteps": timesteps,
                    "latents": current_latents,  # each entry is the latent before timestep t
                    "next_latents": next_latents,  # each entry is the latent after timestep t
                    "log_probs": log_probs,
                    "images":images,
                    "rewards":torch.as_tensor(rewards, device=accelerator.device),
                }
            )
        prompts = samples[0]["prompts"]
        del samples[0]["prompts"]
        samples = {k: torch.cat([s[k] for s in samples]) for k in samples[0].keys()}
        images = samples["images"]
        rewards = accelerator.gather(samples["rewards"]).cpu().numpy()
        #record the better images' reward based on the reward model.
        accelerator.log(
            {"epoch": epoch, "reward_mean": rewards.mean(), "reward_std": rewards.std()},
            step=global_step,
        )
        # this is a hack to force wandb to log the images as JPEGs instead of PNGs
        with tempfile.TemporaryDirectory() as tmpdir:
            for i, image in enumerate(images):
                image_np = image[0].to('cpu').to(torch.float32).numpy()
                pil = Image.fromarray((image_np.transpose(1, 2, 0) * 255).astype(np.uint8))
                pil = pil.resize((256, 256))
                pil.save(os.path.join(tmpdir, f"{i}.jpg"))
            accelerator.log(
                {
                    "images": [
                        wandb.Image(os.path.join(tmpdir, f"{i}.jpg"), caption=f"{prompt:.25} | {reward:.2f}")
                        for i, (prompt, reward) in enumerate(zip(prompts, rewards[:,0]))
                    ],
                },
                step=global_step,
            )
        # save prompts
        del samples["images"]
        torch.cuda.empty_cache()
        total_batch_size, num_timesteps = samples["timesteps"].shape
        assert total_batch_size == config.sample.batch_size * config.sample.num_batches_per_epoch
        assert num_timesteps == config.sample.num_steps
        orig_sample = copy.deepcopy(samples)
        #################### TRAINING ####################
        for inner_epoch in range(config.train.num_inner_epochs):
            # shuffle samples along batch dimension
            perm = torch.randperm(total_batch_size, device=accelerator.device)
            samples = {k: v[perm] for k, v in orig_sample.items()}

            # shuffle along time dimension independently for each sample
            perms = torch.stack(
                [torch.randperm(num_timesteps, device=accelerator.device) for _ in range(total_batch_size)]
            )
            for key in ["latents", "next_latents"]:
                tmp = samples[key].permute(0,2,3,4,5,1)[torch.arange(total_batch_size, device=accelerator.device)[:, None], perms]
                samples[key] = tmp.permute(0,5,1,2,3,4)
            samples["timesteps"] = samples["timesteps"][torch.arange(total_batch_size, device=accelerator.device)[:, None], perms].unsqueeze(1).repeat(1,2,1)
            tmp = samples["log_probs"].permute(0,2,1)[torch.arange(total_batch_size, device=accelerator.device)[:, None], perms]
            samples["log_probs"] = tmp.permute(0,2,1)
            # rebatch for training
            samples_batched = {k: v.reshape(-1, config.train.batch_size, *v.shape[1:]) for k, v in samples.items()}
            # dict of lists -> list of dicts for easier iteration
            samples_batched = [dict(zip(samples_batched, x)) for x in zip(*samples_batched.values())]
            # train
            pipeline.unet.train()
            info = defaultdict(list)
            for i in tqdm(range(0,total_batch_size,config.train.batch_size),
                        desc="Update",
                        position=2,
                        leave=False, 
                          ):
                sample_0 = {}
                sample_1 = {}
                for key, value in samples.items():
                    sample_0[key] = value[i:i+config.train.batch_size, 0]
                    sample_1[key] = value[i:i+config.train.batch_size, 1]
                if config.train.cfg:
                    # concat negative prompts to sample prompts to avoid two forward passes
                    embeds_0 = torch.cat([train_neg_prompt_embeds, sample_0["prompt_embeds"]])
                    embeds_1 = torch.cat([train_neg_prompt_embeds, sample_1["prompt_embeds"]])
                else:
                    embeds_0 = sample_0["prompt_embeds"]
                    embeds_1 = sample_1["prompt_embeds"]

                for j in tqdm(
                    range(num_train_timesteps),
                    desc="Timestep",
                    position=3,
                    leave=False,
                    disable=not accelerator.is_local_main_process,
                ):  
                    with accelerator.accumulate(pipeline.unet):
                        with autocast():
                            if config.train.cfg:
                                noise_pred_0 = pipeline.unet(
                                    torch.cat([sample_0["latents"][:, j]] * 2),
                                    torch.cat([sample_0["timesteps"][:, j]] * 2),
                                    embeds_0,
                                ).sample
                                noise_pred_uncond_0, noise_pred_text_0 = noise_pred_0.chunk(2)
                                noise_pred_0 = noise_pred_uncond_0 + config.sample.guidance_scale * (noise_pred_text_0 - noise_pred_uncond_0)

                                noise_ref_pred_0 = ref(
                                    torch.cat([sample_0["latents"][:, j]] * 2),
                                    torch.cat([sample_0["timesteps"][:, j]] * 2),
                                    embeds_0,
                                ).sample
                                noise_ref_pred_uncond_0, noise_ref_pred_text_0 = noise_ref_pred_0.chunk(2)
                                noise_ref_pred_0 = noise_ref_pred_uncond_0 + config.sample.guidance_scale * (
                                    noise_ref_pred_text_0 - noise_ref_pred_uncond_0
                                )

                                noise_pred_1 = pipeline.unet(
                                    torch.cat([sample_1["latents"][:, j]] * 2),
                                    torch.cat([sample_1["timesteps"][:, j]] * 2),
                                    embeds_1,
                                ).sample
                                noise_pred_uncond_1, noise_pred_text_1 = noise_pred_1.chunk(2)
                                noise_pred_1 = noise_pred_uncond_1 + config.sample.guidance_scale * (noise_pred_text_1 - noise_pred_uncond_1)

                                noise_ref_pred_1 = ref(
                                    torch.cat([sample_1["latents"][:, j]] * 2),
                                    torch.cat([sample_1["timesteps"][:, j]] * 2),
                                    embeds_1,
                                ).sample
                                noise_ref_pred_uncond_1, noise_ref_pred_text_1 = noise_ref_pred_1.chunk(2)
                                noise_ref_pred_1 = noise_ref_pred_uncond_1 + config.sample.guidance_scale * (
                                    noise_ref_pred_text_1 - noise_ref_pred_uncond_1
                                )

                            else:
                                noise_pred_0 = pipeline.unet(
                                    sample_0["latents"][:, j], sample_0["timesteps"][:, j], embeds_0
                                ).sample
                                noise_ref_pred_0 = ref(
                                    sample_0["latents"][:, j], sample_0["timesteps"][:, j], embeds_0
                                ).sample

                                noise_pred_1 = pipeline.unet(
                                    sample_1["latents"][:, j], sample_1["timesteps"][:, j], embeds_1
                                ).sample
                                noise_ref_pred_1 = ref(
                                    sample_1["latents"][:, j], sample_1["timesteps"][:, j], embeds_1
                                ).sample



                            # compute the log prob of next_latents given latents under the current model
                            _, total_prob_0 = ddim_step_with_logprob(
                                pipeline.scheduler,
                                noise_pred_0,
                                sample_0["timesteps"][:, j],
                                sample_0["latents"][:, j],
                                eta=config.sample.eta,
                                prev_sample=sample_0["next_latents"][:, j],
                            )
                            _, total_ref_prob_0 = ddim_step_with_logprob(
                                pipeline.scheduler,
                                noise_ref_pred_0,
                                sample_0["timesteps"][:, j],
                                sample_0["latents"][:, j],
                                eta=config.sample.eta,
                                prev_sample=sample_0["next_latents"][:, j],
                            )
                            _, total_prob_1 = ddim_step_with_logprob(
                                pipeline.scheduler,
                                noise_pred_1,
                                sample_1["timesteps"][:, j],
                                sample_1["latents"][:, j],
                                eta=config.sample.eta,
                                prev_sample=sample_1["next_latents"][:, j],
                            )
                            _, total_ref_prob_1 = ddim_step_with_logprob(
                                pipeline.scheduler,
                                noise_ref_pred_1,
                                sample_1["timesteps"][:, j],
                                sample_1["latents"][:, j],
                                eta=config.sample.eta,
                                prev_sample=sample_1["next_latents"][:, j],
                            )
                    human_prefer = compare(sample_0['rewards'],sample_1['rewards'])
                    # clip the Q value
                    ratio_0 = torch.clamp(torch.exp(total_prob_0-total_ref_prob_0),1 - config.train.eps, 1 + config.train.eps)
                    ratio_1 = torch.clamp(torch.exp(total_prob_1-total_ref_prob_1),1 - config.train.eps, 1 + config.train.eps)
                    loss = -torch.log(torch.sigmoid(config.train.beta*(torch.log(ratio_0))*human_prefer[:,0] + config.train.beta*(torch.log(ratio_1))*human_prefer[:, 1])).mean()

                    # backward pass
                    accelerator.backward(loss)
                    if accelerator.sync_gradients:
                        accelerator.clip_grad_norm_(trainable_layers.parameters(), config.train.max_grad_norm)
                    optimizer.step()
                    optimizer.zero_grad()

                # Checks if the accelerator has performed an optimization step behind the scenes
            if accelerator.sync_gradients:
                assert (j == num_train_timesteps - 1) and (
                    i + 1
                ) % config.train.gradient_accumulation_steps == 0
                # log training-related stuff
                info = {k: torch.mean(torch.stack(v)) for k, v in info.items()}
                info = accelerator.reduce(info, reduction="mean")
                info.update({"epoch": epoch, "inner_epoch": inner_epoch})
                accelerator.log(info, step=global_step)
                global_step += 1
                info = defaultdict(list)

                # make sure we did an optimization step at the end of the inner epoch
                assert accelerator.sync_gradients

            if epoch!=0 and (epoch+1) % config.save_freq == 0:
                accelerator.save_state()


if __name__ == "__main__":
    app.run(main)
