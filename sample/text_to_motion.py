import json
import os
import pickle
import shutil
import time

import matplotlib.pyplot as plt
import numpy as np
import torch
from tqdm import tqdm

import data_loaders.humanml.utils.paramUtil as paramUtil
from data_loaders.get_data import DatasetConfig, get_dataset_loader
from data_loaders.humanml.utils.plot_script import plot_3d_motion
from data_loaders.tensors import collate
from diffusion.gaussian_diffusion import GaussianDiffusion
from dno import DNO, DNOOptions
from model.cfg_sampler import ClassifierFreeSampleModel
from sample import dno_helper
from sample.condition import CondKeyLocationsLoss
from utils import dist_util
from utils.fixseed import fixseed
from utils.model_util import (create_gaussian_diffusion,
                              create_model_and_diffusion, load_model_wo_clip)
from utils.output_util import (construct_template_variables, sample_to_motion,
                               save_multiple_samples)
from utils.parser_util import generate_args


def main(num_trials=3):
    # ===== SET CONSTANTS =====    
    fps = 20
    max_frames = 196
    motion_length_cut = 6.0

    # ===== PARSE ARGS AND SETUP =====
    args = generate_args()
    args.device = 0
    args.use_ddim = True
    fixseed(args.seed)
    dist_util.setup_dist(args.device)

    args.num_samples = 1
    args.num_repetitions = 1            # Usually 1 when we just want to generate a single sample
    args.batch_size = args.num_samples  # Usually 1 when we just want to generate a single sample
    args.gen_batch_size = 1

    # ===== SANITY CHECK PRINTS =====
    print(f"Text Prompt #1: {args.text_prompt}")

    if not args.text_prompt:
        raise ValueError("Please provide a text prompt for the generation.")
    
    # ===== SET OUT_PATH =====
    out_path = args.output_dir
    if out_path == "":
        out_path = os.path.join(os.path.dirname(args.model_path), 
                                "unnamed" if args.text_prompt == "" else args.text_prompt.replace(" ", "_").replace(".", ""))
    args.output_dir = out_path

    if os.path.exists(out_path):
        shutil.rmtree(out_path)
    os.makedirs(out_path)
    with open(os.path.join(out_path, "args.json"), "w") as fw:
        json.dump(vars(args), fw, indent=4, sort_keys=True)

    # ===== SET FPS AND MAX LENGTH =====
    args.fps = fps
    args.max_frames = max_frames

    n_frames = min(args.max_frames, int(args.motion_length * fps))  # Clip to max_frames
    gen_frames = int(motion_length_cut * fps)                       # Calculate how many frames to generate for length and fps
    assert gen_frames <= n_frames, "gen_frames must be less than n_frames"
    args.gen_frames = gen_frames
    print("n_frames", n_frames)
    print("gen_frames", gen_frames)

    # ===== LOAD DATASET =====
    texts = [args.text_prompt]

    print("Loading dataset...")
    data = load_dataset(args, n_frames)

    # =======================
    # ===== SETUP MODEL =====
    # =======================
    print("Creating model and diffusion...")
    model, _ = create_model_and_diffusion(args, data)
    diffusion = create_gaussian_diffusion(args, timestep_respacing="ddim100")

    print(f"Loading checkpoints from [{args.model_path}]...")
    state_dict = torch.load(args.model_path, map_location="cpu")
    load_model_wo_clip(model, state_dict)

    if args.guidance_param != 1:
        model = ClassifierFreeSampleModel(model)  

    model.to(dist_util.dev())
    model.eval() 

    # ===== SETUP MODEL_KWARGS =====
    collate_args = [
        {
            "inp": torch.zeros(n_frames),
            "tokens": None,
            "lengths": gen_frames,
        }
    ] * args.num_samples
    collate_args = [dict(arg, text=txt) for arg, txt in zip(collate_args, texts)]
    _, model_kwargs = collate(collate_args)

    # add CFG scale to batch
    if args.guidance_param != 1:
        model_kwargs["y"]["scale"] = (
            torch.ones(args.batch_size, device=dist_util.dev())
            * args.guidance_param
        )

    model_kwargs["y"]["log_name"] = out_path
    model_kwargs["y"]["traj_model"] = False

    # ===== GENERATE MOTION =====
    print("Generating motion from text...")
    sample = dno_helper.run_text_to_motion(args, diffusion, model, model_kwargs, data, n_frames)  # [1, 263, 1, 120]

    print("Converting motion to XYZ skeleton locations...")
    gen_sample = sample[:, :, :, :gen_frames]
    # Convert motion representation to skeleton motion
    gen_motions, _, _ = sample_to_motion(
        gen_sample,
        args,
        model_kwargs,
        model,
        gen_frames,
        data.dataset.t2m_dataset.inv_transform,
    )
    motion = gen_motions[0][0].transpose(2, 0, 1)  # [120, 22, 3]

    # ===== SAVE MOTION =====
    print("Saving motion...")

    torch.save(sample, os.path.join(out_path, "optimized_x.pt"))  # Call optimized_x.pt to make compatible for loading in gen_dno.py
    np.save(os.path.join(out_path, "motion.npy"), motion)

def load_dataset(args, n_frames):
    conf = DatasetConfig(
        name=args.dataset,
        batch_size=args.batch_size,
        num_frames=args.max_frames,
        split="test",
        hml_mode="text_only",  # 'train'
        traject_only=False,
    )
    data = get_dataset_loader(conf)
    data.fixed_length = n_frames
    return data

if __name__ == "__main__":
    main()
