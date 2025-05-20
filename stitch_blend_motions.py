import os

import torch
import numpy as np


from utils.parser_util import generate_args
from utils.output_util import sample_to_motion
from utils.model_util import create_model_and_diffusion
from utils import dist_util
from data_loaders.humanml.utils.plot_script import plot_3d_motion
from data_loaders.humanml.utils import paramUtil 
from data_loaders.get_data import DatasetConfig, get_dataset_loader
from data_loaders.tensors import collate

def plot_motion(motion, title, save_path="./save/stitched/motion.mp4"):
    """
    Visualize the motion in 3D and save it as a video.
    
    Parameters:
    - motion: The motion data to visualize (numpy array) of shape (joints, coords, frames).
    - title: Title for the visualization.
    - save_path: Path to save the video.
    """
    # Transpose motion data: from (joints, coords, frames) to (frames, joints, coords)
    motion = motion[:, :3, :].transpose(2, 0, 1)
    kinematic_chain = paramUtil.t2m_kinematic_chain
    plot_3d_motion(save_path, kinematic_chain, motion, title=title, dataset='humanml', fps=20)

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
    # Load motions
    walk = torch.load("./save/walk/optimized_x.pt")
    jump = torch.load("./save/jumping/optimized_x.pt")

    blend = torch.load("./save/blend/optimized_x.pt")
    blend_no_offset = torch.load("./save/blend_no_offset/optimized_x.pt")
    blend_bigger_seam = torch.load("./save/blend_bigger_seam/optimized_x.pt")
    blend_weighted = torch.load("./save/blend_weighted/optimized_x.pt")

    args = generate_args()
    args.max_frames = walk.shape[-1]

    print(walk.shape)

    print(f"Taking from walk starting from 0 to {args.max_frames // 2}")

    # Take blend[2] here as blend[0] gets stuck in a local optimum with the fixed seed
    naiive_concat = torch.cat([walk, jump], dim=-1)
    blend = torch.cat([walk[0, :, :, :args.max_frames // 2], blend[2], jump[0, :, :, args.max_frames // 2:]], dim=-1).unsqueeze(0)
    blend_no_offset = torch.cat([walk[0, :, :, :args.max_frames // 2], blend_no_offset[2], jump[0, :, :, args.max_frames // 2:]], dim=-1).unsqueeze(0)
    blend_bigger_seam = torch.cat([walk[0, :, :, :args.max_frames // 2], blend_bigger_seam[2], jump[0, :, :, args.max_frames // 2:]], dim=-1).unsqueeze(0)
    blend_weighted = torch.cat([walk[0, :, :, :args.max_frames // 2], blend_weighted[2], jump[0, :, :, args.max_frames // 2:]], dim=-1).unsqueeze(0)

    # ===== SETUP =====
    if True:
        collate_args = [{
                "inp": torch.zeros(args.max_frames),
                "tokens": None,
                "lengths": args.max_frames,
            }] * args.num_samples
        collate_args = [dict(arg, text=txt) for arg, txt in zip(collate_args, [""] * args.num_samples)]
        _, model_kwargs = collate(collate_args)

        if args.guidance_param != 1:
            model_kwargs["y"]["scale"] = (torch.ones(args.batch_size, device=dist_util.dev()) * args.guidance_param)

        model_kwargs["y"]["log_name"] = "./save/stitched"
        model_kwargs["y"]["traj_model"] = False
    if True:
        data = load_dataset(args, args.max_frames)
    if True:
        model, _ = create_model_and_diffusion(args, data)

    # ===== VISUALIZE =====
    get_motion = lambda x: sample_to_motion(x, args, model_kwargs, model, args.max_frames, data.dataset.t2m_dataset.inv_transform)[0][0].squeeze()

    naiive_concat_motion = get_motion(naiive_concat)
    blend_motion = get_motion(blend)
    blend_no_offset_motion = get_motion(blend_no_offset)
    blend_bigger_seam_motion = get_motion(blend_bigger_seam)
    blend_weighted_motion = get_motion(blend_weighted)

    plot_motion(naiive_concat_motion, "Naiive Concatenation", save_path="./save/stitched/naiive_concat.mp4")
    plot_motion(blend_motion, "DNO Blend", save_path="./save/stitched/dno_blend.mp4")
    plot_motion(blend_no_offset_motion, "DNO Blend (No Offset)", save_path="./save/stitched/dno_blend_no_offset.mp4")
    plot_motion(blend_bigger_seam_motion, "DNO Blend (Bigger Seam)", save_path="./save/stitched/dno_blend_bigger_seam.mp4")
    plot_motion(blend_weighted_motion, "DNO Blend (Weighted)", save_path="./save/stitched/dno_blend_weighted.mp4")

    # Combine all videos side by side using ffmpeg
    os.system(
        "ffmpeg -y "
        "-i ./save/stitched/naiive_concat.mp4 "
        "-i ./save/stitched/dno_blend.mp4 "
        "-i ./save/stitched/dno_blend_no_offset.mp4 "
        "-i ./save/stitched/dno_blend_bigger_seam.mp4 "
        "-i ./save/stitched/dno_blend_weighted.mp4 "
        "-filter_complex \"[0:v:0][1:v:0][2:v:0][3:v:0][4:v:0]hstack=inputs=5\" "
        "-c:v libx264 ./save/stitched/comparison.mp4"
    )

    # Save motions as files:
    np.save("./save/stitched/naiive_concat.npy", {
        "motion": naiive_concat_motion[None, :, :, :],
        "num_samples": 1,
        "lengths": [naiive_concat_motion.shape[-1]],
    })

    np.save("./save/stitched/dno_blend.npy", {
        "motion": blend_motion[None, :, :, :],
        "num_samples": 1,
        "lengths": [blend_motion.shape[-1]],
    })

    np.save("./save/stitched/dno_blend_no_offset.npy", {
        "motion": blend_no_offset_motion[None, :, :, :],
        "num_samples": 1,
        "lengths": [blend_no_offset_motion.shape[-1]],
    })

    np.save("./save/stitched/dno_blend_bigger_seam.npy", {
        "motion": blend_bigger_seam_motion[None, :, :, :],
        "num_samples": 1,
        "lengths": [blend_bigger_seam_motion.shape[-1]],
    })

    np.save("./save/stitched/dno_blend_weighted.npy", {
        "motion": blend_weighted_motion[None, :, :, :],
        "num_samples": 1,
        "lengths": [blend_weighted_motion.shape[-1]],
    })