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
from data_loaders.humanml.data.dataset import HumanML3D
from data_loaders.humanml.utils.plot_script import plot_3d_motion
from data_loaders.humanml.utils.metrics import (calculate_skating_ratio,
                                                compute_jitter)
from data_loaders.tensors import collate
from diffusion.gaussian_diffusion import GaussianDiffusion
from dno import DNO, DNOOptions
from eval.calculate_fid import calculate_fid_given_two_populations
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


def main(num_trials=8):
    # ===== SET CONSTANTS =====
    NUM_ODE_STEPS = 10
    OPTIMIZATION_STEPS = 800
    GRADIENT_CHECKPOINT = False
    
    fps = 20
    max_frames = 196
    motion_length_cut = 6.0

    # task = "trajectory_editing"
    # task = "pose_editing"
    # task = "dense_optimization"
    # task = "motion_projection"
    task = "motion_blending"
    # task = "motion_inbetweening"

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

    skeleton = paramUtil.t2m_kinematic_chain

    # ===== SANITY CHECK PRINTS =====
    print(f"Text Prompt #1: {args.text_prompt}")
    print(f"Text Prompt #2: {args.text_prompt_2}")

    print(f"Load From #1: {args.load_from}")
    print(f"Load From #2: {args.load_from_2}")

    if not (args.load_from or args.text_prompt):
        raise ValueError("Either 'load_from' or 'text_prompt' must be set for the initial motion.")
    if task == "motion_blending" and not (args.load_from_2 or args.text_prompt_2):
        raise ValueError("Either 'load_from_2' or 'text_prompt_2' must be set for the second motion in motion blending task.")

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
    # ! This block must be called BEFORE the dataset is loaded
    if args.text_prompt == "":
        # When specifying an input motion, we do not need to set a proper text prompt
        args.text_prompt = "dummy prompt"
    texts = [args.text_prompt]

    print("Loading dataset...")
    # Loading dataset for transforms and configurations
    data = load_dataset(args, n_frames)
    total_num_samples = args.num_samples * args.num_repetitions # Usually 1 when we just want to generate a single sample

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
    model_device = next(model.parameters()).device

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

    # ===== SETUP LOGGING AND TARGET VARIABLES =====
    all_motions = []
    all_lengths = []
    all_text = []
    obs_list = []
    kframes = []

    target = torch.zeros([args.gen_batch_size, args.max_frames, 22, 3], device=model_device)
    target_mask = torch.zeros_like(target, dtype=torch.bool)
    SHOW_TARGET_POSE = task == "motion_inbetweening"

    for repetition_idx in range(args.num_repetitions):
        # ===== GENERATE OR LOAD MOTION(S) =====
        if args.load_from == '':
            print("Generating motion from text...")
            sample = dno_helper.run_text_to_motion(args, diffusion, model, model_kwargs, data, n_frames)  # [1, 263, 1, 120]
        else:
            print("Loading motion from dir...")
            sample = torch.load(os.path.join(args.load_from , "optimized_x.pt"))[None, 0].clone() # [1, 263, 1, 120]

        if task == "motion_blending":
            if args.load_from_2 == '':
                print("Generating 2nd motion from text...")
                model_kwargs["y"]["text"] = [args.text_prompt_2]
                sample_2 = dno_helper.run_text_to_motion(args, diffusion, model, model_kwargs, data, n_frames)
            else:
                print("Loading 2nd motion from dir...")
                sample_2 = torch.load(os.path.join(args.load_from_2, "optimized_x.pt"))[None, 0].clone()

        # ===== PREPARE MOTION =====
        # Cut motion to the desired length if it was longer
        gen_sample = sample[:, :, :, :gen_frames]

        # Convert motion representation to skeleton motion (XYZ coordinates of joints)
        gen_motions, cur_lengths, initial_text = sample_to_motion(
            gen_sample,
            args,
            model_kwargs,
            model,
            gen_frames,
            data.dataset.t2m_dataset.inv_transform,
        )
        initial_motion = gen_motions[0][0].transpose(2, 0, 1)  # [120, 22, 3]

        # ===== SETUP TASK-SPECIFIC VARIABLES =====
        task_info = {
            "task": task,
            "skeleton": skeleton,
            "initial_motion": initial_motion,
            "initial_text": initial_text,
            "device": model_device,
        }

        if task == "motion_blending":
            print(f"sample.shape: {sample.shape}")
            print(f"sample_2.shape: {sample_2.shape}")
            print(f"================= NAIIVE CONCATENATION =================")
            print(f"Taking from sample starting at {gen_frames // 2}")
            print(f"Taking from sample_2 starting at {args.num_offset} to {gen_frames // 2 + args.num_offset}")
            print(f"================= BLEND MOTION =================")
            print(f"Taking from sample starting at {gen_frames // 2}")
            print(f"Taking from sample_2 starting at {args.num_offset} to {gen_frames - (gen_frames // 2) + args.num_offset}")

            # # Simply concat both motions. This is only used for visualization
            gen_sample = torch.cat([
                    sample[:, :, :, gen_frames // 2 :],
                    sample_2[:, :, :, args.num_offset : gen_frames // 2 + args.num_offset],
                ], dim=-1)  
            # Create the motion which will be used for the optimization
            gen_sample_full = torch.cat([
                    sample[:, :, :, gen_frames // 2 :],
                    sample_2[:, :, :, args.num_offset : gen_frames // 2 + args.num_offset]
                ], dim=-1)

            combined_motions, cur_lengths, cur_texts = sample_to_motion(
                gen_sample_full,
                args,
                model_kwargs,
                model,
                gen_frames,
                data.dataset.t2m_dataset.inv_transform,
            )
            combined_kps = (torch.from_numpy(combined_motions[0][0]).to(model_device).permute(2, 0, 1))  # [120, 22, 3]
            task_info["combine_motion"] = combined_kps

        target, target_mask, kframes, is_noise_init, initial_motion, obs_list = (dno_helper.prepare_task(task_info, args))

        # ===== SETUP OPTIMIZATION CONFIG
        is_editing_task = not is_noise_init
        noise_opt_conf = DNOOptions(
            num_opt_steps=OPTIMIZATION_STEPS, # 300 if is_editing_task else 500,
            diff_penalty_scale=2e-3 if is_editing_task else 0,
            optimizer=args.optimizer
        )
        START_FROM_NOISE = is_noise_init

        # Repeat target to match num_trials
        if target.shape[0] == 1:  
            target = target.repeat(num_trials, 1, 1, 1)
            target_mask = target_mask.repeat(num_trials, 1, 1, 1)
        elif target.shape[0] != num_trials:
            raise ValueError("target shape is not 1 or equal to num_trials")

        # At this point, we need to have (1) target, (2) target_mask, (3) kframes, (4, optional) initial motion

        # Remove text for the optimization
        model_kwargs["y"]["text"] = [""]

        # ===== DDIM INVERSION =====
        # Do inversion to get the initial noise for editing
        inverse_step = 100  # 1000 for more previse inversion
        diffusion_invert = create_gaussian_diffusion(args, timestep_respacing=f"ddim{inverse_step}")
        motion_to_invert = sample.clone()
        inv_noise, pred_x0_list = ddim_invert(
            diffusion_invert,
            model,
            motion_to_invert,
            model_kwargs=model_kwargs,
            dump_steps=[0, 5, 10, 20, 30, 40, 49],
            num_inference_steps=inverse_step,
            clip_denoised=False,
        )

        # ===== OPTIMIZATION =====
        opt_step = noise_opt_conf.num_opt_steps
        inter_out = []
        diffusion = create_gaussian_diffusion(args, f"ddim{NUM_ODE_STEPS}")

        # Setup the list of steps at which to save the intermediate results
        step_out_list = [0, 0.1, 0.2, 0.3, 0.5, 0.7, 0.95]
        step_out_list = [int(aa * opt_step) for aa in step_out_list]
        step_out_list[-1] = opt_step - 1

        if START_FROM_NOISE:
            torch.manual_seed(0)
            # use the batch size that comes from main()
            gen_shape = [num_trials, model.njoints, model.nfeats, n_frames]
            cur_xt = torch.randn(gen_shape).to(model_device)
        else:
            cur_xt = inv_noise.detach().clone()
            cur_xt = cur_xt.repeat(num_trials, 1, 1, 1)

        cur_xt = cur_xt.detach().requires_grad_()

        print(f"Endpoint weight: {args.endpoint_weight}")

        loss_fn = CondKeyLocationsLoss(
            target=target,
            target_mask=target_mask,
            transform=data.dataset.t2m_dataset.transform_th,
            inv_transform=data.dataset.t2m_dataset.inv_transform_th,
            abs_3d=False,
            use_mse_loss=False,
            use_rand_projection=False,
            obs_list=obs_list,
            motion_length=[gen_frames] * target.shape[0],
            w_endpoint=args.endpoint_weight,
        )
        criterion = lambda x: loss_fn(x, **model_kwargs)

        def solver(z):
            return ddim_loop_with_gradient(
                diffusion,
                model,
                (num_trials, model.njoints, model.nfeats, n_frames),
                model_kwargs=model_kwargs,
                noise=z,
                clip_denoised=False,
                gradient_checkpoint=GRADIENT_CHECKPOINT,
            )

        # ! This is the main DNO optimization loop !
        noise_opt = DNO(model=solver, criterion=criterion, start_z=cur_xt, conf=noise_opt_conf)
        out = noise_opt()

        # Save the intermediate results
        for t in step_out_list:
            inter_step = []
            for i in range(num_trials):
                inter_step.append(out["hist"][i]["x"][t])
            inter_step = torch.stack(inter_step, dim=0)
            inter_out.append(inter_step)

        # Output plots of the metrics
        for i in range(num_trials):
            hist = out["hist"][i]
            for key in [
                "loss",
                "loss_diff",
                "loss_decorrelate",
                "grad_norm",
                "lr",
                "perturb_scale",
                "diff_norm",
            ]:
                plt.figure()
                if key in ["loss", "loss_diff", "loss_decorrelate"]:
                    plt.semilogy(hist["step"], hist[key])
                    # plt.ylim(top=0.4)
                    # Plot horizontal red line at lowest point of loss function
                    min_loss = min(hist[key])
                    plt.axhline(y=min_loss, color="r")
                    plt.text(0, min_loss, f"Min Loss: {min_loss:.4f}", color="r")
                else:
                    plt.plot(hist["step"], hist[key])
                plt.legend([key])
                plt.savefig(os.path.join(out_path, f"trial_{i}_{key}.png"))
                plt.close()

        final_out = out["x"].detach().clone()

        # Concatenate the intermediate results for visualization
        if task == "motion_blending":
            motion_to_vis = torch.cat([sample, sample_2, gen_sample, final_out], dim=0)
            captions = [
                "Original 1",
                "Original 2",
                "Naive concatenation",
            ] + [f"Prediction {i+1}" for i in range(num_trials)]
            args.num_samples = 3 + num_trials
        else:
            motion_to_vis = torch.cat([sample, final_out], dim=0)
            captions = [
                "Original",
            ] + [f"Prediction {i+1}" for i in range(num_trials)]
            args.num_samples = 1 + num_trials

        torch.save(out["z"], os.path.join(out_path, "optimized_z.pt"))
        torch.save(out["x"], os.path.join(out_path, "optimized_x.pt"))

        optimized_x = out["x"].detach().clone()

        # Setup motion before edit as naiive concatenation
        sample_before_edit = torch.cat([sample, sample_2], dim=-1)
        motion_before_edit, _, _ = sample_to_motion(
            sample_before_edit,
            args,
            model_kwargs,
            model,
            gen_frames,
            data.dataset.t2m_dataset.inv_transform,
        )
        motion_before_edit = motion_before_edit[0][0].transpose(2, 0, 1)  # [240, 22, 3]
        motion_before_edit = motion_before_edit[None, :, :, :]
        motion_before_edit = torch.from_numpy(motion_before_edit).to(model_device)

        # Setup generated motions
        generated_samples = torch.cat([
            sample[:, :, :, :gen_frames // 2 ].expand(num_trials, -1, -1, -1),
            optimized_x,
            sample_2[:, :, :, gen_frames // 2 + args.num_offset:].expand(num_trials, -1, -1, -1)
        ], dim=-1)
        generated_motions, _, _ = sample_to_motion(
            generated_samples,
            args,
            model_kwargs,
            model,
            gen_frames,
            data.dataset.t2m_dataset.inv_transform,
        )
        generated_motions = generated_motions[0].transpose(0, 3, 1, 2)  # [num_trials, 240, 22, 3]
        generated_motions = torch.from_numpy(generated_motions).to(model_device)

        # Setup target
        target = motion_before_edit.clone().repeat(num_trials, 1, 1, 1)
        target_masks = torch.cat([
            torch.zeros((num_trials, gen_frames // 2, 22, 3), device=model_device).bool(),
            target_mask,
            torch.zeros((num_trials, gen_frames // 2 + args.num_offset - args.num_offset, 22, 3), device=model_device).bool()
        ], dim=1)
        # Visualize the first target video
        target_np = target.detach().cpu().numpy()  # [num_trials, length, 22, 3]
        if target_np.shape[0] > 0:
            motion = target_np[0]
            print(f"Debug Video Motion Shape: {motion.shape}")
            save_path = os.path.join(out_path, f"target_0.mp4")
            plot_3d_motion(
            save_path,
            skeleton,
            motion,
            dataset=args.dataset,
            title="Target 0",
            fps=fps,
            kframes=kframes,
            obs_list=obs_list,
            target_pose=motion,
            gt_frames=[kk for (kk, _) in kframes] if SHOW_TARGET_POSE else [],
        )

        # Visualize the first generated motion video
        generated_motions_np = generated_motions.detach().cpu().numpy()  # [num_trials, length, 22, 3]
        for idx in range(generated_motions_np.shape[0]):
            motion = generated_motions_np[idx]
            print(f"Debug Video Motion Shape: {motion.shape}")
            save_path = os.path.join(out_path, f"generated_{idx}.mp4")
            plot_3d_motion(
            save_path,
            skeleton,
            motion,
            dataset=args.dataset,
            title=f"Generated {idx}",
            fps=fps,
            kframes=kframes,
            obs_list=obs_list,
            target_pose=target_np[idx] if idx < target_np.shape[0] else None,
            gt_frames=[kk for (kk, _) in kframes] if SHOW_TARGET_POSE else [],
            )

        metrics, metrics_before_edit, fid_dict = calculate_results(
            motion_before_edit, generated_motions,
            target, target_masks,
            max_frames=target.shape[1], 
            num_keyframe=1, 
            text=args.text_prompt, 
            dataset=data.dataset.t2m_dataset,
        )

        log_file = os.path.join(out_path, "metrics.txt")
        with open(log_file, 'w') as f:
            for (name, eval_results) in zip(["Before Edit", "After Edit"], [metrics_before_edit, metrics]):
                print(f"==================== {name} ====================")
                print(f"==================== {name} ====================", file=f, flush=True)
                for metric_name, metric_values in eval_results.items():
                    metric_values = np.array(metric_values)
                    unit_name = ""
                    if metric_name == "Jitter":
                        unit_name = "(m/s^3)"
                    elif metric_name == "Foot skating":
                        unit_name = "(ratio)"
                    elif metric_name == "Content preservation":
                        unit_name = "(ratio)"
                    elif metric_name == "Objective Error":
                        unit_name = "(m)"
                    print(f"Metric [{metric_name} {unit_name}]: Mean {metric_values.mean():.4f}, Std {metric_values.std():.4f}")
                    print(f"Metric [{metric_name} {unit_name}]: Mean {metric_values.mean():.4f}, Std {metric_values.std():.4f}", file=f, flush=True)

        ###################
        num_dump_step = 1
        args.num_dump_step = num_dump_step

        # Convert sample to XYZ skeleton locations
        # Each return size [bs, 1, 3, 120]
        cur_motions, cur_lengths, cur_texts = sample_to_motion(
            motion_to_vis,  # sample,
            args,
            model_kwargs,
            model,
            gen_frames,
            data.dataset.t2m_dataset.inv_transform,
        )

        if task == "motion_projection":
            # Visualize noisy motion in the second row last column
            noisy_motion = (
                target[0, :gen_frames, :, :].detach().cpu().numpy().transpose(1, 2, 0)
            )
            noisy_motion = np.expand_dims(noisy_motion, 0)
            cur_motions[-1] = np.concatenate(
                [cur_motions[-1][0:1], noisy_motion, cur_motions[-1][1:]], axis=0
            )
            cur_lengths[-1] = np.append(cur_lengths[-1], cur_lengths[-1][0])
            cur_texts.append(cur_texts[0])

            captions = [
                "Original",
                "Noisy Motion",
            ] + [f"Prediction {i+1}" for i in range(num_trials)]
            args.num_samples = 2 + num_trials

        all_motions.extend(cur_motions)
        all_lengths.extend(cur_lengths)
        all_text.extend(cur_texts)

    ### Save videos
    total_num_samples = args.num_samples * args.num_repetitions * num_dump_step

    # After concat -> [r1_dstep_1, r2_dstep_1, r3_dstep_1, r1_dstep_2, r2_dstep_2, ....]
    all_motions = np.concatenate(all_motions, axis=0)  # [bs * num_dump_step, 1, 3, 120]
    all_motions = all_motions[
        :total_num_samples
    ]  # [bs, njoints, 3, seqlen]
    all_text = all_text[:total_num_samples]  # len() = args.num_samples * num_dump_step
    all_lengths = np.concatenate(all_lengths, axis=0)[:total_num_samples]

    npy_path = os.path.join(out_path, "results.npy")

    print(f"saving results file to [{npy_path}]")
    np.save(
        npy_path,
        {
            "motion": all_motions,
            "text": all_text,
            "lengths": all_lengths,
            "num_samples": args.num_samples,
            "num_repetitions": args.num_repetitions,
        },
    )

    with open(npy_path.replace(".npy", ".txt"), "w") as fw:
        fw.write("\n".join(all_text))
    with open(npy_path.replace(".npy", "_len.txt"), "w") as fw:
        fw.write("\n".join([str(l) for l in all_lengths]))

    print(f"saving visualizations to [{out_path}]...")
    skeleton = paramUtil.t2m_kinematic_chain

    sample_files = []
    num_samples_in_out_file = num_dump_step 
    (
        sample_print_template,
        row_print_template,
        all_print_template,
        sample_file_template,
        row_file_template,
        all_file_template,
    ) = construct_template_variables(args.unconstrained)

    for sample_i in range(args.num_samples):
        rep_files = []

        print("saving", sample_i)
        caption = all_text[sample_i]
        length = all_lengths[sample_i]
        motion = all_motions[sample_i].transpose(2, 0, 1)[:length]
        save_file = sample_file_template.format(0, sample_i)
        print(
            sample_print_template.format(caption, 0, sample_i, save_file)
        )
        animation_save_path = os.path.join(out_path, save_file)
        print(f"Video Motion Shape: {motion.shape}")
        plot_3d_motion(
            animation_save_path,
            skeleton,
            motion,
            dataset=args.dataset,
            title=captions[sample_i],
            fps=fps,
            kframes=kframes,
            obs_list=obs_list,
            target_pose=target[0].cpu().numpy(),
            gt_frames=[kk for (kk, _) in kframes] if SHOW_TARGET_POSE else [],
        )
        rep_files.append(animation_save_path)

        # Check if we need to stack video
        sample_files = save_multiple_samples(
            args,
            out_path,
            row_print_template,
            all_print_template,
            row_file_template,
            all_file_template,
            caption,
            num_samples_in_out_file,
            rep_files,
            sample_files,
            sample_i,
        )

    abs_path = os.path.abspath(out_path)
    print(f"[Done] Results are at [{abs_path}]")

def calculate_results(motion_before_edit, generated_motions, target_motions, target_masks, 
                      max_frames, num_keyframe, text="", dataset: HumanML3D=None, 
                      holdout_before_edit_rep=None,
                      motion_before_edit_rep=None,
                      generated_motions_rep=None,
                      ):
    """
    Args:
        motion_before_edit: (tensor) [1, length, 22, 3]
        generated_motions: (tensor) [num_samples, length, 22, 3]
        target_masks: (tensor) [num_samples, length, 22, 3]
        max_frames: (int)
        text: (string)
    """
    metrics = {
        "Foot skating": [],
        "Jitter": [],
        "Content preservation": [],
        "Objective Error": [],
    }
    metrics_before_edit = {
        "Foot skating": [],
        "Jitter": [],
        "Objective Error": [],
    }

    left_foot_id = 10
    right_foot_id = 11
    left_hand_id = 20
    right_hand_id = 21
    head_id = 15
    opt_batch_size = len(generated_motions) // len(motion_before_edit)
    bf_edit_content_list = []
    # Before edit
    for i in range(len(motion_before_edit)):
        before_edit_cut = motion_before_edit[i, :max_frames, :, :]
        skate_ratio, _ = calculate_skating_ratio(before_edit_cut.permute(1, 2, 0).unsqueeze(0)) # need input shape [bs, 22, 3, max_len]
        metrics_before_edit['Foot skating'].append(skate_ratio.item())
        metrics_before_edit['Jitter'].append(compute_jitter(before_edit_cut).item())
        for j in range(opt_batch_size):
            target_idx = i * opt_batch_size + j

            metrics_before_edit['Objective Error'].append((torch.norm((before_edit_cut - target_motions[target_idx])
                                                                       * target_masks[target_idx], dim=2).sum() / num_keyframe).item())
        if 'jumping' in text or 'jump' in text:
            before_edit_above_ground = (before_edit_cut[:, left_foot_id, 1] > 0.05) & (before_edit_cut[:, right_foot_id, 1] > 0.05)
            bf_edit_content_list.append(before_edit_above_ground)
        elif 'raised hands' in text:
            before_edit_above_head = ((before_edit_cut[:, left_hand_id, 1] > before_edit_cut[:, head_id, 1]) & 
                                            (before_edit_cut[:, right_hand_id, 1] > before_edit_cut[:, head_id, 1]))
            bf_edit_content_list.append(before_edit_above_head)
        elif 'crawling' in text:
            before_edit_head_below = (before_edit_cut[:, head_id, 1] < 1.50)
            bf_edit_content_list.append(before_edit_head_below)

    # fid
    def calculate_fid(gt_motion, holdout_motion, gen_motion):
        # assume that the length = max_length
        device = gt_motion.device
        gt_length = torch.tensor([max_frames] * len(gt_motion))
        holdout_length = torch.tensor([max_frames] * len(holdout_motion))
        gen_length = torch.tensor([max_frames] * len(gen_motion))

        # fid_gt_gt2 = calculate_fid_given_two_populations(gt_motion, holdout_motion, gt_length, holdout_length, dataset=dataset, 
        #                                     dataset_name='humanml', device=device, batch_size=64)
        # fid_gt_gen = calculate_fid_given_two_populations(gt_motion, gen_motion, gt_length, gen_length, dataset=dataset, 
        #                                     dataset_name='humanml', device=device, batch_size=64)

        holdout1, holdout2 = torch.chunk(holdout_motion, 2, dim=0)
        h1_length, h2_length = torch.chunk(holdout_length, 2, dim=0)
        fid_h1_h2 = calculate_fid_given_two_populations(holdout1, holdout2, h1_length, h2_length, dataset=dataset, 
                                            dataset_name='humanml', device=device, batch_size=64)
        fid_h1_gen = calculate_fid_given_two_populations(holdout1, gen_motion, h1_length, gen_length, dataset=dataset, 
                                            dataset_name='humanml', device=device, batch_size=64)
        return {
            # f"fid_gt_holdout{len(holdout_motion)}": fid_h1_h2,
            # "fid_gt_gen": fid_gt_gen,
            "fid_h1_h2": fid_h1_h2,
            "fid_h1_gen": fid_h1_gen,
        }

    for i in range(len(generated_motions)):
        # Generated
        gen_cut = generated_motions[i, :max_frames, :, :]
        skate_ratio, _ = calculate_skating_ratio(gen_cut.permute(1, 2, 0).unsqueeze(0))
        metrics['Foot skating'].append(skate_ratio.item())
        metrics['Jitter'].append(compute_jitter(gen_cut).item())
        metrics['Objective Error'].append((torch.norm((gen_cut - target_motions[i]) * target_masks[i], dim=2).sum() / num_keyframe).item())
        first_gen_idx = i // opt_batch_size
        # Compute content preservation
        if 'jumping' in text or 'jump' in text:
            # Compute the ratio of matched frames where the feet are above the ground or touching the ground
            # First compute which frames in the generated motion that the feet are above the ground
            gen_above_ground = (gen_cut[:, left_foot_id, 1] > 0.05) & (gen_cut[:, right_foot_id, 1] > 0.05)
            content_ratio = (gen_above_ground == bf_edit_content_list[first_gen_idx]).sum() / max_frames
        elif 'raised hands' in text:
            # Compute the ratio of matched frames where the hands are above the head
            gen_above_head = (gen_cut[:, left_hand_id, 1] > gen_cut[:, head_id, 1]) & (gen_cut[:, right_hand_id, 1] > gen_cut[:, head_id, 1])
            content_ratio = (gen_above_head == bf_edit_content_list[first_gen_idx]).sum() / max_frames
        elif 'crawling' in text:
            # Compute the ratio of matched frames where the head is below 1.5m
            gen_head_below = (gen_cut[:, head_id, 1] < 1.50)
            content_ratio = (gen_head_below == bf_edit_content_list[first_gen_idx]).sum() / max_frames
        else:
            content_ratio = 0
            raise ValueError(f"Unknown text prompt for content evaluation: {text}")
        metrics['Content preservation'].append(content_ratio.item())

    # Calculate FID
    if holdout_before_edit_rep is not None:
        assert motion_before_edit_rep is not None and generated_motions_rep is not None, f"motion_before_edit_rep and generated_motions_rep must be provided if holdout_before_edit_rep is provided"
        fid_dict = calculate_fid(motion_before_edit_rep, holdout_before_edit_rep, generated_motions_rep)
    else:
        fid_dict = {}
    return metrics, metrics_before_edit, fid_dict

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

def ddim_loop_with_gradient(
    diffusion: GaussianDiffusion,
    model,
    shape,
    noise=None,
    clip_denoised=False,
    denoised_fn=None,
    cond_fn=None,
    model_kwargs=None,
    device=None,
    progress=False,
    eta=0.0,
    gradient_checkpoint=False,
):
    if device is None:
        device = next(model.parameters()).device
    assert isinstance(shape, (tuple, list))
    if noise is not None:
        img = noise
    else:
        img = torch.randn(*shape, device=device)

    indices = list(range(diffusion.num_timesteps))[::-1]

    if progress:
        # Lazy import so that we don't depend on tqdm.
        from tqdm.auto import tqdm

        indices = tqdm(indices)

    def grad_checkpoint_wrapper(func):
        def func_with_checkpoint(*args, **kwargs):
            return torch.utils.checkpoint.checkpoint(
                func, *args, **kwargs, use_reentrant=False
            )

        return func_with_checkpoint

    for i in indices:
        t = torch.tensor([i] * shape[0], device=device)
        sample_fn = diffusion.ddim_sample
        if gradient_checkpoint:
            sample_fn = grad_checkpoint_wrapper(sample_fn)
        out = sample_fn(
            model,
            img,
            t,
            clip_denoised=clip_denoised,
            denoised_fn=denoised_fn,
            cond_fn=cond_fn,
            model_kwargs=model_kwargs,
            eta=eta,
        )
        img = out["sample"]
    return img

@torch.no_grad()
def ddim_invert(
    diffusion: GaussianDiffusion,
    model,
    motion,  # image: torch.Tensor,
    model_kwargs,  # prompt,
    dump_steps=[],
    num_inference_steps=99,
    eta=0.0,
    clip_denoised=False,
    **kwds,
):
    """
    invert a real motion into noise map with determinisc DDIM inversion
    """
    latents = motion
    xt_list = [latents]
    pred_x0_list = [latents]
    indices = list(range(num_inference_steps))  # start_t #  - skip_timesteps))

    for i, t in enumerate(tqdm(indices, desc="DDIM Inversion")):
        t = torch.tensor([t] * latents.shape[0], device=latents.device)
        out = diffusion.ddim_reverse_sample(
            model,
            latents,
            t,
            model_kwargs=model_kwargs,
            eta=eta,
            clip_denoised=clip_denoised,
        )
        latents, pred_x0 = out["sample"], out["pred_xstart"]
        xt_list.append(latents)
        pred_x0_list.append(pred_x0)

    if len(dump_steps) > 0:
        pred_x0_list_out = []
        for ss in reversed(dump_steps):
            print("save step: ", ss)
            pred_x0_list_out.append(pred_x0_list[ss])
        return latents, pred_x0_list_out

    return latents


if __name__ == "__main__":
    main()
