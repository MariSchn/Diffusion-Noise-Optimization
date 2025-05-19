import json
import math
import os
import pickle

import numpy as np
import torch

import data_loaders.humanml.utils.paramUtil as paramUtil
from data_loaders.get_data import DatasetConfig, get_dataset_loader
from data_loaders.humanml.data.dataset import HumanML3D, sample_to_motion
from data_loaders.humanml.utils.metrics import (calculate_skating_ratio,
                                                compute_jitter)
from data_loaders.humanml.utils.plot_script import plot_3d_motion
from data_loaders.tensors import collate
from dno import DNO, DNOOptions
from eval.calculate_fid import calculate_fid_given_two_populations
from model.cfg_sampler import ClassifierFreeSampleModel
from sample.condition import CondKeyLocationsLoss
from sample.gen_dno import ddim_invert, ddim_loop_with_gradient
from utils import dist_util
from utils.fixseed import fixseed
from utils.model_util import (create_gaussian_diffusion,
                              create_model_and_diffusion, load_model_wo_clip)
from utils.parser_util import generate_args


def plot_debug(motion_to_plot, name, gen_loader, length):
    plot_3d_motion(name, gen_loader.dataset.kinematic_chain, 
                   motion_to_plot[:length].detach().cpu().numpy(), 'length %d' % length, 'humanml', fps=20)


def calculate_results(motion_before_edit, generated_motions, target_motions, target_masks, 
                      max_frames, num_keyframe, text="", dataset: HumanML3D=None, 
                      holdout_before_edit_rep=None,
                      motion_before_edit_rep=None,
                      generated_motions_rep=None,
                      ):
    """
    Args:
        motion_before_edit: (tensor) [1, 196, 22, 3]
        generated_motions: (tensor) [num_samples, 196, 22, 3]
        target_masks: (tensor) [num_samples, 196, 22, 3]
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


def main():
    '''
    Evaluation code for location editing. We used 4 prompts for evaluation:
    "a person is walking with raised hands", "a person is jumping", 
    "a person is crawling", "a person is doing a long jump"
    The command is: python -m eval.eval_edit --model_path ./save/mdm_avg/model000500000.pt --text_prompt "...(above text)..."
    '''
    max_samples = 6 # 32
    opt_batch_size = 6 # 4
    num_total_batches = math.ceil(max_samples / opt_batch_size)
    # We will generate a new original motion for each batch
    n_keyframe = 1


    args = generate_args()
    args.device = 0
    args.use_ddim = True
    print(args.__dict__)
    print(args.arch)

    fixseed(args.seed)

    num_ode_steps = 10
    OPTIMIZATION_STEP = 200
    if "lbfgs" in args.optimizer:
        OPTIMIZATION_STEP = 20
    noise_opt_conf = DNOOptions(
        num_opt_steps=OPTIMIZATION_STEP,
        diff_penalty_scale=2e-3,
        decorrelate_scale=0,
        optimizer=args.optimizer,
    )

    # optimization steps at which metrics will be calculated (currently hard coded to evaluate at 10 steps)
    evaluate_at = np.linspace(0, OPTIMIZATION_STEP - 1, num=10, dtype=int).tolist()

    out_path = args.output_dir
    name = os.path.basename(os.path.dirname(args.model_path))
    niter = os.path.basename(args.model_path).replace("model", "").replace(".pt", "")

    args.max_frames = 196
    fps = 20
    args.fps = fps
    n_frames = min(args.max_frames, int(args.motion_length * fps))
    print("n_frames", n_frames)
    dist_util.setup_dist(args.device)
    # Output directory
    out_path = os.path.join(
        out_path,
        "eval_edit_{}".format(niter),
    )
    out_path = os.path.join(out_path, f"seed{args.seed}")
    out_path += "_" + args.text_prompt.replace(" ", "_").replace(".", "")

    # this block must be called BEFORE the dataset is loaded
    if args.text_prompt != "":
        args.num_samples = num_total_batches
        texts = [args.text_prompt] * args.num_samples
    else:
        raise ValueError("Please specify either text_prompt or input_text")

    # NOTE: Currently not supporting multiple repetitions due to the way we handle trajectory model
    args.num_repetitions = 1

    assert (
        args.num_samples <= args.batch_size
    ), f"Please either increase batch_size({args.batch_size}) or reduce num_samples({args.num_samples})"

    args.batch_size = (
        args.num_samples
    )  # Sampling a single batch from the testset, with exactly args.num_samples

    print("Loading dataset...")
    data = load_dataset(args, n_frames)

    print("Creating model and diffusion...")
    model, diffusion_ori = create_model_and_diffusion(args, data)

    ###################################
    # LOADING THE MODEL FROM CHECKPOINT
    print(f"Loading checkpoints from [{args.model_path}]...")
    state_dict = torch.load(args.model_path, map_location="cpu")
    load_model_wo_clip(model, state_dict)

    if args.guidance_param != 1:
        model = ClassifierFreeSampleModel(
            model
        )  # wrapping model with the classifier-free sampler
    model.to(dist_util.dev())
    model.eval()  # disable random masking
    model_device = next(model.parameters()).device
    ###################################

    collate_args = [
        {
            "inp": torch.zeros(n_frames),
            "tokens": None,
            "lengths": n_frames,
        }
    ] * args.num_samples

    is_t2m = any([args.input_text, args.text_prompt])
    if is_t2m:
        # t2m
        collate_args = [
            dict(arg, text=txt) for arg, txt in zip(collate_args, texts)
        ]

    _, model_kwargs = collate(collate_args)

    model_kwargs["y"]["traj_model"] = False

    #############################################

    all_motions = []
    all_text = []
    obs_list = []

    model_device = next(model.parameters()).device
    # Output path
    os.makedirs(out_path, exist_ok=True)
    args_path = os.path.join(out_path, "args.json")
    with open(args_path, "w") as fw:
        json.dump(vars(args), fw, indent=4, sort_keys=True)

    ############################################

    # add CFG scale to batch
    if args.guidance_param != 1:
        model_kwargs["y"]["scale"] = (
            torch.ones(args.batch_size, device=dist_util.dev())
            * args.guidance_param
        )
    #####################################################

    # Pass functions to the diffusion
    diffusion_ori.data_get_mean_fn = data.dataset.t2m_dataset.get_std_mean
    diffusion_ori.data_transform_fn = data.dataset.t2m_dataset.transform_th
    diffusion_ori.data_inv_transform_fn = data.dataset.t2m_dataset.inv_transform_th

    ###################
    # MODEL INFERENCING
    ###################
    initial_noise = torch.randn((args.batch_size, model.njoints, model.nfeats, n_frames)).to(model_device)
    
    sample_file = os.path.join(out_path, "sample_before_edit.pt")

    # List of samples of shape [bs, njoints, nfeats, nframes]
    sample = diffusion_ori.ddim_sample_loop(
        model,
        (args.batch_size, model.njoints, model.nfeats, n_frames),
        clip_denoised=False,
        # clip_denoised=not args.predict_xstart,
        model_kwargs=model_kwargs,
        skip_timesteps=0,
        init_image=None,
        progress=True,
        dump_steps=None,
        noise=None,
        const_noise=False,
        cond_fn=None,
    )
    # Save sample to file
    torch.save(sample, sample_file)

    def generate_holdout_dataset_for_FID(N_holdout, batch_size):
        # generating the hold out dataset for FID calculation
        assert N_holdout // batch_size > 0, f"N_holdout {N_holdout} must be larger than batch_size {batch_size}"
        assert N_holdout % batch_size == 0, f"N_holdout {N_holdout} must be divisible by batch_size {batch_size}"

        def create_model_args(batch_size):
            texts = [args.text_prompt] * batch_size
            collate_args = [
                {
                    "inp": torch.zeros(n_frames),
                    "tokens": None,
                    "lengths": n_frames,
                }
            ] * batch_size
            is_t2m = any([args.input_text, args.text_prompt])
            if is_t2m:
                collate_args = [
                    dict(arg, text=txt) for arg, txt in zip(collate_args, texts)
                ]
            _, model_kwargs = collate(collate_args)
            if args.guidance_param != 1:
                model_kwargs["y"]["scale"] = (
                    torch.ones(batch_size, device=dist_util.dev())
                    * args.guidance_param
                )
            model_kwargs["y"]["traj_model"] = args.traj_only
            return model_kwargs

        out = []
        # generate the holdout dataset batch by batch
        for i in range(N_holdout // batch_size):
            model_kwargs = create_model_args(batch_size)
            # if the file exists
            if os.path.exists(os.path.join(out_path, f"holdout_{i}.pt")):
                print(f" - Holdout sample {i} already exists. Loading from file.")
                sample = torch.load(os.path.join(out_path, f"holdout_{i}.pt"), map_location='cpu')
            else:
                # generate a batch of data
                # without this the generation will be identical.
                torch.manual_seed(i)
                torch.cuda.manual_seed(i)
                sample = diffusion_ori.ddim_sample_loop(
                    model,
                    (batch_size, model.njoints, model.nfeats, n_frames),
                    clip_denoised=False,  # not args.predict_xstart,
                    model_kwargs=model_kwargs,
                    skip_timesteps=0,
                    init_image=None,
                    progress=True,
                    dump_steps=None,
                    noise=None,
                    const_noise=False,
                    cond_fn=None,
                ).cpu().clone()
                # save the data
                torch.save(sample, os.path.join(out_path, f"holdout_{i}.pt"))
            out.append(sample)
        out = torch.cat(out, dim=0)
        return out
        
    # Used to be the code for FID, but FID is not meaningful in this case.
    do_calculate_fid = False

    if do_calculate_fid:
        N_holdout = 200
        sample_holdout = generate_holdout_dataset_for_FID(N_holdout, batch_size=100)

    sample_before_edit = sample.clone()
    #######################
    ##### Edting here #####
    #######################

    skeleton = (
        paramUtil.kit_kinematic_chain
        if args.dataset == "kit"
        else paramUtil.t2m_kinematic_chain
    )
    task = "trajectory_editing"
    if task == "trajectory_editing":
        # Get obstacle list
        # obs_list = get_obstacles()

        ### Random sample the keyframes and target locations here ###
        obs_list = []
        selected_index = [102]
        target_locations = [(2, 2)]
        target = torch.zeros([1, args.max_frames, 22, 3], device=model_device)
        target_mask = torch.zeros_like(target, dtype=torch.bool)
        kframes = [
            (tt, locs) for (tt, locs) in zip(selected_index, target_locations)
        ]
        for tt, locs in zip(selected_index, target_locations):
            # print("target at %d = %.1f, %.1f" % (tt, locs[0], locs[1]))
            target[0, tt, 0, [0, 2]] = torch.tensor(
                [locs[0], locs[1]], dtype=torch.float32, device=target.device
            )
            target_mask[0, tt, 0, [0, 2]] = True
    else:
        raise ValueError("Unknown task")

    ######################
    ## START OPTIMIZING ##
    ######################
    output_lists = [[] for _ in range(len(evaluate_at))] # Store the outputs at different (intermediate) steps
    target_list = []
    target_mask_list = []
    
    for ii in range(num_total_batches):
        seed_number = ii
        fixseed(seed_number)

        ## Sample points    
        # reusing the target if it exists
        target_batch_file = f'target_{ii:04d}.pt'
        target_batch_file = os.path.join(out_path, target_batch_file)
        if os.path.exists(target_batch_file):
            # [batch_size, n_keyframe]
            saved_target = torch.load(target_batch_file, map_location=model_device)
            target, target_mask = saved_target['target'], saved_target['target_mask']
            print(f'sample keyframes {target_batch_file} exists, loading from file')
        else:
            min_frame = 60
            max_frame = 90
            sampled_keyframes =  ((max_frame - min_frame) * torch.rand(opt_batch_size, n_keyframe) + min_frame).long()
            max_x, max_z = 2.0, 2.0
            sampled_locations = (torch.rand(opt_batch_size, n_keyframe, 2) * 2 - 1) * torch.tensor([max_x, max_z])
            sampled_locations = sampled_locations.to(model_device)
            target = torch.zeros([opt_batch_size, n_frames, 22, 3], device=model_device)
            target_mask = torch.zeros_like(target, dtype=torch.bool)
            for bb in range(opt_batch_size):
                for jj in range(n_keyframe):
                    target[bb, sampled_keyframes[bb, jj], 0, [0, 2]] = sampled_locations[bb, jj]
                    target_mask[bb, sampled_keyframes[bb, jj], 0, [0, 2]] = True

            torch.save({'target': target, 'target_mask': target_mask}, target_batch_file)
        # Add target to list
        for jj in range(opt_batch_size):
            target_list.append(target[jj])
            target_mask_list.append(target_mask[jj])

        batch_file = f"batch_{ii}.pt"
        batch_path = os.path.join(out_path, batch_file)
        # Load results if exists
        if os.path.exists(batch_path):
            print(f"Loading results from [{batch_path}]")
            final_motion = torch.load(batch_path)
        else:
            target = target
            target_mask = target_mask
            diffusion = create_gaussian_diffusion(args, f"ddim{num_ode_steps}")
            if args.guidance_param != 1:
                model_kwargs["y"]["scale"] = (
                    torch.ones(opt_batch_size, device=model_device)
                    * args.guidance_param
                )

            # --- START OF Batched Random Search ---
            print(f"Performing random search (batched targets, looping trials) for DNO batch {ii+1}/{num_total_batches} (opt_batch_size: {opt_batch_size})...")
            num_random_search_trials = 100  # Number of random noises to try for each item in opt_batch

            # Initialize placeholders for the best noise found for each of the opt_batch_size items
            # and their corresponding minimum losses.
            # These will be updated across the trials.
            best_noises_for_opt_batch = torch.zeros(
                (opt_batch_size, model.njoints, model.nfeats, n_frames), 
                device=model_device
            )
            min_losses_for_opt_batch = torch.full(
                (opt_batch_size,), float('inf'), device=model_device
            )

            # The `model_kwargs` prepared before this block is already set up for `opt_batch_size`.
            # This includes text conditioning, scale, etc. This is exactly what we need for each trial.
            # The `target` and `target_mask` are also already for `opt_batch_size`.

            # Create the loss function once, as it will operate on opt_batch_size items.
            # `target` and `target_mask` are already shaped [opt_batch_size, n_frames, 22, 3]
            loss_fn_for_opt_batch = CondKeyLocationsLoss(
                target=target, # This is the `target` for the current DNO batch of size opt_batch_size
                target_mask=target_mask, # Similarly for `target_mask`
                transform=data.dataset.t2m_dataset.transform_th,
                inv_transform=data.dataset.t2m_dataset.inv_transform_th,
                abs_3d=False,
                use_mse_loss=False,
                use_rand_projection=False,
                obs_list=obs_list,
            )
            # The `model_kwargs` here is the one prepared for the DNO step (size opt_batch_size)
            criterion_for_opt_batch = lambda x_gen: loss_fn_for_opt_batch(x_gen, **model_kwargs)

            # Loop through the number of random search trials
            for trial_num in range(num_random_search_trials):
                if trial_num % (num_random_search_trials // 5 + 1) == 0 : # Print progress periodically
                     print(f"  - Random search trial {trial_num + 1}/{num_random_search_trials}...")
                
                # Generate a batch of random noise candidates, one for each of the opt_batch_size items.
                # Shape: [opt_batch_size, model.njoints, model.nfeats, n_frames]
                current_trial_candidate_noises = torch.randn(
                    (opt_batch_size, model.njoints, model.nfeats, n_frames), 
                    device=model_device
                )
                
                with torch.no_grad(): # Random search evaluation does not require gradients
                    # Generate motions for all opt_batch_size items using their respective candidate noises for this trial.
                    # `model_kwargs` is already suitable for opt_batch_size.
                    x0_candidate_motions_trial = diffusion_ori.ddim_sample_loop(
                        model,
                        (opt_batch_size, model.njoints, model.nfeats, n_frames),
                        noise=current_trial_candidate_noises,
                        model_kwargs=model_kwargs, # This is the main model_kwargs for opt_batch_size
                        clip_denoised=False,
                        progress=False, # Suppress progress bar for these inner DDIM calls
                    )
                    
                    # Calculate losses for all opt_batch_size generated motions against their respective targets.
                    # current_trial_losses will have shape [opt_batch_size]
                    current_trial_losses = criterion_for_opt_batch(x0_candidate_motions_trial)

                # Update best_noises_for_opt_batch and min_losses_for_opt_batch
                # Find which items in the current trial had a better (lower) loss than previously found.
                improvement_mask = current_trial_losses < min_losses_for_opt_batch
                
                # Update the minimum losses for those improved items
                min_losses_for_opt_batch[improvement_mask] = current_trial_losses[improvement_mask]
                
                # Update the best noises for those improved items
                # Unsqueeze improvement_mask for broadcasting to noise dimensions
                best_noises_for_opt_batch[improvement_mask] = current_trial_candidate_noises[improvement_mask]

            # After all trials, best_noises_for_opt_batch contains the best noise found for each item.
            cur_xt = best_noises_for_opt_batch
            
            # Check if any item didn't find a noise (e.g., if all trials resulted in inf loss, though unlikely)
            if torch.isinf(min_losses_for_opt_batch).any():
                num_still_inf = torch.isinf(min_losses_for_opt_batch).sum().item()
                print(f"Warning: {num_still_inf} item(s) in the batch still have inf loss after random search. This is unexpected.")
            
            print(f"Random search (batched targets, looping trials) complete for DNO batch {ii+1}.")
            print(f"  Min losses found for batch items: {min_losses_for_opt_batch.cpu().numpy()}")
            # --- END of Batched Random Search ---

            # cur_xt is now [opt_batch_size, model.njoints, model.nfeats, n_frames]
            # Each of the opt_batch_size noises is the best found for its corresponding target.
            cur_xt = cur_xt.detach().requires_grad_()

            ######## Main optimization loop #######
                        ######## Use best noise from random search #######
            # After random search, cur_xt contains the best noise found
            
            # Generate final motion using best noise from random search
            final_motion = diffusion_ori.ddim_sample_loop(
                model,
                (opt_batch_size, model.njoints, model.nfeats, n_frames),
                noise=cur_xt,
                model_kwargs=model_kwargs,
                clip_denoised=False,
                progress=True,
            )
            
            # Save the best motion
            torch.save(final_motion, os.path.join(out_path, f"batch_{ii}.pt"))
            
            # Create a simple history structure for compatibility with the rest of the code
            hist = [{"z": best_noises_for_opt_batch[i:i+1]} for i in range(opt_batch_size)]
            
            # For evaluation, use the same best noise for all steps
            for i, step in enumerate(evaluate_at):
                # Add to output lists
                for j in range(opt_batch_size):
                    output_lists[i].append(final_motion[j])
            #######################################

    #######################
    ### COMPUTE RESULTS ###
    #######################

    for output_list, step in zip(output_lists, evaluate_at): 
        generated_motions = []
        generated_motions_rep = []
        # convert the generated motion to skeleton
        for generated in output_list:
            generated_motions_rep.append(generated)
            generated = sample_to_motion(generated.unsqueeze(0), data.dataset, model, abs_3d=False)
            generated = generated.permute(0, 3, 1, 2)
            generated_motions.append(generated)
        # for FID
        generated_motions_rep = torch.stack(generated_motions_rep, dim=0)
        motion_before_edit = sample_to_motion(sample_before_edit, data.dataset, model, abs_3d=False).permute(0, 3, 1, 2)

        # (num_samples, x, x, x)
        generated_motions = torch.cat(generated_motions, dim=0)
        target_motions = torch.stack(target_list, dim=0).detach().cpu()
        target_masks = torch.stack(target_mask_list, dim=0).detach().cpu()

        save_dir = out_path
        # save_dir = os.path.join(os.path.dirname(args.model_path), f'eval_{task}_{noise_opt_conf.name}')
        log_file = os.path.join(save_dir, f'eval_N{max_samples}.txt')

        # * Currently disabled to prevent exceeding storage quota
        DEBUG = True
        if DEBUG:
            print("Saving debug videos...")
            for ii in range(len(motion_before_edit)):
                before_edit_id = f'{ii:05d}'
                plot_debug(motion_before_edit[ii], os.path.join(save_dir, f"before_edit_{before_edit_id}.mp4"), data, n_frames)

            start_from = 0  # 14
            for ii in range(start_from, len(generated_motions)): 
                motion_id = f'{ii:05d}'
                before_edit_id = f'{(ii//opt_batch_size):05d}'
                plot_debug(generated_motions[ii], os.path.join(save_dir, f"{motion_id}_gen.mp4"), data, n_frames)
                # plot_debug(target_motions[ii], os.path.join(save_dir, f"{motion_id}_target.mp4"), gen_loader, motion_lengths[ii])
                # Concat the two videos
                os.system(f"ffmpeg -y -loglevel warning -i {save_dir}/before_edit_{before_edit_id}.mp4 -i {save_dir}/{motion_id}_gen.mp4 -filter_complex hstack {save_dir}/{motion_id}_combined.mp4")
                # Remove the generated video
                os.system(f"rm {save_dir}/{motion_id}_gen.mp4")
                # if ii > 20:
                if ii > 5:
                    break
                
        # * Currently disabled to prevent exceeding storage quota
        SAVE_FOR_VIS = False
        if SAVE_FOR_VIS:
            # Edited motion
            npy_path = os.path.join(out_path, "results.npy")
            all_motions = generated_motions.permute(0, 2, 3, 1).detach().cpu().numpy()
            print(f"saving results file to [{npy_path}]")
            np.save(
                npy_path,
                {
                    "motion": all_motions,
                    "text": all_text,
                    "lengths": np.array([n_frames] * len(generated_motions)),
                    "num_samples": args.num_samples,
                    "num_repetitions": args.num_repetitions,
                },
            )
            # Before edit motion
            npy_path = os.path.join(out_path, "results_before_edit.npy")
            all_motions = motion_before_edit.permute(0, 2, 3, 1).detach().cpu().numpy()
            print(f"saving results file to [{npy_path}]")
            np.save(
                npy_path,
                {
                    "motion": all_motions,
                    "text": all_text,
                    "lengths": np.array([n_frames] * len(motion_before_edit)),
                    "num_samples": args.num_samples,
                    "num_repetitions": args.num_repetitions,
                },
            )
            # Save pelvis location change
            # Save additional_objects to a pickle file
            pickle_file = os.path.join(out_path, "edit_trajectories.pkl")
            keyframes_edit = []
            st_edit = []
            ed_edit = []
            # loop over target mask to find where the keyframes are for each motion
            for cur_id, target_mask in enumerate(target_masks):
                if cur_id > 15:
                    break
                keyframe = torch.where(target_mask)[0][0].data
                keyframes_edit.append(keyframe)
                before_edit_idx = 0
                st_edit.append(motion_before_edit[before_edit_idx, keyframe, 0, :])
                ed_edit.append(generated_motions[cur_id, keyframe, 0, :])
            keyframes_edit = torch.stack(keyframes_edit).detach().cpu().numpy()
            st_edit = torch.stack(st_edit).detach().cpu().numpy()
            ed_edit = torch.stack(ed_edit).detach().cpu().numpy()
            edit_trajs = {
                "keyframes": keyframes_edit,
                "start": st_edit,
                "end": ed_edit,
            }
            with open(pickle_file, "wb") as f:
                pickle.dump(edit_trajs, f)


        metrics, metrics_before_edit, fid = calculate_results(motion_before_edit, generated_motions, target_motions, 
                                                        target_masks, n_frames, n_keyframe, text=args.text_prompt, 
                                                        dataset=data.dataset, 
                                                        motion_before_edit_rep=sample_before_edit if do_calculate_fid else None,
                                                        holdout_before_edit_rep=sample_holdout if do_calculate_fid else None,
                                                        generated_motions_rep=generated_motions_rep if do_calculate_fid else None,
                                                        )

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

            # Save metrics to a JSON file
            metrics_file = os.path.join(save_dir, f"metrics_step_{step}.json")
            with open(metrics_file, "w") as metrics_fw:
                json.dump({"Before Edit": metrics_before_edit, "After Edit": metrics, "FID": fid}, metrics_fw, indent=4)

            print(f"==================== FID ====================")
            print(f"==================== FID ====================", file=f, flush=True)
            for k, v in fid.items():
                print(f"{k}: {v:.4f}")
                print(f"{k}: {v:.4f}", file=f, flush=True)

    return


def load_dataset(args, n_frames):
    print(f"args: {args}")
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
