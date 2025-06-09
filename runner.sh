#!/bin/bash
#SBATCH --chdir=.
#SBATCH --time=48:00:00
#SBATCH --mem-per-cpu=$SEEDG
#SBATCH -A digital_human_jobs
#SBATCH --output="training_log.out"
#SBATCH --gpus=1
#SBATCH --job-name=training_public
#SBATCH --mail-type=FAIL

set -e
set -o xtrace
echo PWD:$(pwd)
echo STARTING AT $(date)

. /etc/profile.d/modules.sh 

# Define variables
SEED=10
PROMPTS=(
    "a person is jumping"
    "a person is crawling"
    "a person is doing a long jump"
    "a person is walking with raised hands"
)
# TODO: REPLACE WITH YOUR MODEL PATH
MODEL_PATH="./save/model000500000_avg.pt"

# Load modules and activate environment
# TODO: REPLACE WITH YOUR CONDA ENVIRONMENT
source ~/.bashrc
conda activate gmd

# Make sure CUDA is available
python -c "import torch; print('Cuda available?', torch.cuda.is_available())"

# Set random seed for reproducibility
python -c "import torch; torch.manual_seed(42); print('Random seed set to 42')"

# ===== NOISE INITIALIZATION =====

# Run the training script
python -m eval.eval_edit --model_path $MODEL_PATH --text_prompt "a person is jumping" --seed $SEED --noise_init clip_perlin --output_dir ./eval/jumping/clip0.5_perlin_0.5
python -m eval.eval_edit --model_path $MODEL_PATH --text_prompt "a person is crawling" --seed $SEED --noise_init perlin --output_dir ./eval/crawling/perlin_z_norm
python -m eval.eval_edit --model_path $MODEL_PATH --text_prompt "a person is walking with raised hands" --seed $SEED --noise_init perlin --output_dir ./eval/raised_hands/perlin_z_norm
python -m eval.eval_edit --model_path $MODEL_PATH --text_prompt "a person is doing a long jump" --seed $SEED --output_dir ./eval/long_jump/rand

python -m eval.eval_edit --model_path $MODEL_PATH --text_prompt "a person is jumping" --seed $SEED --output_dir ./eval/jumping/search
python -m eval.eval_edit_search --model_path $MODEL_PATH --text_prompt "a person is jumping" --seed $SEED --output_dir ./eval/jumping/search
python -m eval.eval_edit_search_only --model_path $MODEL_PATH --text_prompt "a person is jumping" --seed $SEED --output_dir ./eval/jumping/search_only

python -m eval.eval_edit_search --model_path $MODEL_PATH --text_prompt "a person is crawling" --seed $SEED --output_dir ./eval/crawling/search
python -m eval.eval_edit_search_only --model_path $MODEL_PATH --text_prompt "a person is crawling" --seed $SEED --output_dir ./eval/crawling/search_only

python -m eval.eval_edit_search --model_path $MODEL_PATH --text_prompt "a person is walking with raised hands" --seed $SEED --output_dir ./eval/raised_hands/search
python -m eval.eval_edit_search_only --model_path $MODEL_PATH --text_prompt "a person is walking with raised hands" --seed $SEED --output_dir ./eval/raised_hands/search_only

python -m eval.eval_edit_search --model_path $MODEL_PATH --text_prompt "a person is doing a long jump" --seed $SEED --output_dir ./eval/long_jump/search
python -m eval.eval_edit_search_only --model_path $MODEL_PATH --text_prompt "a person is doing a long jump" --seed $SEED --output_dir ./eval/long_jump/search_only

# ===== OPTIMIZER ABLATION =====

for TEXT_PROMPT in "${PROMPTS[@]}"; do
    # Sanitize prompt for directory name (replace spaces with underscores, remove special chars)
    SANITIZED_PROMPT=$(echo "$TEXT_PROMPT" | tr ' ' '_' | sed 's/[^a-zA-Z0-9_]//g')

    python -m eval.eval_edit --model_path $MODEL_PATH --text_prompt "$TEXT_PROMPT" --seed $SEED --optimizer sgd --output_dir ./eval/optim_ablation/"$SANITIZED_PROMPT"/sgd
    python -m eval.eval_edit --model_path $MODEL_PATH --text_prompt "$TEXT_PROMPT" --seed $SEED --optimizer rmsprop --output_dir ./eval/optim_ablation/"$SANITIZED_PROMPT"/rmsprop
    python -m eval.eval_edit --model_path $MODEL_PATH --text_prompt "$TEXT_PROMPT" --seed $SEED --optimizer adam --output_dir ./eval/optim_ablation/"$SANITIZED_PROMPT"/adam
    python -m eval.eval_edit --model_path $MODEL_PATH --text_prompt "$TEXT_PROMPT" --seed $SEED --optimizer adamw --output_dir ./eval/optim_ablation/"$SANITIZED_PROMPT"/adamw
    python -m eval.eval_edit --model_path $MODEL_PATH --text_prompt "$TEXT_PROMPT" --seed $SEED --optimizer lbfgs --output_dir ./eval/optim_ablation/"$SANITIZED_PROMPT"/lbfgs
    python -m eval.eval_edit --model_path $MODEL_PATH --text_prompt "$TEXT_PROMPT" --seed $SEED --optimizer lbfgs_normalized --output_dir ./eval/optim_ablation/"$SANITIZED_PROMPT"/lbfgs_normalized

done

# ===== LONG-TERM/BLEND EVALUATION =====

Generate initial motions
python -m sample.text_to_motion --model_path $MODEL_PATH --text_prompt "a person is walking forward" --output_dir ./save/walk
python -m sample.text_to_motion --model_path $MODEL_PATH --text_prompt "a person is jumping" --output_dir ./save/jumping

python -m sample.text_to_motion --model_path $MODEL_PATH --text_prompt "a person is doing a long jump" --output_dir ./save/long_jump
python -m sample.text_to_motion --model_path $MODEL_PATH --text_prompt "a person is crawling" --output_dir ./save/crawling

# Run evaluations
python -m eval.eval_blend --model_path $MODEL_PATH --seed $SEED --load_from ./save/walk --text_prompt "a person walking and then jumping" --load_from_2 ./save/jumping --output_dir ./eval/blend_easy/dno  --endpoint_weight 1 --num_offset 20 --seam_width 10 --noise_init random
python -m eval.eval_blend --model_path $MODEL_PATH --seed $SEED --load_from ./save/walk --text_prompt "a person walking and then jumping" --load_from_2 ./save/jumping --output_dir ./eval/blend_easy/dno_weighted --endpoint_weight 10 --num_offset 0 --seam_width 10 --noise_init random
python -m eval.eval_blend --model_path $MODEL_PATH --seed $SEED --load_from ./save/walk --text_prompt "a person walking and then jumping" --load_from_2 ./save/jumping --output_dir ./eval/blend_easy/dno_weighted_concat_noise --endpoint_weight 10 --num_offset 0 --seam_width 10 --noise_init concat
python -m eval.eval_blend --model_path $MODEL_PATH --seed $SEED --load_from ./save/walk --text_prompt "a person walking and then jumping" --load_from_2 ./save/jumping --output_dir ./eval/blend_easy/dno_weighted_sin_noise --endpoint_weight 10 --num_offset 0 --seam_width 10 --noise_init sin

python -m eval.eval_blend --model_path $MODEL_PATH --seed $SEED --load_from ./save/long_jump --text_prompt "a person doing a long jump and then crawling" --load_from_2 ./save/crawling --output_dir ./eval/blend_hard/dno  --endpoint_weight 1 --num_offset 20 --seam_width 30 --noise_init random
python -m eval.eval_blend --model_path $MODEL_PATH --seed $SEED --load_from ./save/long_jump --text_prompt "a person doing a long jump and then crawling" --load_from_2 ./save/crawling --output_dir ./eval/blend_hard/dno_weighted --endpoint_weight 10 --num_offset 0 --seam_width 30 --noise_init random
python -m eval.eval_blend --model_path $MODEL_PATH --seed $SEED --load_from ./save/long_jump --text_prompt "a person doing a long jump and then crawling" --load_from_2 ./save/crawling --output_dir ./eval/blend_hard/dno_weighted_concat_noise --endpoint_weight 10 --num_offset 0 --seam_width 30 --noise_init concat
python -m eval.eval_blend --model_path $MODEL_PATH --seed $SEED --load_from ./save/long_jump --text_prompt "a person doing a long jump and then crawling" --load_from_2 ./save/crawling --output_dir ./eval/blend_hard/dno_weighted_sin_noise --endpoint_weight 10 --num_offset 0 --seam_width 30 --noise_init sin

echo "Training completed successfully."
echo FINISHED at $(date)
