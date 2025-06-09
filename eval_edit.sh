#!/bin/bash

#SBATCH --time=20:00:00
#SBATCH --ntasks=2
#SBATCH --mem-per-cpu=4000
#SBATCH --account=digital_human_jobs
#SBATCH --output=./logs/eval_edit%j.out

# Activate conda environment
source ~/.bashrc
conda activate /home/smarian/.conda/envs/gmd

# Define variables
SEED=10
PROMPTS=(
    # "a person is jumping"
    # "a person is crawling"
    "a person is doing a long jump"
    "a person is walking with raised hands"
)

for TEXT_PROMPT in "${PROMPTS[@]}"; do
    echo ""
    echo "================================================="
    echo "Processing prompt: $TEXT_PROMPT"
    echo "================================================="
    # Sanitize prompt for directory name (replace spaces with underscores, remove special chars)
    SANITIZED_PROMPT=$(echo "$TEXT_PROMPT" | tr ' ' '_' | sed 's/[^a-zA-Z0-9_]//g')

    echo "===================== L1 ====================="
    python -m eval.eval_edit_objectives --model_path save/mdm_avg_dno/model000500000_avg.pt --text_prompt "$TEXT_PROMPT" --seed $SEED --optimizer adam --objective_loss l1  --output_dir ./output/"$SANITIZED_PROMPT"/l1

    echo "===================== L2 ====================="
    python -m eval.eval_edit_objectives --model_path save/mdm_avg_dno/model000500000_avg.pt --text_prompt "$TEXT_PROMPT" --seed $SEED --optimizer adam --objective_loss l2 --output_dir ./output/"$SANITIZED_PROMPT"/l2

    echo "===================== Smooth L1 (Huber) ====================="
    python -m eval.eval_edit_objectives --model_path save/mdm_avg_dno/model000500000_avg.pt --text_prompt "$TEXT_PROMPT" --seed $SEED --optimizer adam --objective_loss smooth_l1 --output_dir ./output/"$SANITIZED_PROMPT"/smooth_l1

    echo "===================== German McClure ====================="
    python -m eval.eval_edit_objectives --model_path save/mdm_avg_dno/model000500000_avg.pt --text_prompt "$TEXT_PROMPT" --seed $SEED --optimizer adam --objective_loss german_mcclure --objective_sigma 0.5 --output_dir ./output/"$SANITIZED_PROMPT"/german_mcclure_sigma0.5
    python -m eval.eval_edit_objectives --model_path save/mdm_avg_dno/model000500000_avg.pt --text_prompt "$TEXT_PROMPT" --seed $SEED --optimizer adam --objective_loss german_mcclure --objective_sigma 1.0 --output_dir ./output/"$SANITIZED_PROMPT"/german_mcclure_sigma1.0
    python -m eval.eval_edit_objectives --model_path save/mdm_avg_dno/model000500000_avg.pt --text_prompt "$TEXT_PROMPT" --seed $SEED --optimizer adam --objective_loss german_mcclure --objective_sigma 2.0 --output_dir ./output/"$SANITIZED_PROMPT"/german_mcclure_sigma2.0

    echo "===================== Cauchy ====================="
    python -m eval.eval_edit_objectives --model_path save/mdm_avg_dno/model000500000_avg.pt --text_prompt "$TEXT_PROMPT" --seed $SEED --optimizer adam --objective_loss cauchy --objective_sigma 0.5 --output_dir ./output/"$SANITIZED_PROMPT"/cauchy_sigma0.5
    python -m eval.eval_edit_objectives --model_path save/mdm_avg_dno/model000500000_avg.pt --text_prompt "$TEXT_PROMPT" --seed $SEED --optimizer adam --objective_loss cauchy --objective_sigma 1.0 --output_dir ./output/"$SANITIZED_PROMPT"/cauchy_sigma1.0
    python -m eval.eval_edit_objectives --model_path save/mdm_avg_dno/model000500000_avg.pt --text_prompt "$TEXT_PROMPT" --seed $SEED --optimizer adam --objective_loss cauchy --objective_sigma 2.0 --output_dir ./output/"$SANITIZED_PROMPT"/cauchy_sigma2.0

    echo ""
done