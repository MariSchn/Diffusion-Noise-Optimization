#!/bin/bash
#SBATCH --chdir=.
#SBATCH --time=2:00:00
#SBATCH --mem-per-cpu=10G
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

# Load modules and activate environment
source /work/scratch/mbilal/miniconda3/etc/profile.d/conda.sh 
conda activate /home/smarian/.conda/envs/gmd

# Make sure CUDA is available
python -c "import torch; print('Cuda available?', torch.cuda.is_available())"

# Set random seed for reproducibility
python -c "import torch; torch.manual_seed(42); print('Random seed set to 42')"

# Run the training script
python -m eval.eval_edit --model_path ./save/mdm_avg_dno/model000500000_avg.pt --text_prompt "a person is jumping" --seed 10 --noise_init clip_perlin --output_dir ./eval/jumping/clip0.5_perlin_0.5
# python -m eval.eval_edit --model_path ./save/mdm_avg_dno/model000500000_avg.pt --text_prompt "a person is crawling" --seed 10 --noise_init perlin --output_dir ./eval/crawling/perlin_z_norm
# python -m eval.eval_edit --model_path ./save/mdm_avg_dno/model000500000_avg.pt --text_prompt "a person is walking with raised hands" --seed 10 --noise_init perlin --output_dir ./eval/raised_hands/perlin_z_norm
# python -m eval.eval_edit --model_path ./save/mdm_avg_dno/model000500000_avg.pt --text_prompt "a person is doing a long jump" --seed 10 --output_dir ./eval/long_jump/rand


echo "Training completed successfully."
echo FINISHED at $(date)
