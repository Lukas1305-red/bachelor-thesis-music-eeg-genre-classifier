#!/usr/bin/env bash
#SBATCH --time=00:10:00
#SBATCH --partition=students # or "staff" or "dfl" if you are a staff member
#SBATCH --gpus=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=5G
#SBATCH --output=./slurm_%A.out

# This activates the environment (given .envrc is in the current directory)
# If you do not want to have the .envrc in your current directory, use this snippet
# pushd $(pwd)
# cd path/to/devshell
# direnv allow . && eval "$(direnv export bash)"
# popd
direnv allow . && eval "$(direnv export bash)"

python --version

