#!/bin/bash
#SBATCH --job-name=res # nom du job
#SBATCH --partition=gpu_p2s # partition
#SBATCH --qos=qos_gpu-t4 # QoS
#SBATCH --output=res_%j.out    # fichier de sortie (%j = job ID)
#SBATCH --error=res_%j.err # fichier d’erreur (%j = job ID)
#SBATCH --time=48:00:00 # temps maximal d’allocation "(HH:MM:SS)"
#SBATCH --nodes=1 # reserver 1 nœud
#SBATCH --ntasks=1 # reserver 8 taches (ou processus MPI)
#SBATCH --gres=gpu:1
# reserver 8 GPU
#SBATCH --cpus-per-task=2
#SBATCH --hint=nomultithread # desactiver l’hyperthreading
#SBATCH -A izg@v100
#SBATCH --mail-user=cartel.gouabou@lis-lab.fr
#SBATCH --mail-type=FAIL

module purge # nettoyer les modules herites par defaut

module load pytorch-gpu/py3/1.11.0 
ulimit -n 4096
set -x # activer l’echo des commandes

cd $WORK/isic2018


srun python -u generate_result.py --loss_type CHML4 --weighting_type CS --hm_delay_type epoch --max_thresh 0.3





