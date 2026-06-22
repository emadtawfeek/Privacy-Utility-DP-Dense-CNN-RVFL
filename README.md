On PowerShell:
cd "dp_cvd_project"

# Install dependencies
py -3.12 -m pip install -r requirements.txt

# Full run
py -3.12 src\run_experiments.py --datasets mnist,cifar10,cardio,heart --models dense,cnn,rvfl --privacy both --epsilons 0.1,0.5,1,2,4,8 --seeds 42,43,44 --production-dp --rdp-alpha-mode wide --parallel --n_jobs 2 --torch_threads 2 --parallel-batch-size 4 --output_dir outputs
