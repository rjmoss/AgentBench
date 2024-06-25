import subprocess

module_name = "src.assigner"
num_runs = 3

for _ in range(num_runs):

    result = subprocess.run(["python", "-m", module_name])
    if result.returncode != 0:
        print(f"Run {_+1} failed with return code {result.returncode}")
        break
    else:
        print(f"Run {_+1} completed successfully")

