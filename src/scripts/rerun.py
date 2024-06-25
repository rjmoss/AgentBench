import os
import subprocess


# TODO - note that here I could set up multiple different yaml files which import different
# os.yaml or similar in order to set off a whole load of runs with different flags. That might
# the best way, or another way would be to change the assigner file so that it can take
# individual flags in the input and replace those specific pieces of the config that way there
# would be less duplication of config files just to change one bit of the os config.

# Also note that I can probably target the config.yaml from a particular run in outputs/
# to re-run it - though I need the main file to be the same as well, I haven't solved this
# so I'd have to keep it to ones where I haven't changed the default config yet/restarted the worker
# I guess I'd need to make this script a bit more full where it launches all the related modules from
# the various configs (if that's even possible, I think it is for the task_worker and isn't necessary
# for the controller). This would have the problem of needing to launch the server in a different thread
# so it's not blocking

def rerun(reruns: list[str]):
    current_dir = os.path.dirname(os.path.abspath(__file__))
    working_directory = os.path.abspath(os.path.join(current_dir, '..', '..'))

    for i, run in enumerate(reruns):
        if not run.endswith("config.yaml"):
            run = f"outputs/{run}/config.yaml"

        command = ["python", "-m", "src.assigner", "-r", "-c", run]

        result = subprocess.run(command, cwd=working_directory)
        if result.returncode != 0:
            print(f"Run {i + 1} failed with return code {result.returncode}")
            break
        else:
            print(f"Run {i + 1} completed successfully")


if __name__ == '__main__':
    rerun(
        reruns=[
            # "outputs/2024-07-08-13-19-25/config.yaml"
            "2024-07-09-17-40-28",
            "2024-07-09-17-57-45",
            "2024-07-09-18-15-54",
        ]
    )
