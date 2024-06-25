import os
import shutil


def clean_up():
    removed = []
    paths = os.listdir("outputs")
    for path in paths:
        output = os.path.join(os.getcwd(), "outputs", path)
        dirs = os.listdir(output)
        if dirs == ["config.yaml"]:
            # Remove directories which only have the config yaml file
            # (might want other cleanup because in a lot of cases I start but don't finish
            # so might be better to remove if it doesn't have the overall file
            shutil.rmtree(output)
            removed.append(path)

    print(f'Removed from outputs {len(removed)}: {removed}')

    removed = []
    paths = os.listdir("analysis")
    for path in paths:
        if path.startswith('.'):
            continue
        output = os.path.join(os.getcwd(), "analysis", path)
        dirs = os.listdir(output)
        if not dirs:
            shutil.rmtree(output)
            removed.append(path)

    print(f'Removed from analysis {len(removed)}: {removed}')


if __name__ == '__main__':
    clean_up()
