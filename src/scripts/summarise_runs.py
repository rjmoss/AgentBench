import json
import os
import traceback
from json.decoder import JSONDecodeError

import pandas as pd

from configs import ConfigLoader
from scripts.compare_runs import flatten_dict
from typings import AssignmentConfig


def read_summary_metrics(path, assignment) -> dict:
    try:
        filepath = os.path.join(path, assignment.agent, assignment.task, "analysis", "overall.json")
        with open(filepath, "r") as f:
            return json.load(f)
    except (JSONDecodeError, FileNotFoundError):
        return {}


def main():
    paths = os.listdir("outputs")

    task_results = []
    for output in sorted(paths):
        if output.startswith('.') or output.startswith('ARCHIVE'):
            continue
        print(output)
        path = "outputs/" + output
        loader = ConfigLoader()
        try:
            config_ = loader.load_from(path + "/config.yaml")
        except Exception as err:
            print(f'Unable to read config for path {output} due to {err}: {traceback.format_exc()}')
            print('Continuing...')
            continue
        try:
            config = AssignmentConfig.parse_obj(config_)
            config = AssignmentConfig.post_validate(config)
            for assignment in config.assignments:
                dict_ = {"run_id": output}
                task_def = config.definition.task[assignment.task]
                dict_.update(assignment.dict())
                td = dict(task_def)
                td['parameters'].pop("data_config", None)
                td['parameters'].pop("docker_config", None)
                td['parameters'].pop("scripts", None)
                dict_.update(td)
                dict_.update(read_summary_metrics(path, assignment))
                dict_.update(config_.get('rh_config', {}))
                row = flatten_dict(dict_, use_separators=False, check_clash=True)
                task_results.append(row)
        except Exception as err:
            traceback.print_exc()
            raise err

    df = pd.DataFrame([flatten_dict(item) for item in task_results])
    df.insert(0, 'description', df.pop('description'))
    df.to_csv("run_summary.csv", index=False)


if __name__ == '__main__':
    main()
