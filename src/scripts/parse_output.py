import json
import logging
import os
import traceback
from abc import ABC, abstractmethod
from typing import List

import numpy as np
import pandas as pd
import torch
import yaml
from pydantic import BaseModel
from torch import Tensor
from torchmetrics.classification import BinaryAUROC

from configs import ConfigLoader
from server.tasks.os_interaction import OSInteraction
from server.tasks.os_interaction.prompts import prompt_dict, PromptType, get_prompt_from_str
from typings import TaskClientOutput, TaskOutput, AssignmentConfig, Assignment, ChatHistoryItem, InstanceFactory
from utils import clear_file
from utils.metrics import (
    plot_calibration_curve,
    calculate_ece,
    plot_confidence_against_iterations,
    plot_roc_curve,
    calculate_temperature_scale, scale_probs_with_temp, calculate_overall_benchmark_score,
)


SHOW = False


class TaskResult(BaseModel):
    assignment: Assignment
    overall: dict
    task_def: InstanceFactory
    task_client_outputs: List[TaskClientOutput]

    @property
    def task_outputs(self) -> List[TaskOutput]:
        return [tco.output for tco in self.task_client_outputs]


class ResultsParser:
    def __init__(self, config: AssignmentConfig):
        self.config = config
        self.task_results: List[TaskResult] = []

    def get_output_dir(self, agent: str, task: str) -> str:
        return os.path.join(self.config.output, agent, task)

    def get_analysis_dir(self, agent, task):
        return self.get_output_dir(agent, task) + '/analysis'

    def read_results(self, from_analysis=False):
        """If `from_analysis` is True read from the already processed analysis folder."""
        for assignment in self.config.assignments:
            try:
                self.read_assignment_results(assignment, from_analysis)
            except Exception as err:
                traceback.print_exc()

    def read_assignment_results(self, assignment, from_analysis):
        agent = assignment.agent
        task = assignment.task
        if from_analysis:
            runs_file = os.path.join(self.get_analysis_dir(agent, task), "runs.jsonl")
            result_file = os.path.join(self.get_analysis_dir(agent, task), "overall.json")
        else:
            runs_file = os.path.join(self.get_output_dir(agent, task), "runs.jsonl")
            result_file = os.path.join(self.get_output_dir(agent, task), "overall.json")
        task_result = self.load_task_overall_result(assignment, result_file)
        self.load_task_runs(task_result, runs_file)
        self.task_results.append(task_result)

    def process_results(self):
        if not self.task_results:
            self.read_results()

        for task_result in self.task_results:
            self.analyse_task_result(task_result, task_result.assignment.agent, task_result.assignment.task)

    def load_task_overall_result(self, assignment: Assignment, result_file: str) -> TaskResult:
        with open(result_file, encoding="utf-8") as f:
            overall_result = json.load(f)

        task_def = self.config.definition.task[assignment.task]
        return TaskResult(assignment=assignment, overall=overall_result, task_def=task_def, task_client_outputs=[])

    @classmethod
    def load_task_runs(cls, task_result: TaskResult, runs_file: str):
        if not os.path.exists(runs_file):
            raise ValueError('Missing runs file')

        try:
            cls.load_normal_json_outputs(task_result, runs_file)
        except ValueError:
            cls.load_human_readable_outputs(task_result, runs_file)

    @classmethod
    def load_normal_json_outputs(cls, task_result: TaskResult, runs_file: str):
        with open(runs_file, "r") as f:
            for line in f:
                run = json.loads(line)
                run = TaskClientOutput.parse_obj(run)
                assert isinstance(run.output, TaskOutput)
                task_result.task_client_outputs.append(run)

    @classmethod
    def load_human_readable_outputs(cls, task_result: TaskResult, runs_file: str):
        with open(runs_file, "r") as f:
            content = f.read()

        json_strings = content.strip().split('}\n{')
        json_strings[0] += '}'
        json_strings[1:-1] = ['{' + js + '}' for js in json_strings[1:-1]]
        json_strings[-1] = '{' + json_strings[-1]

        runs = [json.loads(js) for js in json_strings]

        for run in runs:
            run = TaskClientOutput.parse_obj(run)
            assert isinstance(run.output, TaskOutput)
            task_result.task_client_outputs.append(run)

    def analyse_task_result(self, task_result: TaskResult, agent: str, task: str):
        analysis_folder = self.get_analysis_dir(agent, task)
        os.makedirs(analysis_folder, exist_ok=True)
        clear_file(f'{analysis_folder}/runs.jsonl')
        clear_file(f'{analysis_folder}/overall.json')

        task_definition = self.config.definition.task[task]

        if task_definition.parameters.get('confidence', True):

            for task_client_output in task_result.task_client_outputs:
                self.process_task_output(task_client_output.output, task_definition)

            # confidence_analysis(task_result)
            scale_base = self.config.rh_config.get('scale_base') if self.config.rh_config else None
            analyse_result_and_save_figs(task_result, output_folder=analysis_folder, scale_base=scale_base)
        else:
            make_frame(task_result)

        self.save_task_result(task_result, analysis_folder)

    @staticmethod
    def get_end_of_prompt(task_definition) -> ChatHistoryItem:
        prompt = get_prompt_from_str(task_definition.parameters['prompt'])
        if task_definition.parameters['oneshot']:
            return ChatHistoryItem(**prompt_dict[prompt][PromptType.ONESHOT][-1])
        else:
            return ChatHistoryItem(**prompt_dict[prompt][PromptType.INSTRUCTION][-1])

    def process_task_output(self, task_output: TaskOutput, task_definition):
        intersperse_confidence = task_definition.parameters.get('intersperse_confidence', True)
        confidences, actions = [], []

        if intersperse_confidence or task_output.result.get('confidence'):
            if 'confidence' not in task_output.result:
                logging.warning(f"'confidence' not in {task_output.index} with {task_output.status}")
            else:
                confidences = [root['confidence'] for root in task_output.result['confidence'] if
                               'confidence' in root]
                actions = [root.get('action') for root in task_output.result['confidence']]
        else:
            # The confidence is written into the agents response
            before, after, original_task = split_history_on_prompt(task_output.history, task_definition.parameters)
            for history_item in after:
                if history_item.role != "agent":
                    continue

                parsed = OSInteraction.extract_action(history_item.content)
                actions.append(parsed['action'])
                confidences.append(parsed['confidence'])
                if parsed['confidence'] is None:
                    print(f'No confidence given for task: {task_output.index}')

        task_output.result.update(
            {
                "last_confidence": next((c for c in confidences[::-1] if c is not None), None),
                "first_confidence": next((c for c in confidences if c is not None), None),
                "last_action": next((c for c in actions[::-1] if c is not None), None),
                "confidences": [c for c in confidences if c is not None],
                "actions": actions
            }
        )

    def save(self, folder: str):
        self.save_config(self.config.dict(), folder)
        for task_result in self.task_results:
            path = os.path.join(folder, task_result.assignment.agent, task_result.assignment.task)
            self.save_task_result(task_result, path)

    def save_task_result(self, task_result: TaskResult, folder: str):
        os.makedirs(folder, exist_ok=True)
        self.save_overall(task_result.overall, folder)
        for tco in task_result.task_client_outputs:
            self.save_task_client_output(tco, folder)

    @staticmethod
    def save_config(config: dict, folder: str):
        os.makedirs(folder, exist_ok=True)
        with open(f'{folder}/config.yaml', "a+", encoding="utf-8") as f:
            f.write(yaml.dump(config))

    @staticmethod
    def save_overall(overall: dict, folder: str):
        os.makedirs(folder, exist_ok=True)
        with open(f'{folder}/overall.json', "a+", encoding="utf-8") as f:
            f.write(json.dumps(overall))

    @staticmethod
    def save_task_client_output(task_client_output: TaskClientOutput, folder: str, error=False):
        os.makedirs(folder, exist_ok=True)
        with open(f'{folder}/{"error" if error else "runs"}.jsonl', "a+", encoding="utf-8") as f:
            f.write(json.dumps(task_client_output.dict()) + "\n")


class TaskHandler(ABC):
    def __init__(self, parameters):
        self.parameters = parameters

    @abstractmethod
    def matches(self, item, before):
        pass

    @abstractmethod
    def extract_task(self, item):
        pass


class AlfworldHandler(TaskHandler):
    def matches(self, item, before):
        return item.content.startswith('Here is your task')

    def extract_task(self, item):
        return item.content.split('Your task is to:')[-1]


class OSHandler(TaskHandler):
    def __init__(self, parameters):
        super().__init__(parameters)
        self.preamble = "Now, I will start a new problem in a new OS. My problem is:" if parameters.get(
            'oneshot', True
        ) else 'Now, my problem is:'

    def matches(self, item, before):
        return item.content.startswith(self.preamble)

    def extract_task(self, item):
        return item.content.lstrip(self.preamble)


class KGHandler(TaskHandler):
    def matches(self, item, before):
        return item.content.startswith('A new question: ')

    def extract_task(self, item):
        return item.content.split("A new question: ")[-1]


class DBHandler(TaskHandler):
    def matches(self, item, before):
        return len(before) == 2

    def extract_task(self, item):
        return item.content


TASK_HANDLER_MAP = {
    'alfworld': AlfworldHandler,
    'os': OSHandler,
    'kg': KGHandler,
    'db': DBHandler
}


def get_handler(task_name, parameters):
    for key in TASK_HANDLER_MAP:
        if task_name.startswith(key):
            return TASK_HANDLER_MAP[key](parameters)
    raise ValueError(f'Unrecognized task {task_name}')


def split_history_on_prompt(history: list, parameters: dict):
    before, after = [], []
    original_task = None
    beyond_example = False
    if not history:
        return before, after, original_task

    task = parameters['name']

    handler = get_handler(task, parameters)

    for item in history:
        if not beyond_example:
            if handler.matches(item, before):
                beyond_example = True
                original_task = handler.extract_task(item)
                after.append(item)
            else:
                before.append(item)
        else:
            after.append(item)

    if not after:
        raise RuntimeError(
            "Not getting beyond the prompt, are you matching one shot and prompt correctly?"
        )

    if not original_task:
        raise RuntimeError(
            'Unable to find the original task'
        )

    return before, after, original_task


def make_frame(res: TaskResult):
    rows = []
    for output in res.task_outputs:
        row = {"index": output.index}
        row.update(output.result)
        rows.append(row)

    df = pd.DataFrame(rows)
    return df


def confidence_analysis(res: TaskResult, output_folder: str = None):
    confidence = res.overall['custom']['overall'].setdefault('confidence', {})

    for task_output in res.task_outputs:
        confidences = np.array(task_output.result["confidences"])
        confidences = confidences[confidences != None].astype(float)
        if len(confidences) == 0:
            print(f"No confidences for {task_output.index}")
            continue
        task_output.result["first_confidence"] = confidences[0]
        task_output.result["last_confidence"] = confidences[-1]
        task_output.result["conf_std"] = confidences.std()
        task_output.result["conf_min"] = confidences.min()
        task_output.result["conf_max"] = confidences.max()
        task_output.result["conf_mean"] = confidences.mean()
        task_output.result["largest_step_up"] = (confidences[1:] - confidences[:-1]).max()
        task_output.result["largest_step_down"] = (confidences[1:] - confidences[:-1]).min()

    rows = []
    for output in res.task_outputs:
        row = {"index": output.index}
        row.update(output.result)
        rows.append(row)


def analyze_confidence(conf_type, confidences, res: TaskResult, output_folder: str, temperature: float = None,
                       show=SHOW):
    conf_type_short = conf_type.split('_')[0]
    success_with_conf = np.array(confidences)
    removed_nulls = success_with_conf[success_with_conf[:, -1] != None].astype(float)

    removed_nulls[:, 1] = removed_nulls[:, 1] / 100
    n_nulls = len(success_with_conf) - len(removed_nulls)
    if len(removed_nulls) == 0:
        return
    preds_np = removed_nulls[:, 1]
    targets_np = removed_nulls[:, 0]

    n_success_and_null = len(
        success_with_conf[(success_with_conf[:, 1] == None) & (success_with_conf[:, 0] == True)]
    )

    # 1. Save the temperature with all runs so can be used by other runs if needed
    temperature_fit = calculate_temperature_scale(preds_np, targets_np)

    # 2. For the logits run on os-std we take the temperature from the logits run on os-dev (one for each conf_type)
    if temperature is not None:
        preds_np = scale_probs_with_temp(preds_np, temperature)

    n_bins = 10
    ece, bin_centers, bin_acc, bin_probs, bin_counts = calculate_ece(
        preds_np, targets_np, n_bins=n_bins
    )
    ece_rms, _, _, _, _ = calculate_ece(
        preds_np, targets_np, n_bins=n_bins, rms=True
    )

    preds_t = Tensor(preds_np)
    target_t = Tensor(targets_np).to(torch.int)

    plot_calibration_curve(
        ece, bin_centers, bin_acc, bin_probs, bin_counts, labels=conf_type,
        title=f'Confidence - {conf_type_short}', save_file=f'{output_folder}/calibration_{conf_type_short}.pdf',
        show=show
    )

    brier_score = np.mean((preds_np - targets_np) ** 2)

    path = f'{output_folder}/roc_metric_{conf_type}.pdf'
    plot_roc_curve(preds_t.unsqueeze(0), target_t.unsqueeze(0), labels=[''], path=path, show=SHOW)

    b_auroc = BinaryAUROC(thresholds=None)
    auroc = b_auroc(preds_t, target_t)

    # Update summary dictionary
    # confidence = res.overall['custom']['overall'].setdefault(f'confidence_{conf_type}', {})
    confidence = res.overall['custom']['overall']
    confidence[f'{conf_type}_auroc'] = float(auroc)
    confidence[f'{conf_type}_ece'] = float(ece)
    confidence[f'{conf_type}_ece_rms'] = float(ece_rms)
    confidence[f'{conf_type}_brier'] = float(brier_score)
    confidence[f'{conf_type}_nulls'] = int(n_nulls)
    confidence[f'{conf_type}_n_success_and_null'] = int(n_success_and_null)
    confidence[f'{conf_type}_avg_conf'] = float(np.mean(preds_np))

    confidence[f'{conf_type}_success_avg_conf'] = float(np.mean(preds_np[targets_np == True]))
    confidence[f'{conf_type}_success_std_conf'] = float(np.std(preds_np[targets_np == True]))
    confidence[f'{conf_type}_fail_avg_conf'] = float(np.mean(preds_np[targets_np == False]))
    confidence[f'{conf_type}_fail_std_conf'] = float(np.std(preds_np[targets_np == False]))
    confidence[f'{conf_type}_temperature_fit'] = float(temperature_fit)
    confidence[f'{conf_type}_temperature_used'] = temperature if temperature is not None else 1.0

    return ece, bin_centers, bin_acc, bin_probs, bin_counts


def analyse_result_and_save_figs(res: TaskResult, output_folder: str, scale_base: str, show=SHOW):
    # Confidences over time
    data = [(to.index, to.result['result'], to.result['confidences']) for to in res.task_outputs]

    before_act, after_act = [], []
    for id, success, probs in data:
        if len(probs) > 4:
            before_act.append(np.mean(probs[:-2:2]))
            after_act.append(np.mean(probs[1:-1:2]))

    print(f'Percent difference after acting {np.nanmean(after_act) - np.nanmean(before_act):.2f}')

    plot_confidence_against_iterations(data, output_folder, show=show)
    plot_confidence_against_iterations(
        [(id, success, probs) for id, success, probs in data if success],
        output_folder, show=show
    )
    plot_confidence_against_iterations(
        [(id, success, probs) for id, success, probs in data if not success],
        output_folder, show=show
    )

    confs = {
        'first_confidence': ('first_confidence', 'Confidence (first)'),
        'confidences': ('avg_confidence', 'Confidence (avg)'),
        'last_confidence': ('last_confidence', 'Confidence (last)'),
    }

    # TODO - this would be much better with dataframes or numpy arrays rather than this horrible list method
    # but oh well
    ece_l, bin_centers_l, bin_acc_l, bin_probs_l, bin_counts_l = [], [], [], [], []

    temperature_lkup = {}
    if scale_base is not None:
        # Load the scale base temperatures into a dictionary to look up
        loader = ConfigLoader()
        config_ = loader.load_from("outputs/" + scale_base + "/config.yaml")
        value = AssignmentConfig.parse_obj(config_)
        value = AssignmentConfig.post_validate(value)

        parser = ResultsParser(value)
        parser.read_results(from_analysis=True)

        # Need to match it up to this particular assignment e.g. same agent but different task
        # (e.g. gpt-4 os-std uses gpt-4 os-dev) as base and vice-versa
        overall = next(
            tr.overall for tr in parser.task_results if
            tr.assignment.agent == res.assignment.agent and tr.assignment.task != res.assignment.task
        )

        for conf_key, (conf_name, conf_label) in confs.items():
            temperature_lkup[conf_name] = overall['custom']['overall'][f'{conf_name}_temperature_fit']

    # Again, would be much better with numpy arrays everywhere
    confidences_data_l = []
    for conf_key, (conf_name, conf_label) in confs.items():
        confidence_data = []
        for task_output in res.task_outputs:
            confidence_data.append((task_output.result["result"], np.mean(task_output.result[conf_key])))

        temperature = temperature_lkup.get(conf_name)
        if scale_base is not None and temperature is None:
            raise ValueError(f'Cannot find temperature for {conf_name} from {scale_base}')

        ece, bin_centers, bin_acc, bin_probs, bin_counts = analyze_confidence(
            conf_name, confidence_data, res, output_folder, temperature, show
        )
        ece_l.append(ece)
        bin_centers_l.append(bin_centers)
        bin_acc_l.append(bin_acc)
        bin_probs_l.append(bin_probs)
        bin_counts_l.append(bin_counts)
        confidences_data_l.append((confidence_data, temperature))

    # Overall benchmark score
    confidence = res.overall['custom']['overall']
    confs_in_overall = ['first_confidence', 'last_confidence']

    confidence['overall_benchmark_score'] = calculate_overall_benchmark_score(
        eces=[confidence[f'{conf_type}_ece'] for conf_type in confs_in_overall],
        aurocs=[confidence[f'{conf_type}_auroc'] for conf_type in confs_in_overall],
    )

    # Combined calibration curve
    plot_calibration_curve(
        ece_l, bin_centers_l, bin_acc_l, bin_probs_l, bin_counts_l, labels=[v[1] for v in confs.values()],
        title=f'Confidence calibration', save_file=f'{output_folder}/calibration_all.pdf', show=show
    )

    # Log version of the chart to show differences (less relevant with scaled probabilities)
    # plot_calibration_curve(
    #     ece_l, bin_centers_l, bin_acc_l, bin_probs_l, bin_counts_l, labels=list(confs.values()),
    #     title=f'Confidence calibration (log)', save_file=f'{output_folder}/calibration_all_log.pdf', show=show, log=True
    # )

    # Combined ROC
    targets, preds = [], []
    for confidences_data, temperature in confidences_data_l:
        confidences_data_np = np.array(confidences_data)

        preds_np = confidences_data_np[:, 1] / 100
        if temperature is not None:
            preds_np = scale_probs_with_temp(preds_np, temperature)
        targets_np = confidences_data_np[:, 0]
        targets.append(targets_np)
        preds.append(preds_np)

    preds_t = Tensor(np.stack(preds))
    target_t = Tensor(np.stack(targets)).to(torch.int)
    labels = [conf_label for (conf_name, conf_label) in confs.values()]
    path = f'{output_folder}/roc_metric_all.pdf'

    plot_roc_curve(preds_t, target_t, labels, path, show)


def main(outputs_to_parse: list[str]):
    fail = {}
    for name in outputs_to_parse:
        print(name)
        output = "outputs/" + name
        loader = ConfigLoader()
        try:
            config_ = loader.load_from(output + "/config.yaml")
            value = AssignmentConfig.parse_obj(config_)
            value = AssignmentConfig.post_validate(value)

            parser = ResultsParser(value)
            parser.process_results()
        except FileNotFoundError as err:
            fail[name] = err
            print(f'Error: {err}, {traceback.format_exc()}')

    print(f"Succeeded for {len(outputs_to_parse) - len(fail)}/{len(outputs_to_parse)}")
    if fail:
        print(fail)


if __name__ == "__main__":
    all_paths = os.listdir("outputs")
    all_paths = [path for path in all_paths if not (path.startswith('.') or path.startswith('ARCHIVE'))]
    outputs_to_parse = sorted(all_paths)[-1:]

    example_conf_on_broken = "2024-07-24-17-51-06"

    all_original_os = "2024-08-05-15-10-54"
    final_fixed = "2024-08-25-21-34-31"

    conf_final_fixed = "2024-08-27-11-41-48"
    conf_final_logits = "2024-08-28-15-49-29"
    conf_final_fixed_gtp35_base = "2024-08-28-19-22-48"
    conf_final_fixed_gtp4_turbo_base = "2024-08-29-12-11-11"
    conf_final_logits_dev = "2024-08-31-12-28-49"

    conf_final_fixed_scaled = "2024-08-27-11-41-48_scaled"
    conf_final_logits_scaled = "2024-08-28-15-49-29_scaled"

    outputs_to_parse = [
        conf_final_fixed,  # main verbalised run
        conf_final_fixed_scaled,  # main verbalised run scaled
        conf_final_fixed_gtp35_base, conf_final_fixed_gtp4_turbo_base,  # verbalised runs with different base agent
        conf_final_logits,  # main logits run
        conf_final_logits_dev, conf_final_logits_scaled,  # logits run scaled
        example_conf_on_broken  # For the example with BrokenBench
    ]

    main(outputs_to_parse)
