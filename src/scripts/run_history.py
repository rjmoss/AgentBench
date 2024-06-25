"""
Script which takes an already run task and runs through the history and injects questions about the confidence
"""
import math
import os
import time
import traceback
from typing import Literal, Optional

from tqdm import tqdm

from configs import ConfigLoader
from scripts.parse_output import ResultsParser, split_history_on_prompt
from server.tasks.os_interaction.prompts import (
    INTERSPERSED_CONFIDENCE_REQUEST,
    RETROSPECTIVE_CONFIDENCE_REQUEST_REMINDER,
    PLACEHOLDER,
    RETROSPECTIVE_CONFIDENCE_REQUEST,
    INTERSPERSED_LOGITS_REQUEST, RETROSPECTIVE_LOGITS_REQUEST_REMINDER,
)
from typings import (
    ChatHistoryItem,
    AssignmentConfig,
    AgentOutput,
    AgentContextLimitException,
    AgentOutputStatus,
    get_predefined_structure, SampleStatus,
)
from utils.confidence import extract_confidence


class ConfidenceChatHistoryItem(ChatHistoryItem):
    role: Literal["user", "agent"]
    content: str
    confidence: Optional[float]
    reasoning: Optional[str]
    first_logprob: Optional[float]


def call_agent(history, agent) -> AgentOutput:
    try:
        content, first_logprob = agent.inference([h.dict() for h in history])
        response = AgentOutput(content=content, first_logprob=first_logprob)
    except AgentContextLimitException:
        response = AgentOutput(status=AgentOutputStatus.AGENT_CONTEXT_LIMIT)
    except Exception as e:
        if hasattr(agent, "model_name"):
            model_name = agent.model_name
        elif hasattr(agent, "name"):
            model_name = agent.name
        else:
            model_name = agent.__class__.__name__

        print(f"ERROR: {model_name} agent error", e)
        raise e
    return response


def replay_history(name, rh_config):
    remove_intersperse = rh_config.get('remove_ask_confidence', False)
    ask_confidence_after = rh_config.get('ask_confidence_after', 'agent')
    logits = rh_config.get('logits', False)

    now = get_predefined_structure()['TIMESTAMP']
    output = "outputs/" + name
    loader = ConfigLoader()
    config_ = loader.load_from(output + "/config.yaml")

    if logits:
        for agent_config in config_['definition']['agent'].values():
            agent_config['parameters']['body']['logprobs'] = True

    value = AssignmentConfig.parse_obj(config_)
    value = AssignmentConfig.post_validate(value)

    parser = ResultsParser(value)
    parser.read_results()

    save_dir = f"outputs/{now}"
    parser.config.output = save_dir
    for task_conf in parser.config.definition.task.values():
        task_conf.parameters['confidence'] = True

    save_config = parser.config.dict()
    rh_config['original_run'] = name
    save_config['rh_config'] = rh_config
    parser.save_config(save_config, save_dir)

    for tr in tqdm(parser.task_results, desc='task_results'):
        if rh_config.get('filter_agent', '') and rh_config['filter_agent'] != tr.assignment.agent:
            continue
        if rh_config.get('filter_task', '') and rh_config['filter_task'] != tr.assignment.task:
            continue

        if 'base_agent' in rh_config:
            tr_base = next(
                _tr
                for _tr in parser.task_results
                if _tr.assignment.agent == rh_config['base_agent'] and _tr.assignment.task == tr.assignment.task
            )
        else:
            tr_base = tr

        agent = value.definition.agent[tr.assignment.agent].create()
        # Here we are saving the 'overall' stats for the base task in the new folder which represent the agent
        # which is being run here
        folder = os.path.join(save_dir, tr.assignment.agent, tr.assignment.task)
        parser.save_overall(tr_base.overall, folder)

        for j, task_client_output in enumerate(tqdm(tr_base.task_client_outputs, desc='task_client_outputs')):
            task_output = task_client_output.output
            print(f'Index: {task_client_output.output.index}')
            try:
                process_row(
                    remove_intersperse, task_output, tr_base.task_def, agent, ask_confidence_after, logits=logits
                )
            except Exception:
                traceback.print_exc()
                parser.save_task_client_output(task_client_output, folder, error=True)
                time.sleep(5)  # Be conservative with rate limiter after exception
            else:
                parser.save_task_client_output(task_client_output, folder, error=False)


def process_row(remove_intersperse: bool, task_output, task_definition, agent, ask_confidence_after: str,
                logits: bool = False):
    # Need this so parse output knows to look for the confidences in `task_output.result['confidence']` rather
    # than trying to extract from the agent responses
    task_definition.parameters['intersperse_confidence'] = True
    before, after, original_task = split_history_on_prompt(task_output.history, task_definition.parameters)
    new_history = before

    confidence_list = []
    for i, item in enumerate(tqdm(after, desc='history')):
        item = ConfidenceChatHistoryItem(**item.dict())
        new_history.append(item)
        # Ask for confidence when giving the original task but after that only ask for confidence after making
        # an action
        if ask_confidence_after != 'both' and item.role != ask_confidence_after:
            continue

        request_for_confidence = INTERSPERSED_CONFIDENCE_REQUEST if not logits else INTERSPERSED_LOGITS_REQUEST
        if i + 1 == len(after) or (
                i + 2 == len(after) and task_output.status == SampleStatus.TASK_LIMIT_REACHED
        ):
            request_for_confidence = RETROSPECTIVE_CONFIDENCE_REQUEST if not logits else RETROSPECTIVE_CONFIDENCE_REQUEST
            if original_task is not None:
                request_for_confidence = (
                    RETROSPECTIVE_CONFIDENCE_REQUEST_REMINDER if not logits else RETROSPECTIVE_LOGITS_REQUEST_REMINDER
                )
                request_for_confidence = request_for_confidence.replace(str(PLACEHOLDER), original_task)

        new_history.append(ChatHistoryItem(role="user", content=request_for_confidence))
        response = call_agent(new_history, agent)
        time.sleep(1)  # to avoid hitting rate limiter
        new_history.append(ChatHistoryItem(role="agent", content=response.content))

        # This is the pattern expected in the output parser
        root = {}
        try:
            if logits:
                conf = 100 * math.exp(float(response.first_logprob))
                if response.content == 'Yes':
                    pass
                elif response.content == 'No':
                    conf = 100 - conf
                else:
                    raise ValueError()

                item.confidence = conf
            else:
                item.confidence = extract_confidence(response.content)
            root["confidence"] = item.confidence

        except ValueError:
            print(f'Unable to extract confidence from {response.content}')
        else:
            confidence_list.append(root)

        if remove_intersperse:
            # Remove the question about confidence, but keep the reasoning of the response in the relevant place
            item.reasoning = response.content
            new_history = new_history[:-2]

    task_output.result["confidence"] = confidence_list
    task_output.history = new_history


def main(outputs: list[str], args):
    loader = ConfigLoader()
    config = loader.load_from(args.config)

    fail = {}
    for name in tqdm(sorted(outputs), 'outputs'):
        print(name)
        try:
            replay_history(name, rh_config=config)
        except Exception as err:
            fail[name] = err
            print(f'Error: {err}, {traceback.format_exc()}')

    print(f"Succeeded for {len(outputs) - len(fail)}/{len(outputs)}")
    if fail:
        print(fail)


if __name__ == '__main__':
    import argparse

    arg_parser = argparse.ArgumentParser()

    arg_parser.add_argument(
        "--config", "-c", type=str, default="configs/run_history.yaml"
    )

    arg_parser.add_argument(
        "--auto-retry", "-r", action="store_true", dest="retry"
    )
    args = arg_parser.parse_args()

    all_original_os = "2024-08-05-15-10-54"
    all_fixed_os = "2024-08-25-21-34-31"

    runs = [all_fixed_os]

    main(runs, args)
