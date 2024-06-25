import asyncio
import glob
import json
import os
import re
import socket
import struct
import time
from typing import List, Dict, Any, Tuple

import docker
import docker.models.containers

from server.tasks.os_interaction.prompts import (
    PromptEnum,
    prompt_dict,
    PromptType,
    get_prompt_from_str,
    INTERSPERSED_CONFIDENCE_REQUEST,
    RETROSPECTIVE_CONFIDENCE_REQUEST_REMINDER,
    PLACEHOLDER,
)
from src.server.task import Task, Session
from src.typings import (
    AgentOutputStatus,
    TaskOutput,
    TaskSampleExecutionResult,
    SampleStatus,
)
from utils.confidence import extract_confidence, PERCENTAGE


class Container:
    def __init__(self, image):
        self.image = image
        self.client = docker.from_env()
        self.container: docker.models.containers.Container = self.client.containers.run(
            image,
            detach=True,
            tty=True,
            stdin_open=True,
            remove=True,
            labels={"created_by": "os-pipeline"},
        )
        self.exec_id = self.client.api.exec_create(
            self.container.id, "bash --login", stdin=True, tty=True
        )["Id"]
        self.sock = self.client.api.exec_start(self.exec_id, socket=True)._sock
        self.sock.settimeout(5)
        # clear buffer
        self.sock.recv(1000)

    def __del__(self):
        try:
            self.container.stop()
        except:
            pass

    def execute(self, command: str, original_terminal_cleanup=True):
        class DummyOutput:
            output: bytes
            exit_code: int

            def __init__(self, code, o):
                self.output = o
                self.exit_code = code

        # print("=== EXECUTING ===\n", command)
        if not isinstance(command, str):
            return DummyOutput(-1, b"")
        self.sock.send(command.encode("utf-8") + b"\n")
        # ignore input line
        data = self.sock.recv(8)
        _, n = struct.unpack(">BxxxL", data)
        _ = self.sock.recv(n)

        # Added timelimit
        time_limit = 20  # seconds
        start_time = time.time()

        output = b""
        while True:
            if time.time() - start_time > time_limit:
                print(f"Time limit reached, breaking out of the loop. Command was: `{command}`")
                break
            try:
                data = self.sock.recv(8)
                # print(data)
                if not data:
                    break
                _, n = struct.unpack(">BxxxL", data)
                line = self.sock.recv(n)
                output += line
                if original_terminal_cleanup and re.search(b"\x1b.+@.+[#|$] ", line):
                    break
            except TimeoutError:
                break
            except socket.timeout:
                break

        if not original_terminal_cleanup:
            # Clean up the output by removing terminal control sequences, removes escape sequences starting with
            # ESC (0x1b), followed by...
            # ... any characters, an '@' character, any characters, ending with '#' or '$'
            clean_output = re.sub(b"\x1b.+@.+[#|$] ", b'', output)
            # ... '[' and any combination of digits and semicolons, ending with a letter (a-z or A-Z)
            clean_output = re.sub(b'\x1b\\[[0-9;]*[a-zA-Z]', b'', clean_output)
            # ... ']' and any digits, a semicolon, any characters except BEL (0x07), and ending with BEL
            clean_output = re.sub(b'\x1b\\][0-9]*;[^\x07]*\x07', b'', clean_output)
            # ... '[?2004' and either 'h' or 'l'
            clean_output = re.sub(b'\x1b\[\?2004[hl]', b'', clean_output)

            # Remove BEL characters (0x07)
            # clean_output = re.sub(b'\x07', b'', clean_output)
            output = clean_output

        return DummyOutput(0, output)

    def execute_independent(self, command, *params):
        # print("=== EXECUTING INDEPENDENT ===\n", command)
        language, command = command
        # if params:
        #     print("== Parameters ==\n", params)
        if language == "bash":
            # Modify the command to always reload .bashrc (if it hasn't already). The alternative is to change all the
            # relevant tasks so that they do this
            if 'source ~/.bashrc' not in command:
                command = f"source ~/.bashrc && {command}"
            cmd = ["bash", "-c", command]
            if params:
                cmd.append("--")
                cmd.extend(params)
        elif language == "python":
            cmd = ["python3", "-c", command, *params]
        elif language == "c++":
            self.execute_independent(
                (
                    "bash",
                    f'echo "{json.dumps(command)}" > /tmp/main.cpp && '
                    f"g++ -o /tmp/a.out /tmp/main.cpp",
                ),
                None,
            )
            cmd = ["/tmp/a.out", *params]
        elif language == "c":
            self.execute_independent(
                (
                    "bash",
                    f'echo "{json.dumps(command)}" > /tmp/main.cpp && '
                    f"gcc -o /tmp/a.out /tmp/main.cpp",
                ),
                None,
            )
            cmd = ["/tmp/a.out", *params]
        else:
            raise ValueError("Unsupported language")
        return self.container.exec_run(cmd)


class JudgeConfig:
    image: str = None
    init_script: List[Tuple[str, str]] = None
    start: Tuple[str, str] = None
    description: str
    check: list = None
    match: dict = None
    example_script: str = None

    def get_evaluation_type(self):
        if self.check:
            return "check"
        elif self.match:
            return "match"

    def get_evaluation_content(self):
        return self.check or self.match



class OSInteraction(Task):
    def _load_configs(self, config_path, script_root_dir=".") -> List[JudgeConfig]:
        def load_script(script_obj):
            if script_obj is None:
                return None
            if type(script_obj) is str:
                return "bash", script_obj
            if "language" not in script_obj:
                language = "bash"
            else:
                language = script_obj["language"]
            if "file" in script_obj:
                with open(
                    os.path.join(script_root_dir, script_obj["file"]), encoding="utf-8"
                ) as f:
                    return language, f.read()
            elif "code" in script_obj:
                return language, script_obj["code"]
            else:
                raise ValueError("Invalid Script Object")

        # 1. handle input file:
        if config_path.endswith(".json"):
            with open(config_path, encoding="utf-8") as f:
                config_raw = json.load(f)
            if isinstance(config_raw, list):
                pass
            elif isinstance(config_raw, dict):
                config_raw = [config_raw]
            else:
                raise ValueError("Invalid Config File")
        elif config_path.endswith(".jsonl"):
            with open(config_path, encoding="utf-8") as f:
                config_raw = [json.loads(line) for line in f.readlines()]
        else:
            raise ValueError("Invalid Config File")

        # 2. handle configs
        configs: list[JudgeConfig] = []
        for item in config_raw:
            config = JudgeConfig()
            config.description = item["description"]
            if "create" in item:
                config.image = (
                    item["create"]["image"]
                    if ("image" in item["create"])
                    else (self.docker_config["localhost"] + "/default")
                )
                if "init" in item["create"]:
                    if type(item["create"]["init"]) is not list:
                        config.init_script = [load_script(item["create"]["init"])]
                    else:
                        config.init_script = [
                            load_script(script_obj)
                            for script_obj in item["create"]["init"]
                        ]
                else:
                    config.init_script = []
            else:
                config.image = self.docker_config["localhost"] + "/default"
            if "start" in item:
                config.start = load_script(item["start"])
            evaluation = item["evaluation"]
            if "match" in evaluation:
                if type(evaluation["match"]) is str:
                    config.match = {"answer": evaluation["match"], "strip": True}
                else:
                    config.match = evaluation["match"]
            elif "check" in evaluation:
                if type(evaluation["check"]) is not list:
                    config.check = [load_script(evaluation["check"])]
                else:
                    config.check = [
                        load_script(script_obj) for script_obj in evaluation["check"]
                    ]
            else:
                raise ValueError("check or match must exist.")
            if "check" in evaluation and "example" in evaluation:
                config.example_script = load_script(evaluation["example"])
            configs.append(config)
        return configs

    def __init__(self, data_config, docker_config, round_limit=8, **kwargs):
        super().__init__(**kwargs)
        self.round_limit: int = round_limit

        # --- Custom configs:
        # kwargs here contains the config.yaml task parameters (e.g. task->os-dev->parameters)
        self.prompt_choice: PromptEnum = PromptEnum.ORIGINAL
        if 'prompt' in kwargs:
            self.prompt_choice = get_prompt_from_str(kwargs['prompt'])

        self.one_shot: bool = kwargs.pop('oneshot', True)
        self.warn_remaining_tries: bool = kwargs.pop('warn_remaining_tries', False)
        self.original_terminal_cleanup: bool = kwargs.pop('original_terminal_cleanup', True)
        self.intersperse_confidence: bool = kwargs.pop('intersperse_confidence', False)
        self.remove_intersperse: bool = kwargs.pop('remove_intersperse', False)
        # ------

        self.data_config = data_config
        self.docker_config = docker_config
        self.problem_configs: Dict[str, Dict[str, Any]] = {}  # {index: CONFIG}

        matches = []
        for item in self.data_config["files"]:
            path = item["problem_file"]
            for file in glob.glob(path):
                if file.endswith(".json") or file.endswith(".jsonl"):
                    matches.append(
                        {
                            "problem_file": file,
                            "script_dir": item["script_dir"],
                            "index_prefix": item["index_prefix"]
                            + os.path.basename(file)
                            .removesuffix(".json")
                            .removesuffix(".jsonl")
                            + "-",
                        }
                    )
        self.data_config["files"] = matches

        for item in self.data_config["files"]:
            problem_file = item["problem_file"]
            single_file_configs = self._load_configs(problem_file, item["script_dir"])
            dict_configs = {}
            for idx, config in enumerate(single_file_configs):
                dict_configs[item["index_prefix"] + "%05d" % idx] = {
                    "file": problem_file,
                    "config": config,
                    "index": idx,
                }
            self.problem_configs.update(dict_configs)

    def calculate_overall(self, results: List[TaskOutput]) -> Dict[str, Any]:
        overall = {
            "total": len([config for config in results if config]),
            "pass": len(
                [
                    config
                    for config in results
                    if (config and config.result and config.result.get("result", False))
                ]
            ),
        }
        overall["wrong"] = overall["total"] - overall["pass"]
        overall["acc"] = overall["pass"] / overall["total"] if overall["total"] else 0
        return {
            "overall": overall,
        }

    def get_indices(self) -> List[Any]:
        return list(self.problem_configs.keys())

    @staticmethod
    def extract_action(raw: str):
        think_pattern = r"Think:\s*(.+)"
        act_pattern = r"Act:\s*(.+)"

        think = re.findall(think_pattern, raw)
        act = re.findall(act_pattern, raw)

        ret = {"thought": "\n".join(think), "action": None, "content": None, "confidence": extract_confidence(raw)}

        # reversly iterate over the action list
        for action in act[::-1]:
            if action.lower().startswith("bash"):
                ret["action"] = "bash"
                break
            if action.lower().startswith("finish"):
                ret["action"] = "commit"
                break
            if action.lower().startswith("answer"):
                content = action[6:].strip()
                left_par_pos = content.find("(")
                right_par_pos = content.rfind(")")
                if left_par_pos == -1 or right_par_pos == -1:
                    continue
                content = content[left_par_pos + 1: right_par_pos]
                ret["action"] = "commit"
                ret["content"] = content
                break

        if ret["action"] == "bash":
            # extract from ```bash to ```
            content_pattern = r"```bash\n(.*?)\n```"
            content = re.findall(content_pattern, raw, re.DOTALL)
            content = "\n\n".join(content)
            ret["content"] = content

        return ret

    async def start_sample(self, index, session: Session) -> TaskSampleExecutionResult:
        data_item = self.problem_configs[index]
        config = data_item["config"]
        file = data_item["file"]
        index_in_file = data_item["index"]
        try:
            print("init container")
            container = Container(config.image)
            print("init container ok")
            print("start judge")
            result = await self._judge(session, config, container)
            result.result["file"] = file
            result.result["index_in_file"] = index_in_file
            print("finish judge")
            return result
        except Exception as _:
            print("err")
            import traceback

            return TaskSampleExecutionResult(
                status=SampleStatus.UNKNOWN,
                result={"result": False, "error": traceback.format_exc()},
            )
        finally:
            try:
                container.__del__()
            except:
                pass

    async def _judge(
        self, session: Session, config: JudgeConfig, container: Container
    ) -> TaskSampleExecutionResult:

        print("exec start")
        if config.init_script:
            for script in config.init_script:
                init = await asyncio.to_thread(container.execute_independent, script)
                if init.exit_code != 0:
                    return TaskSampleExecutionResult(
                        status=SampleStatus.UNKNOWN,
                        result={"result": False, "error": f'Init script {script} failed: {init}'}
                    )
        if config.start:
            start = await asyncio.to_thread(container.execute, config.start[1], self.original_terminal_cleanup)
            if start.exit_code != 0:
                return TaskSampleExecutionResult(
                    status=SampleStatus.UNKNOWN,
                    result={"result": False, "error": f'Start script {config.start} failed: {start}'}
                )
        print("exec start ok")

        session.inject(prompt_dict[self.prompt_choice][PromptType.INSTRUCTION])

        if not self.one_shot:
            session.history[-1].content += (
                "Now, my problem is:\n\n" + config.description
            )
        else:
            ONE_SHOT = prompt_dict[self.prompt_choice][PromptType.ONESHOT]
            session.history[-1].content += (
                "Now, my problem is:\n\n" + ONE_SHOT[0]["content"]
            )
            for item in ONE_SHOT[1:]:
                session.inject(item)
            session.inject(
                {
                    "role": "user",
                    "content": "Now, I will start a new problem in a new OS. My problem is:\n\n"
                    + config.description,
                }
            )

        conf = []
        for i in range(self.round_limit):
            root = await session.action()
            if root.status == AgentOutputStatus.AGENT_CONTEXT_LIMIT:
                return TaskSampleExecutionResult(
                    status=SampleStatus.AGENT_CONTEXT_LIMIT, result={"result": False, "confidence": conf}
                )
            if root.status != AgentOutputStatus.NORMAL:
                return TaskSampleExecutionResult(
                    status=SampleStatus.UNKNOWN, result={"result": False, "confidence": conf}
                )
            root_original = root  # keeping for debug as is overwritten here
            root = self.extract_action(root.content)
            if "action" not in root:
                return TaskSampleExecutionResult(
                    status=SampleStatus.AGENT_VALIDATION_FAILED,
                    result={"result": False, "confidence": conf},
                )
            if root["action"] not in ["bash", "commit"]:
                return TaskSampleExecutionResult(
                    status=SampleStatus.AGENT_INVALID_ACTION, result={"result": False, "confidence": conf}
                )

            action = root["action"]
            content = root["content"]

            if self.intersperse_confidence:
                # Ask for confidence after the action (bash or finish)
                request_for_confidence = INTERSPERSED_CONFIDENCE_REQUEST
                if action == "commit" or i + 1 == self.round_limit:
                    # request_for_confidence = RETROSPECTIVE_CONFIDENCE_REQUEST
                    request_for_confidence = RETROSPECTIVE_CONFIDENCE_REQUEST_REMINDER.replace(
                        str(PLACEHOLDER), config.description
                    )

                session.inject(
                    {
                        "role": "user",
                        "content": request_for_confidence
                    }
                )
                conf_root = await session.action()
                try:
                    root["confidence"] = extract_confidence(conf_root.content, [PERCENTAGE])
                except ValueError:
                    print(f'Unable to extract confidence from {conf_root.content}')
                conf.append(root)

                if self.remove_intersperse:
                    session.history = session.history[:-2]

            if action == "commit":
                answer = content
                break
            elif action == "bash":
                result = await asyncio.to_thread(container.execute, content, self.original_terminal_cleanup)
                result = result.output.decode("utf-8")
                if len(result) > 800:
                    result = (
                        result[:780] + "\n[truncated because the output is too long]"
                    )
                session.inject(
                    {
                        "role": "user",
                        "content": ("The output of the OS:\n\n" + result)
                        if result
                        else "The output of the OS is empty.",
                    }
                )
                if self.warn_remaining_tries:
                    session.history[-1].content += (
                        f"\n{self.round_limit - i - 1}/{self.round_limit} attempts to answer remaining."
                    )
        else:
            return TaskSampleExecutionResult(
                status=SampleStatus.TASK_LIMIT_REACHED,
                result={"result": False, "reason": "round limit", "confidence": conf},
            )

        if isinstance(answer, str) and config.match and config.match["strip"]:
            answer = answer.strip()

        jd = False

        if config.match:
            if "answer" in config.match:
                jd = answer == config.match["answer"]
            elif "regex" in config.match:
                jd = re.search(config.match["regex"], answer) is not None
        elif config.check:
            params = [str(answer)]
            for script in config.check:
                if script is None:
                    script = config.example_script
                response = await asyncio.to_thread(
                    container.execute_independent, script, *params
                )
                if response.exit_code != 0:
                    jd = False
                    break
                params.append(response.output.decode("utf-8"))
            else:
                jd = True
        else:
            return TaskSampleExecutionResult(
                status=SampleStatus.UNKNOWN, result={"result": False, "confidence": conf}
            )

        return TaskSampleExecutionResult(
            status=SampleStatus.COMPLETED, result={"result": jd, "confidence": conf}
        )
