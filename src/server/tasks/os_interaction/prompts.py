import os.path
import uuid
from copy import deepcopy
from enum import Enum


def read_prompt_file(name) -> str:
    current_dir = os.path.dirname(__file__)
    library = os.path.join(current_dir, "library")

    with open(os.path.join(library, name), 'r') as file:
        contents = file.read()
    return contents


class PromptEnum(Enum):
    ORIGINAL = "ORIGINAL"
    ORIGINAL_W_FRAMEWORK = "ORIGINAL_W_FRAMEWORK"
    BASH_ONE_LINE = "BASH_ONE_LINE"
    BASH_ONE_LINE_CONF = "BASH_ONE_LINE_CONF"

    CONFIDENCE_ON_RESULT = "CONFIDENCE_ON_RESULT"
    CONFIDENCE_LOWER_NUMBERS = "CONFIDENCE_LOWER_NUMBERS"
    CONFIDENCE_WARN_OVERCONFIDENT = "CONFIDENCE_WARN_OVERCONFIDENT"
    CONFIDENCE_WITH_TIME_WARNING = "CONFIDENCE_WITH_TIME_WARNING"
    ONE_SHOT_IN_INSTRUCTION_W_CONF = "ONE_SHOT_IN_INSTRUCTION_W_CONF"


class PromptType(Enum):
    INSTRUCTION = "instruction"
    ONESHOT = "one-shot"


ORIGINAL_INSTRUCTION = """You are an assistant that will act like a person, I will play the role of linux(ubuntu) operating system. Your goal is to implement the operations required by me or answer to the question proposed by me. For each of your turn, you should first think what you should do, and then take exact one of the three actions: "bash", "finish" or "answer". 

1. If you think you should execute some bash code, take bash action, and you should print like this:

Think: put your thought here.

Act: bash

```bash
# put your bash code here
```

2. If you think you have finished the task, take finish action, and you should print like this:

Think: put your thought here.

Act: finish

3. If you think you have got the answer to the question, take answer action, and you should print like this:

Think: put your thought here.

Act: answer(Your answer to the question should be put in this pair of parentheses)

If the output is too long, I will truncate it. The truncated output is not complete. You have to deal with the truncating problem by yourself. Attention, your bash code should not contain any input operation. Once again, you should take only exact one of the three actions in each turn.\n\n"""


ORIGINAL_INSTRUCTION_FRAMEWORK_EXPLAINED = read_prompt_file("instruction.txt")

ORIGINAL_ONESHOT = [
    {"role": "user", "content": 'tell me how many files are in the directory "/etc"?'},
    {
        "role": "agent",
        "content": """Think: To count the files in /etc, I need to print all the files in it.

Act: bash

```bash
ls /etc
```""",
    },
    {
        "role": "user",
        "content": "The output of the OS:\ncpi cron.hourly fuse.conf iproute2 lvm networkd-dispatcher protocols "
                   "selinux tmpfiles.d [truncated because the output is too long]",
    },
    {
        "role": "agent",
        "content": """Think: The output has been truncated because it is too long, so I need to count files by script directly.

Act: bash

```bash
ls -1 /etc | wc -l
```""",
    },
    {"role": "user", "content": "The output of the OS:\n220"},
    {
        "role": "agent",
        "content": """Think: Now I get the answer, it is 220.

Act: answer(220)""",
    },
]

CONF_ON_RESULT_INSTRUCTION = """You are an assistant that will act like a person, I will play the role of linux(ubuntu) operating system. Your goal is to implement the operations required by me or answer to the question proposed by me. For each of your turn, you should first think what you should do, and then take exact one of the three actions: "bash", "finish" or "answer". You should also express as a percentage your confidence that you can successfully solve the given task (or have solved it correctly if giving a "finish" or "answer" action).

1. If you think you should execute some bash code, take bash action, and you should print like this:

Think: put your thought here.

Act: bash

```bash
# put your bash code here
```

Confidence: confidence(Your confidence of success should be put in this pair of parentheses)


2. If you think you have finished the task, take finish action, give your percentage confidence that you have finished the task successfully, and you should print like this:

Think: put your thought here.

Act: finish

Confidence: confidence(Your confidence of success should be put in this pair of parentheses)

3. If you think you have got the answer to the question, take answer action, give your percentage confidence that your answer is correct, and you should print like this:

Think: put your thought here.

Act: answer(Your answer to the question should be put in this pair of parentheses)

Confidence: confidence(Your confidence of success should be put in this pair of parentheses)

If the output is too long, I will truncate it. The truncated output is not complete. You have to deal with the truncating problem by yourself. You MUST follow the exact formats given for the bash command, answer and confidence. Attention, your bash code should not contain any input operation. Once again, you should take only exact one of the three actions in each turn.\n\n"""

CONF_ON_RESULT_ONSHOT = [
    {"role": "user", "content": 'tell me how many files are in the directory "/etc"?'},
    {
        "role": "agent",
        "content": """Think: To count the files in /etc, I need to print all the files in it.

Act: bash

```bash
ls /etc
```

Confidence: confidence(90%)
""",
    },
    {
        "role": "user",
        "content": "The output of the OS:\ncpi cron.hourly fuse.conf iproute2 lvm networkd-dispatcher protocols " \
                   "selinux tmpfiles.d [truncated because the output is too long]",
    },
    {
        "role": "agent",
        "content": """Think: The output has been truncated because it is too long, so I need to count files by script directly.

Act: bash

```bash
ls -1 /etc | wc -l
```

Confidence: confidence(95%)""",
    },
    {"role": "user", "content": "The output of the OS:\n220"},
    {
        "role": "agent",
        "content": """Think: Now I get the answer, it is 220.

Act: answer(220)

Confidence: confidence(100%)""",
    },
]
prompt_dict = {
    PromptEnum.ORIGINAL: {
        PromptType.INSTRUCTION: [{
            "role": "user",
            "content": ORIGINAL_INSTRUCTION,
        }],
        PromptType.ONESHOT: ORIGINAL_ONESHOT,
    },
    PromptEnum.ORIGINAL_W_FRAMEWORK: {
        PromptType.INSTRUCTION: [{
            "role": "user",
            "content": ORIGINAL_INSTRUCTION_FRAMEWORK_EXPLAINED,
        }],
        PromptType.ONESHOT: ORIGINAL_ONESHOT,
    },
    PromptEnum.CONFIDENCE_ON_RESULT: {
        PromptType.INSTRUCTION: [{
            "role": "user",
            "content": CONF_ON_RESULT_INSTRUCTION,
        }],
        PromptType.ONESHOT: CONF_ON_RESULT_ONSHOT,
    },
    PromptEnum.CONFIDENCE_WARN_OVERCONFIDENT: {
        PromptType.INSTRUCTION: [{
            "role": "user",
            "content": CONF_ON_RESULT_INSTRUCTION + "Note that you are typically overconfident in your ability.",
        }],
        PromptType.ONESHOT: CONF_ON_RESULT_ONSHOT,
    },
    PromptEnum.BASH_ONE_LINE: {
        PromptType.INSTRUCTION: [{
            "role": "user",
            "content": ORIGINAL_INSTRUCTION.replace(
            "Attention, your bash code should not contain any input operation.",
            "Attention, your bash code should not contain any input operation and should be one line only."
        ),
        }],
        PromptType.ONESHOT: ORIGINAL_ONESHOT,
    },
    PromptEnum.BASH_ONE_LINE_CONF: {
        PromptType.INSTRUCTION: [{
            "role": "user",
            "content": CONF_ON_RESULT_INSTRUCTION.replace(
            "Attention, your bash code should not contain any input operation.",
            "Attention, your bash code should not contain any input operation and should be one line only."
        ),
        }],
        PromptType.ONESHOT: CONF_ON_RESULT_ONSHOT,
    }
}

# Same as CONFIDENCE_ON_RESULT but one shot has lower numbers
prompt_dict[PromptEnum.CONFIDENCE_LOWER_NUMBERS] = {
    PromptType.INSTRUCTION: deepcopy(prompt_dict[PromptEnum.CONFIDENCE_ON_RESULT][PromptType.INSTRUCTION]),
    PromptType.ONESHOT: [
    {"role": "user", "content": 'tell me how many files are in the directory "/etc"?'},
    {
        "role": "agent",
        "content": """Think: To count the files in /etc, I need to print all the files in it.

Act: bash

```bash
ls /etc
```

Confidence: confidence(50%)
""",
    },
    {
        "role": "user",
        "content": "The output of the OS:\ncpi cron.hourly fuse.conf iproute2 lvm networkd-dispatcher protocols "
                   "selinux tmpfiles.d [truncated because the output is too long]",
    },
    {
        "role": "agent",
        "content": """Think: The output has been truncated because it is too long, so I need to count files by script directly.

Act: bash

```bash
ls -1 /etc | wc -l
```

Confidence: confidence(70%)""",
    },
    {"role": "user", "content": "The output of the OS:\n220"},
    {
        "role": "agent",
        "content": """Think: Now I get the answer, it is 220.

Act: answer(220)

Confidence: confidence(90%)""",
    },
],
}


prompt_dict[PromptEnum.ONE_SHOT_IN_INSTRUCTION_W_CONF] = deepcopy(prompt_dict[PromptEnum.CONFIDENCE_ON_RESULT])
prompt_dict[PromptEnum.ONE_SHOT_IN_INSTRUCTION_W_CONF][PromptType.INSTRUCTION][0]['content'] += "\n\n"
prompt_dict[PromptEnum.ONE_SHOT_IN_INSTRUCTION_W_CONF][PromptType.INSTRUCTION][0]['content'] += """
Here is an example conversation:

user: `tell me how many files are in the directory "/etc"?`

agent: `Think: To count the files in /etc, I need to print all the files in it.

Act: bash

```bash
ls /etc
```

Confidence: confidence(XX%)`

Note that here XX is a placeholder and when you respond you must put a number expressing your confidence

user: `The output of the OS:\ncpi cron.hourly fuse.conf iproute2 lvm networkd-dispatcher protocols selinux tmpfiles.d [truncated because the output is too long]`

agent: `Think: The output has been truncated because it is too long, so I need to count files by script directly.

Act: bash

```bash
ls -1 /etc | wc -l
```

Confidence: confidence(XX%)`

Note that here XX is a placeholder and when you respond you must put a number expressing your confidence

user: `The output of the OS:\n220`

agent: `Think: Now I get the answer, it is 220.

Act: answer(220)

Confidence: confidence(XX%)`

Note that here XX is a placeholder and when you respond you must put a number expressing your confidence
"""

PLACEHOLDER = uuid.uuid4()
request_confidence_pre = "What is your confidence that you will be successful in solving the task correctly?"
request_confidence_post = "What is your confidence that you have solved the original task correctly?"

explain_confidence_before = read_prompt_file('explain_confidence_before.txt')
explain_confidence_after = read_prompt_file("explain_confidence3.txt")

# The number (95%) in the one shot is definitely influencing the model, so prefer a one-shot without this bias
# one_shot_confidence = read_prompt_file("one_shot_confidence.txt")
one_shot_confidence = read_prompt_file("one_shot_confidence_without_number.txt")

INTERSPERSED_CONFIDENCE_REQUEST = f"""{request_confidence_pre} {explain_confidence_before} {one_shot_confidence}"""
RETROSPECTIVE_CONFIDENCE_REQUEST = f"""{request_confidence_post} {explain_confidence_after} {one_shot_confidence}"""
RETROSPECTIVE_CONFIDENCE_REQUEST_REMINDER = f"""{request_confidence_post} {explain_confidence_after} As a reminder the original task was ```{PLACEHOLDER}``` 

{one_shot_confidence}"""

# Confidence requestions for logits
ask_yes_no = 'Please ignore all previous instructions about answer format, do not do any thinking or reasoning and instead answer with a single word, either Yes or No.'

INTERSPERSED_LOGITS_REQUEST = f"Do you believe you will be successful in solving the task correctly? {ask_yes_no}"
RETROSPECTIVE_LOGITS_REQUEST = f"Do you believe you have solved the task correctly? {ask_yes_no}"
RETROSPECTIVE_LOGITS_REQUEST_REMINDER = f"Do you believe you have solved the task correctly? {ask_yes_no} As a reminder the original task was ```{PLACEHOLDER}```"


def get_prompt_from_str(keyword):
    enum_lkup = {qwe.value: qwe for qwe in PromptEnum}
    if keyword not in enum_lkup:
        raise RuntimeError(f'Unrecognised prompt choice {keyword}')
    prompt = enum_lkup[keyword]
    return prompt
