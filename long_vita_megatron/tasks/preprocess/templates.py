# coding=utf-8
# Copyright (c) 2024, HUAWEI CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os
import re
import json
import logging
from pathlib import Path
from dataclasses import dataclass
from enum import Enum, unique
from typing import TYPE_CHECKING, Dict, List, Optional, Sequence, Tuple, Union

from .formatter import EmptyFormatter, FunctionFormatter, StringFormatter, ToolFormatter

if TYPE_CHECKING:
    from transformers import PreTrainedTokenizer

    from .formatter import SLOTS, Formatter

logger = logging.getLogger(__name__)

cur_file_dir = Path(__file__).absolute().parent

TEMPLATES_DIR = os.path.join(cur_file_dir.parent.parent.parent, "configs/finetune/templates.json")


@dataclass
class AlpacaTemplate:
    system_token = ""
    user_token = "### Instruction:"
    assistant_token = "### Response:"
    end_token = ""
    system = "Below is an instruction that describes a task, paired with an input that provides further context. " \
             "Write a response that appropriately completes the request. " \
             "Please note that you need to think through your response logically and step by step."


class Prompter(object):

    def __init__(self, template, verbose: bool = False):
        self._verbose = verbose
        self.template = template
        self.user_role = "user"
        self.assistant_role = "assistant"
    
    def generate_training_prompt(self, messages) -> str:
        prompt = self.template.system_token + "\n" + self.template.system + self.template.end_token + "\n"

        for message in messages:
            if message["role"] == self.user_role:
                prompt += self.template.user_token + "\n" + message["content"] + self.template.end_token + "\n"
            else:
                prompt += self.template.assistant_token + "\n" + message["content"] \
                + self.template.end_token + "\n"
        
        return prompt


@unique
class Role(str, Enum):
    USER = "user"
    ASSISTANT = "assistant"
    SYSTEM = "system"
    FUNCTION = "function"
    OBSERVATION = "observation"


def infer_max_len(source_len: int, target_len: int, max_len: int, reserved_label_len: int) -> Tuple[int, int]:
    if source_len + target_len == 0:
        max_target_len = 0
    else:
        max_target_len = int(max_len * (target_len / (source_len + target_len)))
    max_target_len = max(max_target_len, reserved_label_len)
    max_source_len = max_len - min(max_target_len, target_len)
    return max_source_len, max_target_len


@dataclass
class Template:
    format_user: "Formatter"
    format_assistant: "Formatter"
    format_system: "Formatter"
    format_function: "Formatter"
    format_observation: "Formatter"
    format_tools: "Formatter"
    format_separator: "Formatter"
    format_prefix: "Formatter"
    default_system: str
    stop_words: List[str]
    efficient_eos: bool
    replace_eos: bool
    force_system: bool


    def encode_oneturn(
        self,
        tokenizer: "PreTrainedTokenizer",
        messages: List[Dict[str, str]],
        system: Optional[str] = None,
        tools: Optional[str] = None,
        cutoff_len: int = 1_000_000,
        reserved_label_len: int = 1,
    ) -> Tuple[List[int], List[int]]:
        r"""
        Returns a single pair of token ids representing prompt and response respectively.
        """
        encoded_pairs = self._encode(tokenizer, messages, system, tools, cutoff_len, reserved_label_len)
        prompt_ids = []
        for query_ids, resp_ids in encoded_pairs[:-1]:
            prompt_ids += query_ids + resp_ids
        prompt_ids = prompt_ids + encoded_pairs[-1][0]
        answer_ids = encoded_pairs[-1][1]
        return prompt_ids, answer_ids


    def encode_multiturn(
        self,
        tokenizer: "PreTrainedTokenizer",
        messages: List[Dict[str, str]],
        system: Optional[str] = None,
        tools: Optional[str] = None,
        cutoff_len: int = 1_000_000,
        reserved_label_len: int = 1,
    ) -> Sequence[Tuple[List[int], List[int]]]:
        r"""
        Returns multiple pairs of token ids representing prompts and responses respectively.
        """
        return self._encode(tokenizer, messages, system, tools, cutoff_len, reserved_label_len)


    def _encode(
        self,
        tokenizer: "PreTrainedTokenizer",
        messages: List[Dict[str, str]],
        system: str,
        tools: str,
        cutoff_len: int,
        reserved_label_len: int,
    ) -> Sequence[Tuple[List[int], List[int]]]:
        r"""
        Encodes formatted inputs to pairs of token ids.
        Turn 0: prefix + system + query        resp
        Turn t: sep + query           resp
        """
        system = system or self.default_system
        encoded_messages = []
        for i, message in enumerate(messages):
            elements = []
            if i == 0:
                elements += self.format_prefix.apply()
                if system or tools:
                    tool_text = self.format_tools.apply(content=tools)[0] if tools else ""
                    elements += self.format_system.apply(content=(system + tool_text))
            elif i > 0 and i % 2 == 0:
                elements += self.format_separator.apply()

            if message["role"] == Role.USER.value:
                elements += self.format_user.apply(content=message["content"], idx=str(i // 2))
            elif message["role"] == Role.ASSISTANT.value:
                elements += self.format_assistant.apply(content=message["content"])
            elif message["role"] == Role.OBSERVATION.value:
                elements += self.format_observation.apply(content=message["content"])
            elif message["role"] == Role.FUNCTION.value:
                elements += self.format_function.apply(content=message["content"])
            else:
                raise NotImplementedError("Unexpected role: {}".format(message["role"]))
            print(f"elements {elements}")
            encoded_messages.append(self._convert_elements_to_ids(tokenizer, elements))

        return self._make_pairs(encoded_messages, cutoff_len, reserved_label_len)


    def _convert_elements_to_ids(
        self, tokenizer: "PreTrainedTokenizer", elements: List[Union[str, Dict[str, str]]]
    ) -> List[int]:
        r"""
        Converts elements to token ids.
        """
        token_ids = []
        for elem in elements:
            if isinstance(elem, str):
                if len(elem) != 0:
                    token_ids += tokenizer.encode(elem, add_special_tokens=False)
            elif isinstance(elem, dict):
                token_ids += [tokenizer.convert_tokens_to_ids(elem.get("token"))]
            elif isinstance(elem, set):
                if "bos_token" in elem and tokenizer.bos_token_id is not None:
                    token_ids += [tokenizer.bos_token_id]
                elif "eos_token" in elem and tokenizer.eos_token_id is not None:
                    token_ids += [tokenizer.eos_token_id]
            else:
                raise ValueError("Input must be string, set[str] or dict[str, str], got {}".format(type(elem)))

        return token_ids


    def _make_pairs(
        self,
        encoded_messages: Sequence[List[int]],
        cutoff_len: int,
        reserved_label_len: int,
    ) -> Sequence[Tuple[List[int], List[int]]]:
        encoded_pairs = []
        total_length = 0
        for i in range(0, len(encoded_messages), 2):
            if total_length >= cutoff_len:
                break

            max_source_len, max_target_len = infer_max_len(
                source_len=len(encoded_messages[i]),
                target_len=len(encoded_messages[i + 1]),
                max_len=(cutoff_len - total_length),
                reserved_label_len=reserved_label_len,
            )
            source_ids = encoded_messages[i][:max_source_len]
            target_ids = encoded_messages[i + 1][:max_target_len]
            total_length += len(source_ids) + len(target_ids)
            encoded_pairs.append((source_ids, target_ids))

        return encoded_pairs


@dataclass
class Llama2Template(Template):
    def _encode(
        self,
        tokenizer: "PreTrainedTokenizer",
        messages: List[Dict[str, str]],
        system: str,
        tools: str,
        cutoff_len: int,
        reserved_label_len: int,
    ) -> Sequence[Tuple[List[int], List[int]]]:
        r"""
        Encodes formatted inputs to pairs of token ids.
        Turn 0: system + query        resp
        Turn t: sep + query           resp
        """
        system = system or self.default_system
        encoded_messages = []
        for i, message in enumerate(messages):
            elements = []
            system_text = ""
            if i == 0:
                elements += self.format_prefix.apply()
                if system or tools:
                    tool_text = self.format_tools.apply(content=tools)[0] if tools else ""
                    system_text = self.format_system.apply(content=(system + tool_text))[0]
            elif i > 0 and i % 2 == 0:
                elements += self.format_separator.apply()

            if message["role"] == Role.USER.value:
                elements += self.format_user.apply(content=system_text + message["content"])
            elif message["role"] == Role.ASSISTANT.value:
                elements += self.format_assistant.apply(content=message["content"])
            elif message["role"] == Role.OBSERVATION.value:
                elements += self.format_observation.apply(content=message["content"])
            elif message["role"] == Role.FUNCTION.value:
                elements += self.format_function.apply(content=message["content"])
            else:
                raise NotImplementedError("Unexpected role: {}".format(message["role"]))

            encoded_messages.append(self._convert_elements_to_ids(tokenizer, elements))

        return self._make_pairs(encoded_messages, cutoff_len, reserved_label_len)


templates: Dict[str, Template] = {}


def get_templates() -> Dict[str, Template]:
    return templates


def get_model_template(name, prompt_type_path):
    name = register_custom_template(name, prompt_type_path)
    if name is None:
        template = templates["empty"]  # placeholder
    else:
        template = get_templates().get(name, None)
        if template is None:
            raise ValueError("Template {} does not exist.".format(name))
    return template


def fix_model_tokenizer(
    tokenizer: "PreTrainedTokenizer",
    name: Optional[str] = None,
    prompt_type_path: Optional[str] = None,
):
    template = get_model_template(name, prompt_type_path)

    stop_words = template.stop_words
    if template.replace_eos:
        if not stop_words:
            raise ValueError("Stop words are required to replace the EOS token.")

        _add_or_replace_eos_token(tokenizer, eos_token=stop_words[0])
        stop_words = stop_words[1:]

    if tokenizer.eos_token_id is None:
        _add_or_replace_eos_token(tokenizer, eos_token="<|endoftext|>")

    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
        logger.info("Add pad token: {}".format(tokenizer.pad_token))

    if stop_words:
        num_added_tokens = tokenizer.add_special_tokens(
            dict(additional_special_tokens=stop_words), replace_additional_special_tokens=False
        )
        logger.info("Add {} to stop words.".format(",".join(stop_words)))
        if num_added_tokens > 0:
            logger.warning("New tokens have been added, make sure `resize_vocab` is True.")

    try:
        tokenizer.chat_template = _get_jinja_template(template, tokenizer)
    except ValueError:
        logger.info("Cannot add this chat template to tokenizer.")
        

def _register_template(
    name: str,
    format_user: Optional["Formatter"] = None,
    format_assistant: Optional["Formatter"] = None,
    format_system: Optional["Formatter"] = None,
    format_function: Optional["Formatter"] = None,
    format_observation: Optional["Formatter"] = None,
    format_tools: Optional["Formatter"] = None,
    format_separator: Optional["Formatter"] = None,
    format_prefix: Optional["Formatter"] = None,
    default_system: str = "",
    stop_words: List[str] = [],
    efficient_eos: bool = False,
    replace_eos: bool = False,
    force_system: bool = False,
) -> None:
    r"""
    Registers a chat template.

    To add the following chat template:
    ```
    [HUMAN]:
    user prompt here
    [AI]:
    model response here

    [HUMAN]:
    user prompt here
    [AI]:
    model response here
    ```

    The corresponding code should be:
    ```
    _register_template(
        name="custom",
        format_user=StringFormatter(slots=["[HUMAN]:\n{{content}}\n[AI]:\n"]),
        format_separator=EmptyFormatter(slots=["\n\n"]),
        efficient_eos=True,
    )
    ```
    """
    eos_slots = [] if efficient_eos else [{"eos_token"}]
    template_class = Llama2Template if name.startswith("llama2") else Template
    default_user_formatter = StringFormatter(slots=["{{content}}"])
    default_assistant_formatter = StringFormatter(slots=["{{content}}"] + eos_slots)
    default_function_formatter = FunctionFormatter(slots=["Action: {{name}}\nAction Input: {{arguments}}"] + eos_slots)
    default_tool_formatter = ToolFormatter(tool_format="default")
    default_separator_formatter = EmptyFormatter()
    default_prefix_formatter = EmptyFormatter()
    templates[name] = template_class(
        format_user=format_user or default_user_formatter,
        format_assistant=format_assistant or default_assistant_formatter,
        format_system=format_system or default_user_formatter,
        format_function=format_function or default_function_formatter,
        format_observation=format_observation or format_user or default_user_formatter,
        format_tools=format_tools or default_tool_formatter,
        format_separator=format_separator or default_separator_formatter,
        format_prefix=format_prefix or default_prefix_formatter,
        default_system=default_system,
        stop_words=stop_words,
        efficient_eos=efficient_eos,
        replace_eos=replace_eos,
        force_system=force_system,
    )


def _add_or_replace_eos_token(tokenizer: "PreTrainedTokenizer", eos_token: str) -> None:
    is_added = tokenizer.eos_token_id is None
    num_added_tokens = tokenizer.add_special_tokens({"eos_token": eos_token})

    if is_added:
        logger.info("Add eos token: {}".format(tokenizer.eos_token))
    else:
        logger.info("Replace eos token: {}".format(tokenizer.eos_token))

    if num_added_tokens > 0:
        logger.warning("New tokens have been added, make sure `resize_vocab` is True.")


def _jinja_escape(content: str) -> str:
    return content.replace("\n", r"\n").replace("'", r"\'")


def _convert_slots_to_jinja(slots: "SLOTS", tokenizer: "PreTrainedTokenizer", placeholder: str = "content") -> str:
    slot_items = []
    for slot in slots:
        if isinstance(slot, str):
            slot_pieces = slot.split("{{content}}")
            if slot_pieces[0]:
                slot_items.append("'" + _jinja_escape(slot_pieces[0]) + "'")
            if len(slot_pieces) > 1:
                slot_items.append(placeholder)
                if slot_pieces[1]:
                    slot_items.append("'" + _jinja_escape(slot_pieces[1]) + "'")
        elif isinstance(slot, set):
            if "bos_token" in slot:
                slot_items.append("'" + tokenizer.bos_token + "'")
            elif "eos_token" in slot:  # do not use {{ eos_token }} since it may be replaced
                slot_items.append("'" + tokenizer.eos_token + "'")
        elif isinstance(slot, dict):
            raise ValueError("Dict is not supported.")

    return " + ".join(slot_items)


def _get_jinja_template(template: "Template", tokenizer: "PreTrainedTokenizer") -> str:
    jinja_template = ""

    prefix = _convert_slots_to_jinja(template.format_prefix.apply(), tokenizer)
    if prefix:
        jinja_template += "{{ " + prefix + " }}"

    if template.default_system:
        jinja_template += "{% set system_message = '" + _jinja_escape(template.default_system) + "' %}"

    jinja_template += (
        "{% if messages[0]['role'] == 'system' %}{% set system_message = messages[0]['content'] %}{% endif %}"
    )

    system_message = _convert_slots_to_jinja(template.format_system.apply(), tokenizer, placeholder="system_message")
    if isinstance(template, Llama2Template):
        pass
    elif template.force_system:
        jinja_template += "{{ " + system_message + " }}"
    else:
        jinja_template += "{% if system_message is defined %}{{ " + system_message + " }}{% endif %}"

    jinja_template += "{% for message in messages %}"
    jinja_template += "{% set content = message['content'] %}"
    if isinstance(template, Llama2Template):
        jinja_template += "{% if loop.index0 == 0 and system_message is defined %}"
        jinja_template += "{% set content = " + system_message + " + message['content'] %}"
        jinja_template += "{% endif %}"
    jinja_template += "{% if message['role'] == 'user' %}"
    user_message = _convert_slots_to_jinja(template.format_user.apply(), tokenizer)
    jinja_template += "{{ " + user_message + " }}"
    jinja_template += "{% elif message['role'] == 'assistant' %}"
    assistant_message = _convert_slots_to_jinja(
        template.format_assistant.apply() + template.format_separator.apply(), tokenizer
    )
    jinja_template += "{{ " + assistant_message + " }}"
    jinja_template += "{% endif %}"
    jinja_template += "{% endfor %}"
    return jinja_template


def register_custom_template(name, json_file_path=TEMPLATES_DIR) -> str:
    if name in templates:
        return name
    
    if not bool(re.match(r'(?:(?:/|\.{1,2}/|[^/\0]+/)(?:[^/\0]+/)*[^/\0]*|\.{1,2})', json_file_path)):
        raise ValueError(f"Invalid Path: {json_file_path}, please provide a valid custom template path.")
    
    with open(json_file_path, 'r') as file:
        config = json.load(file)

    templates_dict = {template['name']: template for template in config}
    config = templates_dict.get(name, None)

    if not config:
        raise ValueError(f"Can't find the template. Please provide a valid prompt type template in the {json_file_path}.")

    format_user = _format_custom_template(config.get("format_user", None))
    format_assistant = _format_custom_template(config.get("format_assistant", None))
    format_system = _format_custom_template(config.get("format_system", None))
    format_function = _format_custom_template(config.get("format_function", None))
    format_observation = _format_custom_template(config.get("format_observation", None))
    format_tools = _format_custom_template(config.get("format_tools", None))
    format_separator = _format_custom_template(config.get("format_separator", None))
    format_prefix = _format_custom_template(config.get("format_prefix", None))
    default_system = _format_custom_template(config.get("default_system", ""))
    stop_words = _format_custom_template(config.get("stop_words", []))
    efficient_eos = _format_custom_template(config.get("efficient_eos", False))
    replace_eos = _format_custom_template(config.get("replace_eos", False))
    force_system = _format_custom_template(config.get("force_system", False))

    if isinstance(default_system, list):
        default_system = "".join(default_system) if all(isinstance(sentence, str) for sentence in default_system) else default_system
    format_user = StringFormatter(**format_user) if format_user else None
    format_assistant = StringFormatter(**format_assistant) if format_assistant else None
    format_system = StringFormatter(**format_system) if format_system else None
    format_function = FunctionFormatter(**format_function) if format_function else None
    format_observation = StringFormatter(**format_observation) if format_observation else None
    format_tools = ToolFormatter(**format_tools) if format_tools else None
    format_separator = EmptyFormatter(**format_separator) if format_separator else None
    format_prefix = EmptyFormatter(**format_prefix) if format_prefix else None


    _register_template(
        name=name,
        format_user=format_user,
        format_assistant=format_assistant,
        format_system=format_system,
        format_function=format_function,
        format_observation=format_observation,
        format_tools=format_tools,
        format_separator=format_separator,
        format_prefix=format_prefix,
        default_system=default_system,
        stop_words=stop_words,
        efficient_eos=efficient_eos,
        replace_eos=replace_eos,
        force_system=force_system
    )

    return name


def _format_custom_template(slots: Dict) -> Dict:
    if slots and isinstance(slots, Dict):
        for key, slot in slots.items():
            slots[key] = list(map(lambda slot: set(slot) if isinstance(slot, list) else slot, slot)) if slot else None
    return slots
