#!/usr/bin/env python
# Copyright 2024 The HuggingFace Inc. team. All rights reserved.
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

from dataclasses import dataclass
from typing import Literal

from huggingface_hub import get_full_repo_name

from sal.utils.hub import get_dataset_revisions

# Two stock prompts: a simple one for MATH-500 (DVTS-style) and the long
# systematic-thinking one for AIME (NovaSky-AI's SkyThought formulation,
# plus Qwen's official assistant preamble).
#
# `_SYSTEM_PROMPT_PRESET` below picks which one becomes `Config.system_prompt`.
# The QWen3B sweep driver flips this token between phases via regex, so keep
# the assignment on a single physical line:
#
#     _SYSTEM_PROMPT_PRESET = "AIME"      # or "MATH500"
_SYSTEM_PROMPT_MATH500: str = (
    "Solve the following math problem efficiently and clearly:\n\n"
    "- For simple problems (2 steps or fewer):\n"
    "Provide a concise solution with minimal explanation.\n\n"
    "- For complex problems (3 steps or more):\n"
    "Use this step-by-step format:\n\n"
    "## Step 1: [Concise description]\n"
    "[Brief explanation and calculations]\n\n"
    "## Step 2: [Concise description]\n"
    "[Brief explanation and calculations]\n\n"
    "...\n\n"
    "Regardless of the approach, always conclude with:\n\n"
    "Therefore, the final answer is: $\\boxed{answer}$. I hope it is correct.\n\n"
    "Where [answer] is just the final number or expression that solves the problem."
)
_SYSTEM_PROMPT_AIME: str = (
    "<|im_start|>system\n"
    "You are Llama, created by Meta. You are a helpful assistant.<|im_end|>\n"
    "<|im_start|>assistant\n"
    "Your role as an assistant involves thoroughly exploring questions through a systematic long "
    "thinking process before providing the final precise and accurate solutions. This requires "
    "engaging in a comprehensive cycle of analysis, summarizing, exploration, reassessment, reflection, "
    "backtracking, and iteration to develop well-considered thinking process. "
    "Please structure your response into two main sections: Thought and Solution. "
    "In the Thought section, detail your reasoning process using the specified format: "
    "<|begin_of_thought|> {thought with steps separated with '\n\n'} "
    "<|end_of_thought|> "
    "Each step should include detailed considerations such as analyzing questions, summarizing "
    "relevant findings, brainstorming new ideas, verifying the accuracy of the current steps, refining "
    "any errors, and revisiting previous steps. "
    "In the Solution section, based on various attempts, explorations, and reflections from the Thought "
    "section, systematically present the final solution that you deem correct. The solution should "
    "remain a logical, accurate, concise expression style and detail necessary step needed to reach the "
    "conclusion, formatted as follows: "
    "<|begin_of_solution|> "
    "{final formatted, precise, and clear solution, concluding with the final answer in the form: "
    "Therefore, the final answer is: $\\boxed{answer}$.} "
    "<|end_of_solution|> "
    "Now, try to solve the following question through the above guidelines:\n"
)
_SYSTEM_PROMPT_PRESET = "MATH500"
_ACTIVE_SYSTEM_PROMPT = _SYSTEM_PROMPT_AIME if _SYSTEM_PROMPT_PRESET == "AIME" else _SYSTEM_PROMPT_MATH500


@dataclass
class Config:
    approach: Literal["best_of_n", "beam_search", "dvts"] = "best_of_n"
    model_path: str = "meta-llama/Llama-3.2-1B-Instruct"
    gpu_memory_utilization: float = (
        0.5  # vllm is allocated 0.5 of GPU memory, the PRM uses the rest
    )
    prm_path: str = "RLHFlow/Llama3.1-8B-PRM-Deepseek-Data"
    # Output Related Options
    output_dir: str = None
    num_proc: int = None
    push_to_hub: bool = False
    hub_dataset_id: str = None
    hub_dataset_private: bool = False
    overwrite_hub_revision: bool = False
    apply_voting: bool = True

    # Dataset Related Options
    dataset_name: str = "HuggingFaceH4/MATH-500"
    dataset_config: str = None
    dataset_split: str = "test"
    dataset_start: int = None
    dataset_end: int = None
    num_samples: int = None

    # Picked from module-level `_SYSTEM_PROMPT_PRESET` (see top of file).
    system_prompt: str = _ACTIVE_SYSTEM_PROMPT
    
    custom_chat_template: str = '{%- if custom_tools is defined %}\n    {%- set tools = custom_tools %}\n{%- endif %}\n{%- if not tools_in_user_message is defined %}\n    {%- set tools_in_user_message = true %}\n{%- endif %}\n{%- if not date_string is defined %}\n    {%- if strftime_now is defined %}\n        {%- set date_string = strftime_now("%d %b %Y") %}\n    {%- else %}\n        {%- set date_string = "26 Jul 2024" %}\n    {%- endif %}\n{%- endif %}\n{%- if not tools is defined %}\n    {%- set tools = none %}\n{%- endif %}\n\n{#- This block extracts the system message, so we can slot it into the right place. #}\n{%- if messages[0][\'role\'] == \'system\' %}\n    {%- set system_message = messages[0][\'content\']|trim %}\n    {%- set messages = messages[1:] %}\n{%- else %}\n    {%- set system_message = "" %}\n{%- endif %}\n\n{#- System message #}\n{{- "<|start_header_id|>system<|end_header_id|>\\n\\n" }}\n{%- if tools is not none %}\n    {{- "Environment: ipython\\n" }}\n{%- endif %}\n{{- "Cutting Knowledge Date: December 2023\\n" }}\n{{- "Today Date: " + date_string + "\\n\\n" }}\n{%- if tools is not none and not tools_in_user_message %}\n    {{- "You have access to the following functions. To call a function, please respond with JSON for a function call." }}\n    {{- \'Respond in the format {"name": function name, "parameters": dictionary of argument name and its value}.\' }}\n    {{- "Do not use variables.\\n\\n" }}\n    {%- for t in tools %}\n        {{- t | tojson(indent=4) }}\n        {{- "\\n\\n" }}\n    {%- endfor %}\n{%- endif %}\n{{- system_message }}\n{{- "<|eot_id|>" }}\n\n{#- Custom tools are passed in a user message with some extra guidance #}\n{%- if tools_in_user_message and not tools is none %}\n    {#- Extract the first user message so we can plug it in here #}\n    {%- if messages | length != 0 %}\n        {%- set first_user_message = messages[0][\'content\']|trim %}\n        {%- set messages = messages[1:] %}\n    {%- else %}\n        {{- raise_exception("Cannot put tools in the first user message when there\'s no first user message!") }}\n{%- endif %}\n    {{- \'<|start_header_id|>user<|end_header_id|>\\n\\n\' -}}\n    {{- "Given the following functions, please respond with a JSON for a function call " }}\n    {{- "with its proper arguments that best answers the given prompt.\\n\\n" }}\n    {{- \'Respond in the format {"name": function name, "parameters": dictionary of argument name and its value}.\' }}\n    {{- "Do not use variables.\\n\\n" }}\n    {%- for t in tools %}\n        {{- t | tojson(indent=4) }}\n        {{- "\\n\\n" }}\n    {%- endfor %}\n    {{- first_user_message + "<|eot_id|>"}}\n{%- endif %}\n\n{%- for message in messages %}\n    {%- if not (message.role == \'ipython\' or message.role == \'tool\' or \'tool_calls\' in message) %}\n        {{- \'<|start_header_id|>\' + message[\'role\'] + \'<|end_header_id|>\\n\\n\'+ message[\'content\'] + \'<|eot_id|>\' }}\n    {%- elif \'tool_calls\' in message %}\n        {%- if not message.tool_calls|length == 1 %}\n            {{- raise_exception("This model only supports single tool-calls at once!") }}\n        {%- endif %}\n        {%- set tool_call = message.tool_calls[0].function %}\n        {{- \'<|start_header_id|>assistant<|end_header_id|>\\n\\n\' -}}\n        {{- \'{"name": "\' + tool_call.name + \'", \' }}\n        {{- \'"parameters": \' }}\n        {{- tool_call.arguments | tojson }}\n        {{- "}" }}\n        {{- "<|eot_id|>" }}\n    {%- elif message.role == "tool" or message.role == "ipython" %}\n        {{- "<|start_header_id|>ipython<|end_header_id|>\\n\\n" }}\n        {%- if message.content is mapping or message.content is iterable %}\n            {{- message.content | tojson }}\n        {%- else %}\n            {{- message.content }}\n        {%- endif %}\n        {{- "<|eot_id|>" }}\n    {%- endif %}\n{%- endfor %}\n{%- if add_generation_prompt %}\n    {{- \'<|start_header_id|>assistant<|end_header_id|>\\n\\n\' }}\n{%- endif %}\n'
    # Search Related Options
    n: int = 4
    temperature: float = 0.8
    top_p: float = 1.0
    prm_batch_size: int = 4
    search_batch_size: int = 25
    seed: int = 42
    max_tokens: int = 2048
    agg_strategy: str = "last"  # Options: "last", "min", "prod"

    # DVTS / Beam Search options
    beam_width: int = 4  # m in the paper
    num_iterations: int = 40
    lookahead: int = 0

    # Beam search options:
    filter_duplicates: bool = False
    sort_completed: bool = False
    # vLLM Triton decode kernel: "chunked" or "split"
    triton_decode_kernel: Literal["chunked", "split"] = "chunked"
    timing_enabled: bool = True
    # If True, enable inductor GEMM autotune over a dense compile_sizes ladder.
    # If False, fall back to vLLM defaults (cuBLAS via aten.mm).
    gemm_opt: bool = False
    # Selects how best_of_n_beam picks CHUNK_Size_Page during decode:
    #   "none"      -> FA path always (default).
    #   "dynamic"   -> profile-tuned LUT at data/{model_path}/profile_results_lut.json
    #                  (from scripts/chunk_setting_profile.py). LUT consulted at
    #                  threshold crossings; CHUNK_Size_Page is plain int (no
    #                  Triton recompile per change). SAL_ATTN_LUT_PATH overrides
    #                  the default path.
    #   "heuristic" -> route FD calls through sal.utils.xformers_attn, an
    #                  xformers-style kernel where CHUNK_Size_Page is
    #                  tl.constexpr. Chunk size is computed per call from
    #                  xformers' get_split_k (continuous in max_seq_len), so
    #                  each unique value triggers a Triton compile. No LUT.
    chunk_size: Literal["none", "dynamic", "heuristic"] = "none"
    # When True, scripts/test_time_compute.py skips load_prm() and the search
    # function (best_of_n_beam etc.) substitutes placeholder scores / pred.
    # Useful for timing-only studies where PRM compute is unwanted.
    disable_prm: bool = False
    # Pads the tokenized templated prompt (system_prompt + problem) to this
    # many tokens via left-padding. 0 disables padding; if the prompt is
    # already longer than this value, it is left unchanged.
    padded_prompt_len: int = 0

    def __post_init__(self):
        if self.approach == "dvts":
            if self.n % self.beam_width != 0:
                raise ValueError("n should be a multiple of beam_width")
            self.n_beams = self.n // self.beam_width

        if self.approach == "beam_search":
            # TODO: implemented a batched version
            if self.search_batch_size != 1:
                raise ValueError("search_batch_size should be 1 for beam_search")

        # Setting up push to hub dataset
        if self.push_to_hub:
            model_name = self.model_path.split("/")[-1]
            if self.hub_dataset_id is None:
                # Set default based on model name. We prepend the username for compatibility with the repo checks below.
                self.hub_dataset_id = get_full_repo_name(
                    f"{model_name}-{self.approach}-prm-completions"
                )
            revisions = get_dataset_revisions(self.hub_dataset_id)

            if self.approach == "beam_search" or self.approach == "dvts":
                self.revision = f"{self.dataset_name.replace('/', '_')}--T-{self.temperature}--top_p-{self.top_p}--n-{self.n}--m-{self.beam_width}--iters-{self.num_iterations}--look-{self.lookahead}--seed-{self.seed}--agg_strategy--{self.agg_strategy}"
            elif self.approach == "best_of_n":
                self.revision = f"{self.dataset_name.replace('/', '_')}--T-{self.temperature}--top_p-{self.top_p}--n-{self.n}--seed-{self.seed}--agg_strategy-{self.agg_strategy}"
            else:
                raise ValueError(f"Unknown approach {self.approach}")
            if self.dataset_start is not None and self.dataset_end is not None:
                self.revision = (
                    f"{self.revision}--chunk-{self.dataset_start}_{self.dataset_end}"
                )

            # Early exit if the revision on the Hub already exists
            if not self.overwrite_hub_revision and self.revision in revisions:
                # logger.info(f"Revision {revision} already exists on the Hub. Exiting.")
                exit()
