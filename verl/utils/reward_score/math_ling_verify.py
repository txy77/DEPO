# Copyright 2024 Bytedance Ltd. and/or its affiliates
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

try:
    from math_verify.metric import math_metric
    from math_verify.parser import LatexExtractionConfig, ExprExtractionConfig
except ImportError:
    print("To use Math-Verify, please install it first by running `pip install math-verify`.")

import re
from math_verify import parse, verify


'''
def compute_score(model_output: str, ground_truth: str) -> bool:
    verify_func = math_metric(
        gold_extraction_target=(LatexExtractionConfig(),),
        pred_extraction_target=(ExprExtractionConfig(), LatexExtractionConfig()),
    )
    ret_score = 0.

    # Wrap the ground truth in \boxed{} format for verification
    ground_truth_boxed = "\\boxed{" + ground_truth + "}"
    try:
        ret_score, _ = verify_func([ground_truth_boxed], [model_output])
    except Exception as e:
        pass

    return ret_score
'''

def parse_dapo(response):
    pattern = re.compile(r"Answer:\s*(.*?)(?=\s*Answer:|\Z)", re.DOTALL)
    matches = re.findall(pattern, response)
    matches = [match for match in matches if match]

    return matches[-1] if matches else ""

def convert_to_boxed(response):
    if not response:
        return response
    elif response.startswith('\\boxed{'):
        if response.endswith('}'):
            return response
        else:
            return response + '}'
    else:
        return '\\boxed{' + response + '}'

def compute_score(model_output: str, ground_truth: str) -> bool:
    parsed_ground_truth = parse(convert_to_boxed(ground_truth))
    parsed_output = parse(convert_to_boxed(parse_dapo(model_output)))
    correct = verify(parsed_ground_truth, parsed_output)

    reward = 1.0 if correct else -1.0
    acc = correct

    return {
        "score": reward,
        "acc": acc,
        # "pred": parsed_output,
    }