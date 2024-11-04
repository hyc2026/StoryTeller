# Copyright (2024) Bytedance Ltd. and/or its affiliates

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
from tasks.utils import load_model_and_processor
from dataset.utils import *
import os
from tqdm import tqdm
import json
import argparse
import torch
import torch.nn.functional as F

letters = [319, 350, 315, 360, 382]
numbers = [29900, 29896, 29906, 29941, 29946, 29945, 29953, 29955, 29947, 29929]

def process_one(model, processor, data, generate_kwargs):
    inputs = processor(data)
    for k in inputs:
        if isinstance(inputs[k], torch.Tensor):
            inputs[k] = inputs[k].to(model.device)

    score, output_text = -1, ""
    if inputs['input_ids'][0][-1] == 396: # 结尾是`"▁#": 396`则要计算logits
        score = 1
        model_out = model(
            attention_mask=torch.ones_like(inputs['input_ids']).to(model.device),
            **inputs,
        )
        # 找到最后一个`":": 29901`的index
        for i in range(-1, -len(inputs['input_ids'][0]), -1):
            if inputs['input_ids'][0][i] == 29901:
                break
        assert  inputs['input_ids'][0][i - 2] in letters, "error letter: {}".format(inputs['input_ids'][0][i - 2])
        assert  inputs['input_ids'][0][i - 1] in numbers, "error number: {}".format(inputs['input_ids'][0][i - 1])
        softmax_tensor = F.softmax(model_out.logits, dim=2)
        for j in range(i, -2):
            score *= softmax_tensor[0][j][inputs['input_ids'][0][j + 1]].item()
    else:
        outputs = model.generate(
            **inputs,
            **generate_kwargs,
        )
        output_text = processor.tokenizer.decode(outputs[0][inputs['input_ids'][0].shape[0]:], skip_special_tokens=True)
    return output_text, score


def run():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name_or_path', type=str, default="checkpoints/Whisper-large-v2-Tarsier-7B-character-identification")
    parser.add_argument('--input_path', type=str, default="data/audio_visual_diarization/data.jsonl")
    parser.add_argument('--output_path', type=str, default="data/audio_visual_diarization/0.jsonl")
    parser.add_argument("--max_new_tokens", type=int, default=512, help="max number of generated tokens")
    parser.add_argument("--temperature", type=float, default=0, help="Set temperature > 0 to enable sampling generation.")
    args = parser.parse_args()

    model, processor = load_model_and_processor(args.model_name_or_path)
    model.cuda()
    generate_kwargs = {
        "do_sample": True if args.temperature > 0 else False,
        "max_new_tokens": args.max_new_tokens,
        "temperature": args.temperature,
        "use_cache": True
    }
    assert os.path.exists(args.input_path), f"input_path not exist: {args.input_path}"
    with open(args.input_path) as fin, open(args.output_path, "w") as fout:
        for line in tqdm(fin.readlines()):
            data = json.loads(line)
            data["text"]["prediction"], data["text"]["score"] = process_one(model, processor, data, generate_kwargs)
            fout.write(json.dumps(data, ensure_ascii=False) + '\n')

if __name__ == "__main__":
    run()