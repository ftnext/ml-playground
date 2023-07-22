#!/usr/bin/env bash
set -euo pipefail

for i in 1 2 3 4 5
do
  output_dir=data/${i}
  mkdir -p ${output_dir}
  python create_prompts.py ${output_dir}/prompts.jsonl --instruction_number ${i}
done
