# Exploring Reversal Mathematical Reasoning Ability for Large Language Models (ACL 2024 Findings)

Some codes are borrowed from 

[MetaMath]: https://github.com/meta-math/MetaMath



<h2 id="env">Quick Start</h2>

Clone the repository and install the required packages:

```bash
git clone 
cd ReversalMath
pip install -r requirements.txt
```



<h2 id="env">Datasets</h2>

We release all the test sets experimented on in our paper, and they can be found in "./data". Specially, for GSM8K and MathQA datasets, we have released **forward and backward reasoning data**.




<h2 id="train">Training</h2>

you need to prepare the  llama-2 base model and the training data which included in "./data"

```
bash run.sh
```
or

```
export MODEL_PATH='Llama-2-7b-hf-path'
export SAVE_PATH='./save/path'
export WANDB_DISABLED=true
wandb offline
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python3 train.py \
    --model_name_or_path $MODEL_PATH \
    --data_path "data/gsm8k/train.json" \
    --data_length 10000000 \
    --bf16 True \
    --output_dir $SAVE_PATH \
    --num_train_epochs 3 \
    --per_device_train_batch_size 4 \
    --per_device_eval_batch_size 4 \
    --gradient_accumulation_steps 4 \
    --evaluation_strategy "no" \
    --save_strategy "steps" \
    --save_steps 200 \
    --save_total_limit 3 \
    --learning_rate 2e-5 \
    --weight_decay 0. \
    --warmup_ratio 0.03 \
    --lr_scheduler_type "cosine" \
    --logging_steps 1 \
    --tf32 True \
    2>&1 & 
```



<h2 id="evaluation">Evaluation</h2>

we use the vllm to help the fast generation:

```bash
python eval.py --model $SAVE_PATH --data_file test_data_path
```
The inference prompt for our MetaMath is:
```
"Below is an instruction that describes a task. Write a response that appropriately completes the request.\n\n### Instruction:\n{instruction}\n\n### Response: "
```
