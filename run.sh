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

python eval.py --model $SAVE_PATH --data_file test_data_path
