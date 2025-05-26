export DEBUG_MODE="true"

RUN_NAME="UniVG-R1"
export LOG_PATH="./debug_log_$RUN_NAME.txt"
MODEL_NAME=/path/to/your/stage1_cotsft_model

torchrun --nproc_per_node="8" \
    --nnodes="1" \
    --node_rank="0" \
    --master_addr="127.0.0.1" \
    --master_port="12346" \
    /path/to/UniVG-R1/src/open-r1-multimodal/src/open_r1/grpo_univg.py \
    --deepspeed /path/to/UniVG-R1/src/open-r1-multimodal/local_scripts/zero3.json \
    --output_dir /path/to/UniVG-R1/src/open-r1-multimodal/output/$RUN_NAME \
    --model_name_or_path $MODEL_NAME \
    --dataset_name /path/to/your/stage2_rl.json \
    --image_root /path/to/your/stage2_rl.json \
    --freeze_vision_modules true \
    --max_prompt_length 1024 \
    --num_generations 8 \
    --per_device_train_batch_size 8 \
    --gradient_accumulation_steps 2 \
    --logging_steps 1 \
    --bf16 \
    --torch_dtype bfloat16 \
    --data_seed 42 \
    --report_to wandb \
    --gradient_checkpointing true \
    --attn_implementation flash_attention_2 \
    --num_train_epochs 1 \
    --run_name $RUN_NAME \
    --save_steps 200 \
    --save_only_model true \
    --single_img_with_cot False \
    --miou_adjust True \
    --miou_adjust_math exp