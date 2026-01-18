accelerate launch --num_processes=1 --gpu_ids="0" --main_process_port 29409 src/test.py \
    --sd_path="./SD" \
    --output_dir="" \
    --resolution=512 \
    --train_batch_size=4 \
    --enable_xformers_memory_efficient_attention \
    --pretrained_path="" 