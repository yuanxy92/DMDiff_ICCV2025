accelerate launch --num_processes=1 --gpu_ids="5" --main_process_port 29409 src/test.py \
    --sd_path="./SD" \
    --output_dir="/data/jianing/DMDiff_ICCV2025-main/output_dir_images/" \
    --resolution=512 \
    --train_batch_size=4 \
    --enable_xformers_memory_efficient_attention \
    --pretrained_path="/data/jianing/DMDiff_ICCV2025-main/output_dir/checkpoints/model_14501.pkl" 