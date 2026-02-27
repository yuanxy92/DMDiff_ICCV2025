accelerate launch --num_processes=4 --gpu_ids="1,2,3,4" --main_process_port 29302 src/train.py \
    --sd_path="./SD" \
    --output_dir="/data/jianing/DMDiff_ICCV2025-main/output_dir" \
    --resolution=512 \
    --train_batch_size=4 \
    --enable_xformers_memory_efficient_attention \
    --num_samples_eval=5 
