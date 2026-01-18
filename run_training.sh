accelerate launch --num_processes=4 --gpu_ids="0,1,2,3" --main_process_port 29302 src/train.py \
    --sd_path="./SD" \
    --output_dir="" \
    --resolution=512 \
    --train_batch_size=4 \
    --enable_xformers_memory_efficient_attention \
    --num_samples_eval=5 \ 
    #--pretrained_path="" 