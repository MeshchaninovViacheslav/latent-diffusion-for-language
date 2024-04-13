python train_text_diffusion.py --eval \
    --dataset_name rocstories \
    --resume_dir saved_diff_models/roc/2024-03-16_12-41-51 \
    --sampling_timesteps 250 \
    --num_samples 1000 \
    --wandb_name roc_ddpm \
    --sampler ddpm \
    --sampling_schedule cosine \
    --mode unconditional \
    --model_id ld4lg