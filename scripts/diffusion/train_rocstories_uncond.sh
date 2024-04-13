python train_text_diffusion.py \
    --dataset_name rocstories \
    --learning_rate 2e-4 \
    --num_train_steps 250000 \
    --train_batch_size 128 \
    --tx_dim 768 \
    --tx_depth 12 \
    --objective pred_v \
    --enc_dec_model facebook/bart-base \
    --num_samples 1000 \
    --self_condition \
    --scale_shift \
    --loss_type l2 \
    --train_schedule cosine \
    --wandb_name roc_latent_v \
    --sampling_timesteps 250 \
    --latent_model_path "saved_latent_models/roc/2024-03-07_14-57-58" \
    --save_and_sample_every 25000 \
    --num_dense_connections 3  \
    --optimizer adamw \
    --train_prob_self_cond 0.5 \
    --mode unconditional \
    --model_id ld4lg

    # --resume_training \
    # --resume_dir "saved_diff_models/roc/2024-04-13_16-26-20"
# Need to update latent_model_path to the correct path
