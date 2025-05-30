PRE_SEQ_LEN=128
LR=1e-2

CUDA_VISIBLE_DEVICES=0 python3 main.py \
    --do_train \
    --train_file "/home/longjing/zlz/ChatGLM-6B-main/ptuning/data/CAIL_big/std_train_big1.json" \
    --validation_file "/home/longjing/zlz/ChatGLM-6B-main/ptuning/data/CAIL_big/std_train_big1.json" \
    --prompt_column input \
    --response_column output \
    --overwrite_cache \
    --model_name_or_path "/home/longjing/zlz/ChatGLM-6B-main/chatglm1/" \
    --output_dir output/output_lawer/checkpoint_big \
    --overwrite_output_dir \
    --max_source_length 1600 \
    --max_target_length 100 \
    --per_device_train_batch_size 1 \
    --per_device_eval_batch_size 1 \
    --gradient_accumulation_steps 16 \
    --predict_with_generate \
    --max_steps 5000 \
    --logging_steps 10 \
    --save_steps 1000 \
    --learning_rate $LR \
    --pre_seq_len $PRE_SEQ_LEN \
    --fp16


