PRE_SEQ_LEN=128
CHECKPOINT=adgen1-chatglm-6b-pt-128-1e-2
STEP=3000

CUDA_VISIBLE_DEVICES=1,2 python3 main.py \
    --do_predict \
    --validation_file "/home/longjing/zlz/ChatGLM-6B-main/ptuning/data/CAIL_small/dev5_1_modify.json" \
    --test_file "/home/longjing/zlz/ChatGLM-6B-main/ptuning/data/CAIL_small/dev5_1_modify.json" \
    --overwrite_cache \
    --prompt_column input \
    --response_column output \
    --model_name_or_path "/home/longjing/zlz/ChatGLM-6B-main/chatglm1/" \
    --ptuning_checkpoint "/home/longjing/zlz/ChatGLM-6B-main/ptuning/output/output_small/checkpoint-3000/" \
    --output_dir ./output/output_small/checkpoint-3000 \
    --overwrite_output_dir \
    --max_source_length 1500 \
    --max_target_length 100 \
    --per_device_eval_batch_size 1 \
    --predict_with_generate \
    --pre_seq_len $PRE_SEQ_LEN \
    --fp16
