# -*- coding: UTF-8 -*-
import logging
import os

os.environ["CUDA_VISIBLE_DEVICES"] = '2'
import sys
import json
import sys
import time
import numpy as np
from datasets import load_dataset
import jieba
from rouge_chinese import Rouge
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
import torch

import transformers
from transformers import (
    AutoConfig,
    AutoModel,
    AutoTokenizer,
    AutoTokenizer,
    DataCollatorForSeq2Seq,
    HfArgumentParser,
    Seq2SeqTrainingArguments,
    set_seed,
)
from trainer_seq2seq import Seq2SeqTrainer
from arguments import ModelArguments, DataTrainingArguments
from transformers import AutoTokenizer, AutoModel, AutoConfig

logger = logging.getLogger(__name__)
from loguru import logger

logger.add(
    "./log/charge_error.log",
    encoding="utf-8",
    format="{level} | {time:YYYY-MM-DD HH:mm:ss} | {file} | {line} | {message}",
    retention="30 days",
    rotation="500 MB"
)

checkpoint_path = "/home/longjing/zlz/ChatGLM-6B-main/ptuning/output/output_lawer/adgen5-chatglm-6b-pt-128-1e-2/checkpoint-5000/"

config = AutoConfig.from_pretrained("/home/longjing/zlz/ChatGLM-6B-main/chatglm1/", trust_remote_code=True)
config.pre_seq_len = 128
tokenizer = AutoTokenizer.from_pretrained("/home/longjing/zlz/ChatGLM-6B-main/chatglm1/", trust_remote_code=True)
model = AutoModel.from_pretrained("/home/longjing/zlz/ChatGLM-6B-main/chatglm1/", config=config,
                                  trust_remote_code=True).half().cuda()

prefix_state_dict = torch.load(os.path.join(checkpoint_path, "pytorch_model.bin"))

new_prefix_state_dict = {}
for k, v in prefix_state_dict.items():
    new_prefix_state_dict[k[len("transformer.prefix_encoder."):]] = v

model.transformer.prefix_encoder.load_state_dict(new_prefix_state_dict)

model = model.eval()

# logger.info("chatGLM's answer: " + result + f'\treal answer:{real_answer}' + f'\t当前准确率: {correct_ratio}')
out_list = []
error_list = []
with open("/home/longjing/zlz/ChatGLM-6B-main/ptuning/data/CAIL_small/random_lawer_test.json", 'r') as reader:
    data = json.load(reader)
    print("总的条数:" + str(len(data)))
    p=0.0
    r=0.0
    f=0.0
    count = 0
    num = float(len(data))
    n = 0
    correct_ratio = 0.0
    for file in data:
        temp_p=0.0
        temp_r=0.0
        temp_f=0.0
        correct=0.0
        acc=0.0
        n += 1
        input = file["input"][:1600]
        temp = file["output"]
        label1 = str.replace(temp, ';', ',')

        prediction1, history = model.chat(tokenizer, input, history=[])
        prediction2 = prediction1.replace('[', '')
        prediction3=prediction2.replace(']', '')
        prediction4=prediction3.replace('；','，')
        prediction = prediction4

        # 防止罪名顺序不一致导致的不相同，先分词再排序
        a1 = label1.split(",")
        b1 = prediction.split("，")
        new_label = []
        new_pre = []

        for i in a1:
            new_label.append(i.replace(" ", ""))
        
        for j in b1:
            # if '罪' in j:
            #     new_pre.append(j.replace("罪",""))
            # else:
            new_pre.append(j.replace(" ", ""))

        label3 = sorted(new_label)
        prediction3 = sorted(new_pre)

        for pre in prediction3:
            if pre in label3:
                correct+=1
        temp_p=correct/len(prediction3)
        temp_r=correct/len(label3)

        p+=temp_p
        r+=temp_r

        label = str(label3)
        prediction = str(prediction3)

        if label==prediction:
            count+=1
            acc+=1

        out_dict = {"input":file["input"],'prediction': prediction, 'real_answer': label}
        out_list.append(out_dict)

        logger.info(str(n) + "chatGLM's answer: " + prediction + f'\treal answer:{label}' + f'\t当前acc值：{acc}'+f'\t当前p值: {temp_p}'+ f'\t当前r值：{temp_r}')
    f = (2 * p * r) / (p + r)
    #logger.info("原生预测出错的字数少于1600加上法条"+"\t"+str(len(data)))
    logger.info("总的正确个数：" + str(count))
    logger.info("Acc：" + str(count / num))
    logger.info("P值：" + str(p/num)+'\t'+str(p))
    logger.info("R值：" + str(r / num)+'\t'+str(r))
    logger.info("F1值：" + str(f / num)+'\t'+str(f))

out_path = "./results/CAIL_small/random_lawer_2825_lawer.json"

if os.path.exists(out_path):
    os.remove(out_path)

with open(out_path, "a", encoding="utf-8") as writer:
    writer.write(json.dumps({"data": out_list}, ensure_ascii=False))

