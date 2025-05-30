import pandas as pd
import json
import os
import zhipuai
import sys
import time
from tqdm import tqdm

def get_result(content):
    zhipuai.api_key = "25818ad549e6c7896829a09973122636.UAjQJFMngcXMIVhg"
    response = zhipuai.model_api.invoke(
        model="chatglm_pro",
        prompt=[
            {"role": "user", "content": content},
        ],
        temperature=0.95,
        top_p=0.7,
        incremental=True
    )
    try:
        text=response['data']['choices'][0]['content']
    except:
        return " "
    return text

out_list=[]

outpath = "./retrive_train_change2_2.json"
with open("./retrive_train_data2_2.json","r",encoding="utf-8") as reader:
    dataset = json.load(reader)
    nall_data = dataset['data']
    count = len(nall_data)
    print('总数据量：' + str(count))
    n=0
    if os.path.exists(outpath):
        with open(outpath,"r",encoding='utf-8') as tempreader:
            start = len(tempreader.readlines())
            nall_data = nall_data[start:]
    for data in tqdm(nall_data):
        time.sleep(0.1)
        content = data['fact']
        n+=1
        content.replace('\n', '').replace('\r', '')
        input = '假设你是一名法官，请总结输入案情中的犯罪行为，输出格式为"被告人+犯罪过程+后果"：' + content
        output = get_result(input)
        out_dict = {'fact': content, 'law': data['law'], 'crime': output, 'num': data['num']}
        print(output)
        with open(outpath,"a",encoding="utf-8") as writer:
            writer.write(str(json.dumps(out_dict,ensure_ascii=False)))
            writer.write("\n")

