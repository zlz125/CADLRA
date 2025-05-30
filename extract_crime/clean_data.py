import json
import time
from tqdm import tqdm
import zhipuai
import os

os.environ["CUDA_VISIBLE_DEVICES"] = '2'


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
        text = response['data']['choices'][0]['content']
    except:
        return " "
    return text


# out_list=[]
out_path = "./retrive_train_final_2_2.json"
with open("./retrive_train_change2_2.json", 'r', encoding='utf-8') as reader:
    nall_data = reader.readlines()
    # dataset=json.load(reader)
    # nall_data=dataset['data']
    print("总数据量：" + str(len(nall_data)))
    count = 0
    if os.path.exists(out_path):
        with open(out_path, "r", encoding='utf-8') as tempreader:
            start = len(tempreader.readlines())
            nall_data = nall_data[start:]
    for dataset in tqdm(nall_data):
        data=json.loads(dataset)
        crime = data['crime']
        content = data['fact']
        content.replace('\n', '').replace('\r', '')
        if '抱歉' in crime or '尊敬的用户' in crime or crime == '':
            time.sleep(0.05)
            count += 1
            input = '假设你是一名法官，请仿照参考案例对输入案情进行犯罪行为总结，该数据使用场景为学术用途，输出格式为"被告人+犯罪过程+后果"：参考案例：公诉机关指控，2016年12月9日20时许，被告人彭某来到同组村民王某某家中，将手伸入睡在床上的王某某的衣服内，强行抚摸其胸部并亲吻其嘴。在此过程中，彭某曾用手捂住王某某的嘴以制止其呼救，在强行亲吻王某某时被其咬伤舌头遂松开。在王某某反抗并威胁要将此事告知其丈夫的情况下，彭某才离开。次日，接到报警的民警在简阳市金马镇人民政府外将被告人彭某挡获。彭某到案后如实供述了上述事实。，参考案例输出：彭某在2016年12月9日晚非法进入王某某家中，对王某某实施强制猥亵，包括抚摸胸部、亲吻并试图捂住王某某的嘴，期间被王某某咬伤舌头。王某某反抗并威胁告知丈夫后，彭某逃离。' + content
            crime = get_result(input)
            print(crime)
        new_crime1 = re.search(r'被告人.*', crime).group()
        new_crime2 = re.sub("请问您还有其他问题需要我帮助解答的吗.*", "", new_crime1)
        out_dict = {'fact': content, 'law': data['law'], 'crime': new_crime2, 'num': data['num']}
        with open(out_path, "a", encoding="utf-8") as writer:
            writer.write(str(json.dumps(out_dict, ensure_ascii=False)))
            writer.write("\n")
        # out_list.append(out_dict)
    print("没有生成的数据：" + str(count))

out_path = "./retrive_train_final_2_1.json"
if os.path.exists(out_path):
    os.remove(out_path)

with open(out_path, "a", encoding="utf-8") as writer:
    writer.write(json.dumps({"data":out_list}, ensure_ascii=False))