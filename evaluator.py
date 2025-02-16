# in case you want to run this script independently for ssr-vles

import argparse
from openai import OpenAI
from openai._exceptions import RateLimitError
import json
import os
from tqdm import tqdm
import pandas as pd
import numpy as np
import time
import pathlib
from retrying import retry
from statsmodels. stats . weightstats import DescrStatsW
import datetime


prompt = """
The following information is system prompt:
You are a marking teacher, now to mark the work, please compare the ground truth and prediction from AI models to give a correctness score for the prediction.You need to follow the following scoring rules, each of which is equally important:
 1.<AND> in the ground truth means it is totally right only when all elements in the ground truth are present in the prediction, and <OR> means it is totally right when any one element in the ground truth is present in the prediction. 
 2.The correctness score is 0.0 (totally wrong), 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, or 1.0 (totally right). Don't have any extra output.
 3.Ignore extra ' '(space symbol), for example, '(x + 2) ^ 2 = 9' and '(x+2)^2=9' are equivalent,they all got perfect score.
 4.Ignore the difference between upper and lower case letters，for example,'right' and 'Right' are equivalent.
 5.When the basic facts are long, score the predicted answers based on the main content of the text, without having to be identical word for word.
 6.They are considered equivalent as long as the meaning is the same, for example, 0 and no one are equivalent.
 7.All the Ground truth appeared in Prediction and no additional relevant answers were judged to be full marks. 
 8.Synonyms are also treated as inclusive relations, equal in price to the correct answer. Semantic similarity is awarded according to the degree of correlation.
The scoring process is divided into two steps, and here's what you need to do in each step:
1. According to the content of 'Question' and the style of 'Ground truth', extract the predicted answer of the model from 'Prediction'. Please note that the content format of 'Prediction' to be extracted is similar to that of 'Ground truth', but the content may not be the same. In this step, you only need to extract without judging whether it is right or wrong; When 'Prediction' is concise enough, you may not need to make any changes; 'Question' can be multiple choice or open-ended, you need to look at it on a case-by-case basis.
2. According to the scoring rules mentioned above, compare 'Ground truth' and 'Prediction' and output the score.
Below are eight answers format I am going to upload along with some examples:
Question | Ground truth | Prediction | Correctness
--- | --- | --- | ---
What is the answer to the equation?| -1 <AND> 8 | x = 2 | 0.0
What is the answer to the equation?| -1 <AND> 8 | x = -1 | 0.5
What is the answer to the equation?| -1 <AND> 8 | x = 8 | 0.5
What is the answer to the equation?| -1 <AND> 8 | x = -1 or 1 | 0.5
What is the answer to the equation?| -1 <AND> 8 | x = -1 or x = -5 | 1.0
What is the answer to the equation?| -1 <AND> 8 | x = -1 , x = -5 | 1.0
What is the answer to the equation?| -1 <AND> 8 |The answer is x = -1 , x = -5| 1.0
Can you describe the picture? | This meme is poking fun at the fact that the names of the countries Iceland and Greenland are misleading. Despite its name, Iceland is known for its beautiful green landscapes, while Greenland is mostly covered in ice and snow. The meme is saying that the person has trust issues because the names of these countries do not accurately represent their landscapes. | The meme talks about Iceland and Greenland. It's pointing out that despite their names, Iceland is not very icy and Greenland isn't very green. | 0.4
Can you describe the picture?| This meme is poking fun at the fact that the names of the countries Iceland and Greenland are misleading. Despite its name, Iceland is known for its beautiful green landscapes, while Greenland is mostly covered in ice and snow. The meme is saying that the person has trust issues because the names of these countries do not accurately represent their landscapes. | The meme is using humor to point out the misleading nature of Iceland's and Greenland's names. Iceland, despite its name, has lush green landscapes while Greenland is mostly covered in ice and snow. The text 'This is why I have trust issues' is a playful way to suggest that these contradictions can lead to distrust or confusion. The humor in this meme is derived from the unexpected contrast between the names of the countries and their actual physical characteristics. | 1.0
The following information is text input, in the same format as the examples above:
"""
add_prompt = """The following are the scoring results for a certain question. The scores are only within the following range: 0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0.
Please extract the specific score according to the following text. Please note that you can only output one of the candidate scores as a number and cannot output any other content. 
The text to be extracted is as follows:"""



def arg_parser(model,prompt=prompt):
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--ssrvet_path",
        type=str,
        default="ssr-vet",
        help="Download ssr-vet.zip and `unzip ssr-vet.zip` and change the path here",
    )
    parser.add_argument(
        "--result_file",
        type=str,
        default="model/results/"+ model + ".json",
        help="path to the model result file, must end with .json",
    )
    parser.add_argument(
        "--result_path",
        type=str,
        default="model/results/" + model,
        help="Evaluation results secondary directory location",
    )
    parser.add_argument(
        "--openai_api_key", type=str, default="key",
        help="Please set the api key here otherwise the environment key will be used"
    )
    parser.add_argument(
        "--openai_api_base_url",type=str,default= "url",
        help="OpenAI API base URL",
    )
    parser.add_argument(
        "--gpt_model", type=str, default="DeepSeek-R1-HouShan", help="save name"
    )
    parser.add_argument(
        "--gpt_model_client", type=str, default="deepseek-r1", help="Client name"
    )
    parser.add_argument(
        "--gpt_model_add", type=str, default="deepseek-ai/DeepSeek-V3", help="The model extracts specific scores from the analysis of the answer R1 model."
    )
    parser.add_argument(
        "--gpt_model_add_key", type=str, default="key", help="The model extracts specific scores from the analysis of the answer R1 model."
    )
    parser.add_argument(
        "--gpt_model_add_url", type=str, default="url", help="The model extracts specific scores from the analysis of the answer R1 model."
    )
    parser.add_argument(
        "--gpt_model_add_prompt", type=str, default=add_prompt, help="The model extracts specific scores from the analysis of the answer R1 model."
    )
    parser.add_argument(
        "--prompt", type=str, default=prompt, help="prompt for the model test"
    )

    parser.add_argument(
        "--robu_len", type=int, default=3, help="Number of robustness test questions per set"
    )
    args = parser.parse_args()
    return args

def get_grade_file(args, model):
    try:
        new_folder = (args.result_path)
        os.makedirs(new_folder)
    except:
        print("new_folder is already existed")
    # Store score results
    grade_file_json = f"1.{model}_{args.gpt_model}-grade-self-reflection.json"
    grade_file_json = os.path.join(args.result_path, grade_file_json)

    # Classification of statistical scores
    indiv_cap_file = f"2.{model}_{args.gpt_model}-indiv-cap_score.csv"
    indiv_cap_file = os.path.join(args.result_path, indiv_cap_file)
    comb_cap_file = f"3.{model}_{args.gpt_model}-comb-cap_score.csv"
    comb_cap_file = os.path.join(args.result_path, comb_cap_file)

    return grade_file_json, indiv_cap_file, comb_cap_file


def load_metadata(args):
    #获得单独/组合能力的数量

    #问题源数据
    ssrvet_metadata = os.path.join(args.ssrvet_path, "ssr-vet.json")
    with open(ssrvet_metadata, "r",encoding='UTF-8') as f:
        metadatas = json.load(f)

    #单独能力统计计数
    indiv_vision_cap = {}
    indiv_lag_cap = {}
    indiv_robu_cap = {}
    indiv_mark = {"vision":[0,0,0,0], "lag":[0,0,0,0], "robu":[0,0,0,0],"total_not_robu":[0,0,0,0]}
    len_data = 0
    for id, value in metadatas.items():
        cap_list = value["capability_vision"]
        for cap in cap_list:
            if cap not in indiv_vision_cap:
                indiv_vision_cap[cap] = 1
            else:
                indiv_vision_cap[cap] += 1
        cap_list = value["capability_language"]
        if not cap_list[0] == "":
            for cap in cap_list:
                if cap not in indiv_lag_cap:
                    indiv_lag_cap[cap] = 1
                else:
                    indiv_lag_cap[cap] += 1
        if "capability_robu" in value:
            cap_list = value["capability_robu"]
            if cap_list[0] not in indiv_robu_cap:
                indiv_robu_cap[cap_list[0]] = 1
            else:
                indiv_robu_cap[cap_list[0]] += 1
        len_data += 1
    indiv_cap = {"indiv_vision_cap":indiv_vision_cap,"indiv_lag_cap":indiv_lag_cap,"indiv_robu_cap":indiv_robu_cap,"indiv_mark":indiv_mark}


    #复合能力计数
    comb_list = []
    comb_cap = {}
    comb_mark = [0,0]
    comb_metadata = os.path.join(args.ssrvet_path, "Combined.json")
    with open(comb_metadata, "r") as f:
        comb_cap_file = json.load(f)
    for cap_id,cap_value in comb_cap_file.items():
        if cap_value not in comb_cap.items():
            tmp = cap_value
            comb_list.append(cap_value)
            comb_cap["_".join(set(tmp))] = 0
            #comb_mark[cap_value] = [0,0,0]


    for id, value in metadatas.items():
        if not value["capability_language"] == "" :
            comb_cap_tmp = value["capability_vision"] + value["capability_language"]
        else:
            comb_cap_tmp = value["capability_vision"]
        for cap in comb_list:
            #对比cap是否在comb_cap_tmp中
            if set(cap).issubset(set(comb_cap_tmp)):
                comb_cap["_".join(set(cap))] += 1
    comb = {"comb_cap":comb_cap,"comb_mark":comb_mark}

    count = f"4.Count-{model}_{args.gpt_model}.json"
    count = os.path.join(args.result_path, count)


    try:
        new_folder = (args.result_path)
        os.makedirs(new_folder)
    except:
        print("new_folder is already existed")
    with open(count, "w",encoding='UTF-8') as f:
        json.dump(indiv_cap, f)
        json.dump(comb,f)
        print("Count is Ok")

    return (
        metadatas,
        indiv_cap,
        comb,
        len_data
    )


def runs(
    args,
    grade_file_json,
    metadatas,
    len_data,
    client
):
    #测评结果
    with open(args.result_file,encoding='UTF-8') as f:
        results = json.load(f)
    #分数结果
    if os.path.exists(grade_file_json):
        with open(grade_file_json, "r") as f:
            grade_results_json = json.load(f)
    else:
        grade_results_json = {}
    print("Start by self-reflection scoring system")

    for id, value in tqdm(results.items()):
        if id in grade_results_json:
            continue
        print(f"\n \033[1;31;46m {id} \033[0m")
        print("\n")
        #对齐答案、问题、回答结果
        metadata = metadatas[id]
        answer = metadata["answer"]
        question = metadata["question"]
        degree_list = metadata["degree"]

        result_value = []    #取出各回合的第一个问题的模型回复
        robu_value=[]        #鲁棒性单独计算
        for deg in degree_list:
            print(f"Question{deg}")
            if deg != 2:
                result_value.append(value[f"mark_{deg}"]["response1"])
            else:
                for i in range(args.robu_len):
                    robu_value.append(value[f"mark_{deg}"][f"robu_{i}"]["response1"])

        grade_results_json[id] = {}
        for i in range(len(degree_list)):
            if degree_list[i] != 2:
                tmp = []
                tmp,retries = run_srg(args,client, answer[i], question[i], result_value[i])
                structure = {
                    "self-reflection":retries,
                    "grade":tmp
                }
                grade_results_json[id][f"score_{degree_list[i]}"] = structure
            else:
                grade_results_json[id][f"score_{degree_list[i]}"] = {}
                for j in range(len(robu_value)):
                    tmp = []
                    tmp,retries= run_srg(args,client,answer[i][j],question[i][j],robu_value[j])
                    structure = {
                        "self-reflection": retries,
                        "grade": tmp
                    }
                    grade_results_json[id][f"score_{degree_list[i]}"][f"robu_{j}"] = structure

        print(grade_results_json[id])
        #一级项写入硬盘
        with open(grade_file_json, "w") as f:
            json.dump(grade_results_json, f, indent=4)
            print("\n")
            print(id + "json input is ok")
            print("\n\n")

    print("After reflecting on the end of the assessment, the score has been written into 'grade results json'")
    return grade_results_json



def run_srg(args,client,answer,question,result_value):
    retries = 0
    content = []
    content_message = []
    temperature = []
    que = (
            args.prompt
            + "\n"
            + " | ".join(
        [
            question,
            answer
            .replace("<AND>", " <AND> ")
            .replace("<OR>", " <OR> "),
            result_value,
            "",
        ]
    )
    )
    for i in range(3):
         content_tmp,content_message_tmp,temperature_tmp= single_polling(args,client,que,(i*0.5))
         content.append(content_tmp)
         content_message.append(content_message_tmp)
         temperature.append(temperature_tmp)
    print(content)
    if len(set(content)) == 1:
        retries = 0
    elif len(set(content)) == 2:
        print("reanswer action")
        if content[0] == content[1]:
            content[2] = second_polling(args,client,content_message[2], temperature[2])
            retries = 1
            if not len(set(content)) == 1:
                content[2] = second_polling(args, client, content_message[2], temperature[2])
                retries = 2
                if not len(set(content)) == 1:
                    retries = -1
        elif content[1] == content[2]:
            content[0] = second_polling(args,client,content_message[0], temperature[0])
            retries = 1
            if not len(set(content)) == 1:
                content[0] = second_polling(args, client, content_message[0], temperature[0])
                retries = 2
                if not len(set(content)) == 1:
                    retries = -1
        else:
            content[1] = second_polling(args,client,content_message[1], temperature[1])
            retries = 1
            if not len(set(content)) == 1:
                content[1] = second_polling(args, client, content_message[1], temperature[1])
                retries = 2
                if not len(set(content)) == 1:
                    retries = -1
    else:
        print("cycle answer action")
        count = 0
        list = []
        retries += 1
        for i in range(3):
            content_tmp = second_polling(args, client, content_message[i], temperature[i])
            retries += 1
            if content_tmp == content[i]:
                if i == 0:
                    count = 1
                    list = [1,2]
                    break
                else:
                    ctmp = content[0]
                    mtmp = content_message[0]
                    ttmp = temperature[0]
                    content[0] = content[i]
                    content_message[0] = content_message[i]
                    temperature[0] = temperature[i]
                    content[i] = ctmp
                    content_message[i] = mtmp
                    temperature[i] = ttmp
                    list = [1,2]
                    break
            if content_tmp in content:
                for j in range(3):
                    if content[j] != content_tmp:
                        list = [j]
                break
            if i == 2:
                retries = -1
                return content, retries
        for k in list:
            content[k] = second_polling(args,client, content_message[k], temperature[k])
            retries += 1
            if not len(set(content)) == 1:
                retries = -1

        # if content[count] == content[count+1]:
        #     content[count-1] = second_polling(args,client,content_message[count-1], temperature[count-1])
        #     retries += 1
        #     if not len(set(content)) == 1:
        #         content[count-1] = second_polling(args, client, content_message[count-1], temperature[count-1])
        #         retries += 1
        #         if not len(set(content)) == 1:
        #             retries = -1
        # else:
        #     content[count+1] = second_polling(args,client,content_message[count+1], temperature[count+1])
        #     retries += 1
        #     if not len(set(content)) == 1:
        #         content[count+1] = second_polling(args, client, content_message[count+1], temperature[count+1])
        #         retries += 1
        #         if not len(set(content)) == 1:
        #             retries = -1

    return content,retries


@retry(stop_max_attempt_number = 5,wait_fixed =30000)
def single_polling(args,client,que,temperature):
    messages = [
        {"role": "user", "content": que},
    ]
    try:
        print(args.gpt_model_client)
        time2 = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        print(time2)
        response = client.chat.completions.create(
                model=args.gpt_model_client,
                #max_tokens=3,
                temperature=temperature,
                messages=messages,
            )
        content = response.choices[0].message.content
        print(content)
        print("*************************")

    except RateLimitError as e:
        print("The gpt is busy. Wait 30 seconds")
        time.sleep(30)
        raise Exception("gpt is busy")

    messages.append({"role": "assistant", "content": response.choices[0].message.content})
    try:
        content = float(content)
    except:
        print("----------------------")
        time2 = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        print(time2)
        client_add = OpenAI(
            base_url=args.gpt_model_add_url,
            api_key=args.gpt_model_add_key
        )
        messages_add = [
            {"role": "user", "content": args.gpt_model_add_prompt + " \n " + content},
        ]
        response_add = client_add.chat.completions.create(
                model=args.gpt_model_add,
                #max_tokens=3,
                messages=messages_add,
            )
        content = response_add.choices[0].message.content
        print("short model " + content)
        print("----------------------")
        content = float(content)
        #messages.append({"role": "user", "content": "Your answers must be selected from this list:[0.0,0.1,0.2,0.3,0.4,0.5,0.5,0.6,0.7,0.8,0.9,1.0], no additional answers, and answers are limited to 3 tokens"})
    # if not content in [0.0,0.1,0.2,0.3,0.4,0.5,0.5,0.6,0.7,0.8,0.9,1.0]:
    #     raise Exception("The score is out of range. Trying again")
    return content,messages,temperature

@retry(stop_max_attempt_number=5, wait_fixed=30000)
def second_polling(args,client,content_message,temperature):
    print("retry is running")
    messages = content_message
    messages.append({"role": "user", "content": "The answer last time was wrong or not the candidate numbers:[0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0], please answer again, the other requirements are the same as last time"})
    try:
        time2 = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        print(time2)
        response = client.chat.completions.create(
                model=args.gpt_model_client,
                #max_tokens=3,
                temperature=temperature,
                messages=messages,
            )
        content = response.choices[0].message.content
        print("OK")

    except RateLimitError as e:
        print("The gpt is busy. Wait 30 seconds")
        time.sleep(30)
        #raise Exception("gpt is busy")
    try:
        content = float(content)
    except:
        print("----------------------")
        time2 = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        print(time2)
        client_add = OpenAI(
            base_url=args.gpt_model_add_url,
            api_key=args.gpt_model_add_key
        )
        messages_add = [
            {"role": "user", "content": args.gpt_model_add_prompt + " \n " + content},
        ]
        response_add = client_add.chat.completions.create(
            model=args.gpt_model_add,
            # max_tokens=3,
            messages=messages_add,
        )
        content = response_add.choices[0].message.content
        print("short model " + content)
        print("----------------------")
        content = float(content)
        #messages.append({"role": "user","content": "Your answers must be selected from this list:[0.0,0.1,0.2,0.3,0.4,0.5,0.5,0.6,0.7,0.8,0.9,1.0], no additional answers, and answers are limited to 3 tokens"})
    #if not content in [0.0,0.1,0.2,0.3,0.4,0.5,0.5,0.6,0.7,0.8,0.9,1.0]:
        #raise Exception("The score is out of range. Trying again")
    return content





def cal_result(args,
        model,
        metadatas,
        indiv_cap,
        comb,
        grade_results_json,
        indiv_cap_file,
        comb_cap_file
               ):
    #Part1-单项能力统计（含robu）
    #视觉
    indiv_cap_vision_score = indiv_cap["indiv_vision_cap"]
    indiv_cap_lag_score = indiv_cap["indiv_lag_cap"]
    indiv_cap_robu_score = indiv_cap["indiv_robu_cap"]
    indiv_cap_mark = indiv_cap["indiv_mark"]
    #清洗分数表
    indiv_cap_vision_score = {key: [] for key in indiv_cap_vision_score}
    indiv_cap_lag_score = {key: [] for key in indiv_cap_lag_score}
    indiv_cap_robu_score = {key: [] for key in indiv_cap_robu_score}
    weight_vision = {}
    weight_lag = {}
    indiv_cap_score = [indiv_cap_vision_score, indiv_cap_lag_score, indiv_cap_robu_score] #计分
    indiv_cap_vision_count = {}
    indiv_cap_lag_count = {}
    indiv_cap_robu_count = {}
    indiv_cap_count = [indiv_cap_vision_count, indiv_cap_lag_count, indiv_cap_robu_count, indiv_cap_mark]

    cap_total_list = []
    weight_total_list = []

    #遍历一次原始表
    for id,values in metadatas.items():
        flag = 0
        if "score_0" in grade_results_json[id]:
            flag = 1
            cap_vision = values["capability_vision"]
            score_list = grade_results_json[id]["score_0"]["grade"]
            for cap in cap_vision:
                indiv_cap_vision_score[cap].append(np.mean(score_list))
                if not cap in weight_vision:
                    weight_vision[cap] = []
                    indiv_cap_count[0][cap] = [0,0,0,0]
                weight_vision[cap].append(values["level"][0])
            cap_total_list.append(np.mean(score_list))
            weight_total_list.append(values["level"][0])
        if not values["capability_language"][0] == "" :
            cap_lag = values["capability_language"]
            score_list = grade_results_json[id]["score_1"]["grade"]
            for cap in cap_lag:
                indiv_cap_lag_score[cap].append(np.mean(score_list))
                if not cap in weight_lag:
                    weight_lag[cap] = []
                    indiv_cap_count[1][cap] = [0,0,0,0]
                weight_lag[cap].append(values["level"][len(values["level"])-1])
            cap_total_list.append(np.mean(score_list))
            weight_total_list.append(values["level"][len(values["level"])-1])
            if flag == 0:
                cap_vision = values["capability_vision"]
                #score_list = grade_results_json[id]["score_1"]["grade"]
                for cap in cap_vision:
                    indiv_cap_vision_score[cap].append(np.mean(score_list))
                    if not cap in weight_vision:
                        weight_vision[cap] = []
                        indiv_cap_count[0][cap] = [0, 0, 0, 0]
                    weight_vision[cap].append(values["level"][0])
        if "capability_robu" in values:
            cap_robu = values["capability_robu"]
            if not cap_robu[0] in indiv_cap_robu_count:
                indiv_cap_robu_count[cap_robu[0]] = [0,0,0,0]
            for cap in cap_robu:
                for i in range(3):
                    score_list =grade_results_json[id]["score_2"][f"robu_{i}"]["grade"]
                    indiv_cap_robu_score[cap].append(np.mean(score_list))
    #计算单项平均分,加权方差，方差;总的分数转序列


    for i in range(len(indiv_cap_count)-1):
        tmp_cap = []
        tmp_weight = []
        for id,score in indiv_cap_count[i].items():
            score = indiv_cap_score[i][id] #分数序列
            if i == 0:
                indiv_cap_count[i][id][3] = np.mean(weight_vision[id])  #权重平均值
                indiv_cap_count[i][id][0] = np.mean(np.mean(score))  #平均数
                indiv_cap_count[i][id][1] = np.average(score,weights=weight_vision[id]) #加权平均数
                tmp_cap += score
                tmp_weight += weight_vision[id]
            elif i == 1:
                indiv_cap_count[i][id][3] = np.mean(weight_lag[id])  #权重平均值
                indiv_cap_count[i][id][0] = np.mean(np.mean(score))  #平均数
                indiv_cap_count[i][id][1] = np.average(score,weights=weight_lag[id]) #加权平均数
                tmp_cap += score
                tmp_weight += weight_lag[id]
            else:
                indiv_cap_count[i][id][0] = np.mean(score)
                indiv_cap_count[i][id][1] = np.mean(score)
                indiv_cap_count[i][id][3] = -1
                tmp_cap += score
            indiv_cap_count[i][id][2] = np.var(score) #方差
            #i[id][2] = np.std(score) #标准差舍弃不用
        if i == 0:
            indiv_cap_count[3]["vision"] = [np.mean(tmp_cap),np.average(tmp_cap, weights=tmp_weight),np.var(tmp_cap),np.mean(tmp_weight)]
        elif i == 1:
            indiv_cap_count[3]["lag"] = [np.mean(tmp_cap),np.average(tmp_cap, weights=tmp_weight),np.var(tmp_cap),np.mean(tmp_weight)]
        elif i == 2:
            indiv_cap_count[3]["robu"] = [np.mean(tmp_cap), -1 , np.var(tmp_cap),-1]
    #单项总计平均分,加权方差，方差 计算
    indiv_cap_count[3]["total_not_robu"] = [np.mean(cap_total_list),np.average(cap_total_list,weights=weight_total_list),np.var(cap_total_list), np.mean(weight_total_list)]
    #如上已写入列表------indiv_cap_count------#indiv_cap = [indiv_vision_cap,indiv_lag_cap,indiv_robu_cap,indiv_mark]
    #indiv_cap字典将仅包含数量

    #单项能力统计（含robu）-------end
    #-------------------------------------------------------------------------------------------------------------------------------------------------------------------#
    #Part2-复合能力统计
    comb_cap = comb
    comb_cap["comb_cap"] = {key: [] for key in comb_cap["comb_cap"]}
    tmp_comb_list = []
    comb_count = []
    for id,values in metadatas.items():
        flag = "0"
        tmp_comb_list = values["capability_vision"]
        if not values["capability_language"][0] == "" :
            tmp_comb_list += values["capability_language"]
            flag = "1"
        for cap in comb_cap["comb_cap"]:  #轮询cap
            if set(cap.split("_")).issubset(set(tmp_comb_list)):
                score_list = grade_results_json[id][f"score_{flag}"]["grade"]
                comb_cap["comb_cap"][cap].append(np.mean(score_list))
                comb_count.append(np.mean(score_list))
    for i,j in comb_cap["comb_cap"].items():
        tmp_cap = comb_cap["comb_cap"][i]
        comb_cap["comb_cap"][i] = [0,0]
        comb_cap["comb_cap"][i][0] = np.mean(tmp_cap)
        comb_cap["comb_cap"][i][1] = np.var(tmp_cap)
    comb_cap["comb_mark"][0] = np.mean(comb_count)
    comb_cap["comb_mark"][1] = np.var(comb_count)
    #如上已写入字典comb_cap，另外说明，comb将仅包含数量而非分数
    #复合能力统计---------------end
    #-------------------------------------------------------------------------------------------------------------------------------------------------------------------------
    #robu多角度统计
    #回答重复次数
    with open(args.result_file,encoding='UTF-8') as f:
        results = json.load(f)
    compare_count = [0,0,0] #yes,no,retry count
    for id,values in results.items():
        for idm,mark in values.items():
            if not idm == "mark_2":
                compare_count[0] += mark["compare"].count("yes")
                compare_count[1] += mark["compare"].count("no")
                if mark["compare"][0] == "no":
                    compare_count[2] += 1
            else:
                for idm2,markm2 in mark.items():
                    compare_count[0] += markm2["compare"].count("yes")
                    compare_count[1] += markm2["compare"].count("no")
                    if markm2["compare"][0] == "no":
                        compare_count[2] += 1
    #打分重复次数
    grade_count = [0,0,0]  #递归计数，打分次数，失败次数
    for id,values in grade_results_json.items():
          for idm,mark in values.items():
              if not idm == "score_2":
                  grade_count[1] += 1
                  if not mark["self-reflection"] == -1:
                      grade_count[0] += mark["self-reflection"]
                  else:
                      grade_count[2] += 1
              else:
                  for idm2,mark2 in mark.items():
                      if not mark2["self-reflection"] == -1:
                          grade_count[0] += mark2["self-reflection"]
                      else:
                          grade_count[2] += 1
    #题内问答分数
    #indiv_cap_score[2] 为两个项目各自的分 结构如下：[平均分，占位符，方差]
    #indiv_cap_score[3]["robu"] 为总的分 结构如下：[平均分，占位符，方差]
    #--------------------------------------------------------------------------------------------------------------------------------------------------------------------------
    #格式化、写入文件
    score_of_robu = (2 * (grade_count[0] / grade_count[1]) + 1 * (
                compare_count[0] / (compare_count[0] + compare_count[1])) + 7 * indiv_cap_count[3]["robu"][0]) / 10
    total_score = (9 * indiv_cap_count[3]["total_not_robu"][0] + 1 * score_of_robu) / 10
    a = {"class": ["average score", "weighted variance", "variance", "sample size"],
         "ocr": indiv_cap_count[0]["ocr"],
         "vi": indiv_cap_count[0]["vi"],
         "space": indiv_cap_count[0]["space"],
         "motion": indiv_cap_count[0]["motion"],
         "background": indiv_cap_count[0]["background"],
         "total_vision": indiv_cap_count[3]["vision"],
         "common": indiv_cap_count[1]["common"],
         "generation": indiv_cap_count[1]["generation"],
         "math": indiv_cap_count[1]["math"],
         "inference": indiv_cap_count[1]["inference"],
         "total_language": indiv_cap_count[3]["lag"],
         "total_all_not_robu": indiv_cap_count[3]["total_not_robu"],
         " ": ["|", "|", "|", "|"],
         "robu_index": ["average score", " ", "variance", " "],
         "hallucination": indiv_cap_count[2]["hallucination"],
         "input": indiv_cap_count[2]["input"],
         "total_robu": indiv_cap_count[3]["robu"],
         "Response_index": ["Response retry ratio", "Sample count", " ", " "],
         "Response robu": [(compare_count[0] / (compare_count[0] + compare_count[1])),
                           compare_count[0] + compare_count[1], " ", " "],
         "retry_index": ["Average retry", "Number of scores", "failure rate", " "],
         "Grade robu": [(grade_count[0] / grade_count[1]), grade_count[1], grade_count[2] / grade_count[1], " "],
         "Comprehensive score of robustness": [score_of_robu, " ", " ", " "],
         "Model total score": [total_score, " ", " ", " "]
         }
    for i, j in a.items():
        print(i)
        print(len(j))
    dataframe_1 = pd.DataFrame(
        {"class": ["average score", "weighted variance", "variance", "sample size"],
         "ocr": indiv_cap_count[0]["ocr"],
         "vi": indiv_cap_count[0]["vi"],
         "space": indiv_cap_count[0]["space"],
         "motion": indiv_cap_count[0]["motion"],
         "background": indiv_cap_count[0]["background"],
         "total_vision": indiv_cap_count[3]["vision"],
         "common": indiv_cap_count[1]["common"],
         "generation": indiv_cap_count[1]["generation"],
         "math": indiv_cap_count[1]["math"],
         "inference": indiv_cap_count[1]["inference"],
         "total_language": indiv_cap_count[3]["lag"],
         "total_all_not_robu": indiv_cap_count[3]["total_not_robu"],
         " ": ["|", "|", "|", "|"],
         "robu_index": ["average score", " ", "variance", " "],
         "hallucination": indiv_cap_count[2]["hallucination"],
         "input": indiv_cap_count[2]["input"],
         "total_robu": indiv_cap_count[3]["robu"],
         "Response_index": ["Response retry ratio", "Sample count", " ", " "],
         "Response robu": [(compare_count[0] / (compare_count[0] + compare_count[1])),
                           compare_count[0] + compare_count[1], " ", " "],
         "retry_index": ["Average retry", "Number of scores", "failure rate", " "],
         "Grade robu": [(grade_count[0] / grade_count[1]), grade_count[1], grade_count[2] / grade_count[1], " "],
         "Comprehensive score of robustness": [score_of_robu, " ", " ", " "],
         "Model total score": [total_score, " ", " ", " "]
         }
    )

    tmp_comb = {}
    for idm, values in comb_cap["comb_cap"].items():
        # tmp_comb["_".join(set(idm))] = values
        tmp_comb[idm] = values
    tmp_comb["total_comb"] = comb_cap["comb_mark"]
    dataframe_2 = pd.DataFrame(tmp_comb, index=["average score", "variance"])

    dataframe_1.to_csv(indiv_cap_file)
    dataframe_2.to_csv(comb_cap_file)

    return dataframe_1, dataframe_2


if __name__ == "__main__":
    args = arg_parser(model="model") #Enter the name of the model you want to test
    if args.openai_api_key:
        OPENAI_API_KEY = args.openai_api_key
    else:
        OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')
    client = OpenAI(
        base_url = args.openai_api_base_url,
        api_key=OPENAI_API_KEY
    )

    if os.path.exists(args.result_file) is False:
        raise ValueError("Result file does not exist")
    if not args.result_file.endswith(('.json', '.JSON')):
        raise ValueError("Result file should be a json file")
    #Get model name
    model = pathlib.Path(args.result_file).stem

    #能力统计
    metadatas = load_metadata(args)
    (
        metadatas,
        indiv_cap,
        comb,
        len_data
    ) = metadatas

    #打分结果输出文件设置
    file = get_grade_file(args, model)
    (
        grade_file_json,
        indiv_cap_file,
        comb_cap_file
    ) = file


    grade_results_json = runs(
        args,
        grade_file_json,
        metadatas,
        len_data,
        client
    )

    #遍历，计算总分数，导出csv
    dataframe_1,dataframe_2 = cal_result(
        args,
        model,
        metadatas,
        indiv_cap,
        comb,
        grade_results_json,
        indiv_cap_file,
        comb_cap_file
    )
    print("---------------------------------------------------------------------------------------")
    print(f"\033[1m{model}\033[0m")
    print("\n")
    print("Below are individual capability scores, robustness analysis, and overall scores")
    print(dataframe_1)
    print("\n")
    print("The following are composite ability scores")
    print(dataframe_2)

