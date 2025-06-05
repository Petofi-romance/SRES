import os
import json
import base64


from lib.compare_hash import hashdis, hashsim
from lib.compare_llm import compare, com_str,robu
from lib.compare_llm import com_sort
from lib.meta_prompt import arg_meta


# Function to encode the image
def encode_image(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')


def evaluate_on_sres(args, model, count):
    argm = arg_meta()  # prompt set

    if os.path.exists(args.result_path) is False:
        os.makedirs(args.result_path)
    results_path = os.path.join(args.result_path, f"{args.model_name}.json")
    image_folder = os.path.join(args.SRES_path, "images")

    meta_data = os.path.join(args.SRES_path, "SRES.json")
    with open(meta_data, 'r', encoding='utf-8') as f:
        data = json.load(f)

    if os.path.exists(results_path):
        with open(results_path, 'r',encoding='UTF-8') as f:
            results = json.load(f)
    else:
        results = {}

    for i in range(count):
        id = f"v1_{i}"
        if id in results:
            continue
        imagename = data[id]['imagename']
        img_path = os.path.join(image_folder, imagename)
        prompt = data[id]['question']
        degree = data[id]['degree']

        main_question = data[id]['capability_language'][0]

        print(f"\n \033[1;31;46m {id} \033[0m")
        print(f"Image: {imagename}")

        #------------问题轮询---------#
        answer = ''
        rps = 'empty'
        message_tem = {}
        for j in range(len(degree)):

            if j == 0:
                results[id] = {}
            print(j,degree[j])
            result,answer,rps,message_tem = run(model,img_path,prompt[j],message_tem,rps,main_question,degree[j],j)
            results[id][f"mark_{degree[j]}"] = result

        with open(results_path, 'w',encoding='UTF-8') as f:
            json.dump(results, f, indent=4)
            print(id + "json input is ok")
            print("********************************************")



def run(model,img_path,prompt,message_tem,rps,main_question:str,count:int = 0,j:int = 0):
    print(f"Question_{count+1}--------------------------------------------------------------------------------------------------")
    if count == 0:
        #丢弃message_tem,rps,main_question；
        return run_level_0(model,img_path,prompt)
    elif count == 1:
        return run_level_1(model,img_path,prompt,message_tem,rps,main_question)
    elif count == 2:
        return run_level_2(model,img_path,prompt[j],main_question)
    else:print("xxxxxxxxxxx")

def run_level_0(model,img_path,prompt):
    answer1, answer2, response, message_tem = compare(model, img_path, prompt)
    answer1, answer2 = com_str(answer1, answer2)
    print_res(response,answer1,answer2)
    #-----------写入----------
    result = Wr_result(answer1,answer2,response)

    return result,answer2,response[0],message_tem
def run_level_1(model,img_path,prompt,message_tem,rps,main_question):
    answer1, answer2, response, message_tem = compare(model, img_path, prompt,message_tem,rps,main_question)
    answer1, answer2 = com_str(answer1, answer2)
    print_res(response,answer1,answer2)
    #-----------写入----------
    result = Wr_result(answer1,answer2,response)

    return result,answer2,response[0],message_tem
def run_level_2(model,img_path,prompt,main_question):

    #---------robu数据结构--------
    result = [{},{},{}]
    for i in range(3):
        print(f"question robu {i} is testing")
        answer1, answer2, response, = robu(model,img_path,prompt[i])
        answer1, answer2 = com_str(answer1, answer2)
        print_res(response, answer1, answer2)
        result[i] = Wr_result(answer1, answer2, response)

    print(f"Question robu is writing")
    results = {
        "robu_0":result[0],
        "robu_1":result[1],
        "robu_2":result[2]
    }

    return results,'','',''

def Wr_result(answer1,answer2,response):
    if answer1 == "yes":
        result = {
                "compare":[answer1,answer2],
                "response1":response[0],
                "response2":response[1],
            }
    else:
        result = {
                "compare":[answer1,answer2],
                "response1":response[0],
                "response2":response[1],
                "response3":response[2]
            }
    return result

def print_res(response,answer1,answer2):
    print(hashsim(response[0], response[1]), " ~ ", hashdis(response[0], response[1]))
    print("compare is " + f"\033[32m{answer1}\033[0m" + " ~ " + f"\033[31m{answer2}\033[0m")  # 结果标识
    print("Note: If the comparison result is 'no ~ yes', 'response1' will be the response1 reflection output")
    print("----------response-----------")
    if len(response) > 2:
        com_sort(answer2, response)
        print(f"\033[1;32;46m Response_1:\033[0m  \033[32m{response[0]}\033[0m")
        print(f"\033[1;32;46m Response_2:\033[0m  \033[33m{response[1]}\033[0m")
        print(f"\033[1;32;46m Response_2:\033[0m  \033[31m{response[2]}\033[0m")
    else:
        print(f"\033[1;32;46m Response_1:\033[0m  \033[32m{response[0]}\033[0m")
        print(f"\033[1;32;46m Response_2:\033[0m  \033[33m{response[1]}\033[0m")