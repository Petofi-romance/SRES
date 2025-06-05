from openai import OpenAI

from lib.compare_hash import hashdis, hashsim
from lib.meta_prompt import  argm_set

client = OpenAI(
    # 若没有配置环境变量，请用百炼API Key将下行替换为：api_key="sk-xxx",
    api_key="your api key",
    # 若没有配置环境变量，请用百炼API Base URL将下行替换为：
    base_url="your url",
    model=" your model name"  # 例如 "Qwen/Qwen2-VL-72B-Instruct",
)

class Duibi:
    def __init__(self,model):
        self.model = model

    def get_response1(self, prompt1,prompt2):
        messages = []
        content = [
            {
                # "meta_prompt": "You are a debater. Welcome to the visual question-answering competition. You can agree or disagree with others' viewpoints, aiming to find the correct answer. Criteria for a good answer:\n1. Align with the content in the image.\n2. Only output content visible in the image.\n3. Distinguish nuances between synonyms accurately.\n Please do not repeat the question.",
                "type": "text",
                "text": "You are a teacher, there are two students submitted homework, please compare their answers are the same, if the same please output 1, different please output 0, other than do not reply other content."
                        "Please compare the two answers and answer whether the content is the same"
                        " | " + "The first student's answer is : " + prompt1 + " | " + "The second student's answer is : " + prompt2

            }
        ]

        messages.append({
            "role": "user",
            "content": content,
        })

        response = client.chat.completions.create(
            model=self.model,
            messages=messages
        )

        rps = response.choices[0].message.content
        return rps

    def get_response2(self, prompt1, prompt2,prompt3):
        messages = []
        content = [
            {
                # "meta_prompt": "You are a debater. Welcome to the visual question-answering competition. You can agree or disagree with others' viewpoints, aiming to find the correct answer. Criteria for a good answer:\n1. Align with the content in the image.\n2. Only output content visible in the image.\n3. Distinguish nuances between synonyms accurately.\n Please do not repeat the question.",
                "type": "text",
                "text": "You are a teacher, there are four students submitted homework, please compare their answers are the same. If at least two answers are the same,please output 1. Otherwise, pleaseoutput 0, other than do not reply other content."
                        " | " + "The first student's answer is : " + prompt1 +
                        " | " + "The second student's answer is : " + prompt2 +
                        " | " + "The third student's answer is : " + prompt3


            }
        ]

        messages.append({
            "role": "user",
            "content": content,
        })

        response = client.chat.completions.create(
            model=self.model,
            messages=messages
        )

        rps = response.choices[0].message.content
        return rps

def compare(model,img_path, prompt,message_tem = {},rps = 'empty',main_question = "level one"):
    response = []
    #------------选择元提示-------------#
    meta_prompt = argm_set(main_question)
    #-------------两次访问------------#
    if rps != 'empty':  #----------嵌套提问，问题追加
        res_tem,message_tem = model.get_response_add(message_tem,rps, prompt, meta_prompt)
        response.append(res_tem)
        message1 = message_tem
        res_tem,message_tem = model.get_response_add(message_tem,rps, prompt, meta_prompt)
        response.append(res_tem)
        message2 = message_tem #message2留作以后拓展用
    else:
        res_tem,message_tem = model.get_response(img_path, prompt, meta_prompt)
        response.append(res_tem)
        message1 = message_tem
        res_tem,message_tem = model.get_response(img_path, prompt, meta_prompt)
        response.append(res_tem)
        message2 = message_tem #message2留作以后拓展用

    #-------------对比------------#
    duibi = Duibi()
    # 如果两次回答相同，直接返回0
    if response[0] == response[1]:
        answer1 = '0'
        answer2 = answer1
        com_sort(answer2, response)
        return answer1, answer2, response, message1
    elif hashsim(response[0], response[1]) < 1 and hashdis(response[0], response[1]) > 0.9:
        # 如果两次回答相似度大于0.9，海明距离小于1，认为是相同的
        answer1 = '0'
        answer2 = answer1
        com_sort(answer2, response)
        return answer1, answer2, response, message1   
    
    # 如果两次回答不同，使用对比函数获取结果
    if response[0] == response[1]:
        answer1 = '0'
        answer2 = answer1
    else:
        answer1 = duibi.get_response1(response[0], response[1])
        answer2 = answer1

    #-------------结果判断------------#
    if answer1 != '1':
        prompt_again = "The answer last time was wrong, please answer again, the other requirements are the same as last time"
        meta_prompt = ['Please refer to the previous input to answer the following questions','']
        res_tem,message1 =model.get_response_add(message1,response[0],prompt_again,meta_prompt)
        response.append(res_tem)

        answer2 = duibi.get_response2(response[0], response[1],response[2])

        com_sort(answer2, response)
        if answer2 != '1':
            answer2 = '0'
        else :
            answer2 = '1'


    return answer1,answer2,response,message1


def robu(model,img_path, prompt):
    response = []
    meta_prompt = ['','']

    res_tem,message_tem = model.get_response(img_path, prompt, meta_prompt)
    response.append(res_tem)
    message1 = message_tem
    res_tem,message_tem = model.get_response(img_path, prompt, meta_prompt)
    response.append(res_tem)
    message2 = message_tem #message2留作以后拓展用
    #------对比校验-------
    duibi = Duibi()
    answer1 = duibi.get_response1(response[0], response[1])
    answer2 = answer1
    #---------二次校验--------
    if answer1 != '1':
        prompt_again = "The answer last time was wrong, please answer again, the other requirements are the same as last time"
        meta_prompt = ['Please refer to the previous input to answer the following questions','']
        res_tem,message1 =model.get_response_add(message1,response[0],prompt_again,meta_prompt)
        response.append(res_tem)

        answer2 = duibi.get_response2(response[0], response[1],response[2])

        if answer2 != '1':
            answer2 = '0'
        else :
            answer2 = '1'
            com_sort(answer2, response)
    return answer1, answer2, response


def com_str(answer1,answer2):
    if answer1 == '1':
        answer1 = "yes"
    else:
        answer1 = "no"
    if answer2 == '1':
        answer2 = "yes"
    else:
        answer2 = "no"
    return answer1, answer2

def com_sort(answer2,response):
    if answer2 == '1':
        res = response[2]
        response[2] = response[0]
        response[0] = res

if __name__ == "__main__":
    anwer = compare("aholifhaosihnfaso","1")
    print(anwer)
