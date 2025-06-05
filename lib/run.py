from lib.compare_hash import hashsim, hashdis
from lib.compare_llm import compare, com_sort,com_str


def run_m1(model,img_path,prompt,results):
    answer1, answer2, response = compare(model, img_path, prompt)
    answer1, answer2 = com_str(answer1, answer2)
    print("------question 2---------------------------------------------")
    print(hashsim(response[0], response[1]), " ~ ", hashdis(response[0], response[1]))
    print("compare is " + f"\033[32m{answer1}\033[0m" + " ~ " + f"\033[31m{answer2}\033[0m")  # 结果标识
    print("Note: If the comparison result is 'no ~ yes', 'response1' will be the response1 reflection output")
    print("----------response 1-----------")
    print(f"\033[1;32;46m Response_1:\033[0m  \033[32m{response[0]}\033[0m")
    print(f"\033[1;32;46m Response_2:\033[0m  \033[33m{response[1]}\033[0m")
    if len(response) > 2:
        com_sort(answer2, response)
        print(f"\033[1;32;46m Response_2:\033[0m  \033[31m{response[2]}\033[0m")

    for i in range(1, len(response)):
        if i == 0:
            results[id + "_m1"] = {answer1, answer2, response[0]}
        else:
            results[id + f"_m1_rep{i}"] = response[i]

def run_m2(model,img_path,prompt,results):
    answer1, answer2, response = compare(model, img_path, prompt)
    answer1, answer2 = com_str(answer1, answer2)
    print("------question 2---------------------------------------------")
    print(hashsim(response[0], response[1]), " ~ ", hashdis(response[0], response[1]))
    print("compare is " + f"\033[32m{answer1}\033[0m" + " ~ " + f"\033[31m{answer2}\033[0m")  # 结果标识
    print("Note: If the comparison result is 'no ~ yes', 'response1' will be the response1 reflection output")
    print("----------response 2-----------")
    print(f"\033[1;32;46m Response_1:\033[0m  \033[32m{response[0]}\033[0m")
    print(f"\033[1;32;46m Response_2:\033[0m  \033[33m{response[1]}\033[0m")
    if len(response) > 2:
        com_sort(answer2, response)
        print(f"\033[1;32;46m Response_2:\033[0m  \033[31m{response[2]}\033[0m")

    for i in range(1, len(response)):
        if i == 0:
            results[id + "_m2"] = {answer1, answer2, response[0]}
        else:
            results[id + f"_m2_rep{i}"] = response[i]

def run_m3(model,img_path,prompt,results):
    answer1, answer2, response = compare(model, img_path, prompt)
    answer1, answer2 = com_str(answer1, answer2)
    print("---robustness ~ question 3---------------------------------------------")
    print(hashsim(response[0], response[1]), " ~ ", hashdis(response[0], response[1]))
    print("compare is " + f"\033[32m{answer1}\033[0m" + " ~ " + f"\033[31m{answer2}\033[0m")  # 结果标识
    print("Note: If the comparison result is 'no ~ yes', 'response1' will be the response1 reflection output")
    print("----------response 3-----------")
    print(f"\033[1;32;46m Response_1:\033[0m  \033[32m{response[0]}\033[0m")
    print(f"\033[1;32;46m Response_2:\033[0m  \033[33m{response[1]}\033[0m")
    if len(response) > 2:
        com_sort(answer2, response)
        print(f"\033[1;32;46m Response_2:\033[0m  \033[31m{response[2]}\033[0m")

    for i in range(1, len(response)):
        if i == 0:
            results[id + "_m3"] = {answer1, answer2, response[0]}
        else:
            results[id + f"_m3_rep{i}"] = response[i]

