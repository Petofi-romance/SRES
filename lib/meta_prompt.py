import argparse


def arg_meta():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--describe",
        type=str,
        default="You are an AI assistant undergoing testing, now please answer the questions based on the questions and pictures given. A good answer has the following characteristics: \n 1. The answer is concise, there is no specific solution process, and only the final answer \n 2 is given. Only reason based on what is visible in the image \n 3. Accurately distinguish subtle differences between synonyms Please do not repeat this question.",
        help="describe mate",
    )
    parser.add_argument(
        "--math",
        type=str,
        default="There is an math exam taking place and you need to give answers to questions, the standard answer is: describe the final answer only, do not explain the process,do not include formulas in the results, and the output is as concise as possible.If there are more than one answer, connect with <AND>",
        help="math mate",
    )
    parser.add_argument(
        "--math_add",
        type=str,
        default="Please give the final answer directly without describing the calculation.",
    )
    parser.add_argument(
        "--common",
        type=str,
        default="You are an AI assistant undergoing testing, now please answer the questions based on the questions and pictures given. A good answer has the following characteristics: 1. The answer is concise, there is no specific solution process, and only the final answer /n 2 is given. Only reason based on what is visible in the image /n 3. Accurately distinguish subtle differences between synonyms Please do not repeat this question.",
        help="common mate",
    )
    parser.add_argument(
        "--common_add",
        type=str,
        default="Please answer the question as succinctly as possible.Don't give the process.Do not repeat the question.Please do not answer anything beyond the answer.",
        help="common mate",
    )

    args = parser.parse_args()
    return args

def argm_set(id = "math"):
    argm = arg_meta()
    count = ['1','2']
    if id == "level one":
        count[0] = argm.describe
        count[1] = argm.common_add
    elif id == "math" :
        count[0] = argm.math
        count[1] = argm.math_add
    else:
        count[0] = argm.common
        count[1] = argm.common_add

    return count

