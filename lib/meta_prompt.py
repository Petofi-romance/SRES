import argparse


def arg_meta():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--describe",
        type=str,
        default="You are a debater. Welcome to the visual question-answering competition. You can agree or disagree with others' viewpoints, aiming to find the correct answer. Criteria for a good answer:\n1. Align with the content in the image.\n2. Only output content visible in the image.\n3. Distinguish nuances between synonyms accurately.\nIf there are more than one answer, connect with the '<AND>' symbol.\n Please do not repeat the question.",
        help="describe mate",
    )
    parser.add_argument(
        "--math",
        type=str,
        default="There is an math exam taking place and you need to give answers to questions, the standard answer is: describe the final answer only, do not explain the process,do not include formulas in the results, and the output is as concise as possible",
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
        default="You are a candidate, now taking a test, please use relatively concise language to describe the answer to the following question, a good answer has the following characteristics: \n1. Align with the content in the image; \n2. Output only what is visible in the image; 3. Accurately distinguish subtle differences between synonyms.4.If there are more than one answer, connect with the '<AND>' symbol. \n Please don't repeat the question.",
        help="common mate",
    )
    parser.add_argument(
        "--common_add",
        type=str,
        default="Please answer the question as succinctly as possible.",
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

