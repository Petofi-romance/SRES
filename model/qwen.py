
import argparse
from openai import OpenAI
from retrying import retry

from lib.utils_m2 import evaluate_on_sres, encode_image 

# prepare the model
model = 'Model to be tested'
savename = 'The name you want to store'  #The name change needs to be consistent with Evaluate_onSRES
client = OpenAI(
    base_url='your url',
    api_key='your api key',
    model=model,
)


class Model:
    def __init__(self):
        self.model = model

    @retry(stop_max_attempt_number=3, wait_fixed=30000)
    def get_response(self, image_path, prompt, meta_prompt):
        base64_image = encode_image(image_path)
        messages = []
        content = [
            {
                "type": "image_url",
                "image_url": {
                    "url": f"data:image/png;base64,{base64_image}"
                },
            },
            {
                #"meta_prompt": "You are a debater. Welcome to the visual question-answering competition. You can agree or disagree with others' viewpoints, aiming to find the correct answer. Criteria for a good answer:\n1. Align with the content in the image.\n2. Only output content visible in the image.\n3. Distinguish nuances between synonyms accurately.\n Please do not repeat the question.",
                "meta_prompt": meta_prompt[0],
                "type": "text", "text": prompt + meta_prompt[1]
            }
        ]

        messages.append({
            "role": "user",
            "content": content,
        })
        print(f"{model} is start")
        response = client.chat.completions.create(
            model=self.model,
            messages=messages
        )


        rps_m1 = response.choices[0].message.content
        return rps_m1, messages

    @retry(stop_max_attempt_number=3, wait_fixed=30000)
    def get_response_add(self, messages, rps, prompt, meta_prompt):
        # ------add history response---------
        message_tem = {
            "role": "assistant",
            "content": rps,
        }
        messages.append(message_tem)
        # ------add history response---------
        content = [
            {
                # "meta_prompt": "You are a debater. Welcome to the visual question-answering competition. You can agree or disagree with others' viewpoints, aiming to find the correct answer. Criteria for a good answer:\n1. Align with the content in the image.\n2. Only output content visible in the image.\n3. Distinguish nuances between synonyms accurately.\n Please do not repeat the question.",
                "meta_prompt": meta_prompt[0],
                "type": "text",
                "text": prompt + meta_prompt[1]
            }
        ]

        messages.append({
            "role": "user",
            "content": content,
        })
        print(f"{model} is retry")
        response = client.chat.completions.create(
            model=self.model,
            messages=messages
        )

        rps_m2 = response.choices[0].message.content
        return rps_m2, messages


def arg_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--SRES_path",
        type=str,
        default="../SRES-vet",
        help="Evaluation dataset",
    )
    parser.add_argument(
        "--result_path",
        type=str,
        default="../model/results",
    )
    parser.add_argument(
        "--model_name",
        type=str,
        default=savename,
        help="model name",
    )

    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = arg_parser()
    model = Model()
    evaluate_on_sres(args, model, 181)

