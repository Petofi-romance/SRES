"""
Usage:
python qwen.py --ssr-vles_path /path/to/ssr-vles --dashscope_api_key
"""
import argparse
from openai import OpenAI
from retrying import retry

from lib.utils_m2 import evaluate_on_ssrvet, encode_image

# prepare the model


model = 'model'
savename = 'model'
client = OpenAI(
    base_url='url',
    api_key='key'
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

        response = client.chat.completions.create(
            model=self.model,
            messages=messages
        )


        rps_m1 = response.choices[0].message.content
        return rps_m1, messages

    @retry(stop_max_attempt_number=3, wait_fixed=30000)
    def get_response_add(self, messages, rps, prompt, meta_prompt):
        # ------add old response---------
        message_tem = {
            "role": "assistant",
            "content": rps,
        }
        messages.append(message_tem)
        # ------add old response---------
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

        response = client.chat.completions.create(
            model=self.model,
            messages=messages
        )

        rps_m2 = response.choices[0].message.content
        return rps_m2, messages


def arg_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--ssrvet_path",
        type=str,
        default="../ssr-vles",
        help="Download ssr-vles.zip and `unzip ssr-vles.zip` and change the path here",
    )
    parser.add_argument(
        "--result_path",
        type=str,
        default="results",
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

    evaluate_on_ssrvet(args, model, 181)

