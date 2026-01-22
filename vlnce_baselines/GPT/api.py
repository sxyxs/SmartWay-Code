from openai import OpenAI
import openai
import base64
from tenacity import (
    retry,
    stop_after_attempt,
    wait_random_exponential,
)  # for exponential backoff
import torch
import numpy as np
from PIL import Image
import io

# import os 
# openai.api_key  = "SETYOURKEY"  # GPT key

generation_key = "SETYOURKEY"  # GPT key
client = OpenAI(
    api_key=generation_key,
)


@retry(wait=wait_random_exponential(min=1, max=60), stop=stop_after_attempt(6))
def completion_with_backoff(**kwargs):
    return client.chat.completions.create(**kwargs)

def encode_torch_rgb_to_base64(torch_tensor):
    if torch_tensor.is_cuda:
        torch_tensor = torch_tensor.cpu()
    
    if torch_tensor.max() <= 1.0:
        torch_tensor = (torch_tensor * 255).byte()

    #  (224, 224, 3) 
    np_array = torch_tensor.numpy()

    image = Image.fromarray(np_array)

    buffer = io.BytesIO()
    image.save(buffer, format="JPEG")
    buffer.seek(0)

    image_base64 = base64.b64encode(buffer.read()).decode('utf-8')
    
    return image_base64


def gpt_infer_back_track(system, text, image_list, model="gpt-4o", max_tokens=1000, response_format=None,step=None,num_epi=None,semantic_map=None,exp_name=None,img_id=None):

    user_content = []
    if step==0:
        for i, image in enumerate(image_list, start=1):
            if image is not None:
                image_base64 = encode_torch_rgb_to_base64(image)
                user_content.append(
                    {
                        "type": "text",
                        "text": f"Image {i}:"
                    },
                )

                # with open(image, "rb") as image_file:
                #     image_base64 = base64.b64encode(image_file.read()).decode('utf-8')

                image_message = {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/jpeg;base64,{image_base64}",
                            "detail": "low"
                        }
                    }
                user_content.append(image_message)
    else:
        for i, image in enumerate(image_list):
            
            if image is not None:
                image_base64 = encode_torch_rgb_to_base64(image)
                user_content.append(
                    {
                        "type": "text",
                        "text": f"Image {img_id[i]}:"
                    },
                )

                # with open(image, "rb") as image_file:
                #     image_base64 = base64.b64encode(image_file.read()).decode('utf-8')

                image_message = {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/jpeg;base64,{image_base64}",
                            "detail": "low"
                        }
                    }
                user_content.append(image_message)

    user_content.append(
        {
            "type": "text",
            "text": text
        }
    )

    messages = [
        {"role": "system",
         "content": system
         },
        {"role": "user",
         "content": user_content
         }
    ]

    if response_format:
        chat_message = completion_with_backoff(model=model, messages=messages, temperature=0, max_tokens=max_tokens, response_format=response_format)
    else:
        chat_message = completion_with_backoff(model=model, messages=messages, temperature=0, max_tokens=max_tokens)

    # print(chat_message)
    answer = chat_message.choices[0].message.content
    tokens = chat_message.usage
    if num_epi !=None:
        import torch
        import os
        from PIL import Image

        if exp_name is not None:
            save_dir = f"rgb_images/{exp_name}/episode_{num_epi}/step_{step}"
        else:
            save_dir = f"rgb_images/episode_{num_epi}/step_{step}"
        os.makedirs(save_dir, exist_ok=True)


        image_list = (image_list).byte().cpu().numpy()


        for i in range(image_list.shape[0]):
            img = Image.fromarray(image_list[i])
            img.save(os.path.join(save_dir, f"image_{i}.png"))
            
        print(f"Saved {image_list.shape[0]} images to {save_dir}")
        return answer, tokens

