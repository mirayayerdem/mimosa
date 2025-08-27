import torch
from transformers import AutoTokenizer, AutoModel
import requests
from PIL import Image
from transformers import MllamaForConditionalGeneration, AutoProcessor
import pandas as pd
import math
import numpy as np
import torchvision.transforms as T
from torchvision.transforms.functional import InterpolationMode
import datetime
from transformers import  AutoModelForCausalLM
from transformers import Qwen2VLForConditionalGeneration, AutoTokenizer, AutoProcessor, LlavaForConditionalGeneration,InstructBlipProcessor, InstructBlipForConditionalGeneration
import base64
from openai import OpenAI, RateLimitError
from google.cloud import vision
import time

api_key = OpenAI(api_key=os.environ["OPENAI_API_KEY"])
client = OpenAI(api_key = api_key)
hf_token = os.getenv("HF_TOKEN")

from qwen_vl_utils import process_vision_info
path = "OpenGVLab/InternVL2_5-38B-MPO"

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

def build_transform(input_size):
    MEAN, STD = IMAGENET_MEAN, IMAGENET_STD
    transform = T.Compose([
        T.Lambda(lambda img: img.convert('RGB') if img.mode != 'RGB' else img),
        T.Resize((input_size, input_size), interpolation=InterpolationMode.BICUBIC),
        T.ToTensor(),
        T.Normalize(mean=MEAN, std=STD)
    ])
    return transform

def find_closest_aspect_ratio(aspect_ratio, target_ratios, width, height, image_size):
    best_ratio_diff = float('inf')
    best_ratio = (1, 1)
    area = width * height
    for ratio in target_ratios:
        target_aspect_ratio = ratio[0] / ratio[1]
        ratio_diff = abs(aspect_ratio - target_aspect_ratio)
        if ratio_diff < best_ratio_diff:
            best_ratio_diff = ratio_diff
            best_ratio = ratio
        elif ratio_diff == best_ratio_diff:
            if area > 0.5 * image_size * image_size * ratio[0] * ratio[1]:
                best_ratio = ratio
    return best_ratio

def dynamic_preprocess(image, min_num=1, max_num=12, image_size=448, use_thumbnail=False):
    orig_width, orig_height = image.size
    aspect_ratio = orig_width / orig_height

    # calculate the existing image aspect ratio
    target_ratios = set(
        (i, j) for n in range(min_num, max_num + 1) for i in range(1, n + 1) for j in range(1, n + 1) if
        i * j <= max_num and i * j >= min_num)
    target_ratios = sorted(target_ratios, key=lambda x: x[0] * x[1])

    # find the closest aspect ratio to the target
    target_aspect_ratio = find_closest_aspect_ratio(
        aspect_ratio, target_ratios, orig_width, orig_height, image_size)

    # calculate the target width and height
    target_width = image_size * target_aspect_ratio[0]
    target_height = image_size * target_aspect_ratio[1]
    blocks = target_aspect_ratio[0] * target_aspect_ratio[1]

    # resize the image
    resized_img = image.resize((target_width, target_height))
    processed_images = []
    for i in range(blocks):
        box = (
            (i % (target_width // image_size)) * image_size,
            (i // (target_width // image_size)) * image_size,
            ((i % (target_width // image_size)) + 1) * image_size,
            ((i // (target_width // image_size)) + 1) * image_size
        )
        # split the image
        split_img = resized_img.crop(box)
        processed_images.append(split_img)
    assert len(processed_images) == blocks
    if use_thumbnail and len(processed_images) != 1:
        thumbnail_img = image.resize((image_size, image_size))
        processed_images.append(thumbnail_img)
    return processed_images

def load_image(image_file, input_size=448, max_num=12):
    image = Image.open(image_file).convert('RGB')
    transform = build_transform(input_size=input_size)
    images = dynamic_preprocess(image, image_size=input_size, use_thumbnail=True, max_num=max_num)
    pixel_values = [transform(image) for image in images]
    pixel_values = torch.stack(pixel_values)
    return pixel_values

def split_model(model_name):
    device_map = {}
    world_size = torch.cuda.device_count()
    num_layers = {
        'InternVL2_5-1B': 24, 'InternVL2_5-2B': 24, 'InternVL2_5-4B': 36, 'InternVL2_5-8B': 32,
        'InternVL2_5-26B': 48, 'InternVL2_5-38B': 64, 'InternVL2_5-78B': 80}[model_name]
    # Since the first GPU will be used for ViT, treat it as half a GPU.
    num_layers_per_gpu = math.ceil(num_layers / (world_size - 0.5))
    num_layers_per_gpu = [num_layers_per_gpu] * world_size
    num_layers_per_gpu[0] = math.ceil(num_layers_per_gpu[0] * 0.5)
    layer_cnt = 0
    for i, num_layer in enumerate(num_layers_per_gpu):
        for j in range(num_layer):
            device_map[f'language_model.model.layers.{layer_cnt}'] = i
            layer_cnt += 1
    device_map['vision_model'] = 0
    device_map['mlp1'] = 0
    device_map['language_model.model.tok_embeddings'] = 0
    device_map['language_model.model.embed_tokens'] = 0
    device_map['language_model.output'] = 0
    device_map['language_model.model.norm'] = 0
    device_map['language_model.model.rotary_emb'] = 0
    device_map['language_model.lm_head'] = 0
    device_map[f'language_model.model.layers.{num_layers - 1}'] = 0

    return device_map



#### import csv
import os
import json
from google.cloud import vision
import imagehash
from PIL import Image
import requests
from io import BytesIO
import pandas as pd
import logging
from newspaper import Article
import nltk
from collections import OrderedDict
from googleapiclient.discovery import build


TRUSTED_FACT_CHECK_SOURCES = ["politifact.com", "factcheck.org","snopes.com","npr.org" ]
TRUSTED_NEWS_SOURCES = [
    "bbc.com", "bbc.co.uk",
    "theguardian.com",
    "washingtonpost.com",
    "usatoday.com",
    "reuters.com",
    "nytimes.com",
    "cnn.com",
    "npr.org",
    "apnews.com",
    "forbes.com",
    "bloomberg.com",
    "theatlantic.com",
    "economist.com"
]

def detect_news_links(image_path):
    """Detects web-related information about an image and fetches relevant news articles."""
    client = vision.ImageAnnotatorClient()
    
    # Read the image
    with open(image_path, "rb") as image_file:
        content = image_file.read()

    image = vision.Image(content=content)


    response = client.web_detection(image=image)
    web_detection = response.web_detection
    relevant_fact_checker_links = []
    relevant_news_links = []
    similar_news_links = []
    # Filtering only trusted news sources
    if web_detection.pages_with_matching_images:
        print("\nRelevant News Pages:")

        for page in web_detection.pages_with_matching_images:
            url = page.url
            if any(source in url for source in TRUSTED_FACT_CHECK_SOURCES):  # Check if URL is from a trusted source
                print(f"News URL Found: {url}")
                relevant_fact_checker_links.append(url)
            if any(source in url for source in TRUSTED_NEWS_SOURCES):
                relevant_news_links.append(url)
                print(f"News URL Found: {url}")
    if relevant_fact_checker_links:
        return 0,relevant_fact_checker_links[0]   
    if relevant_news_links:
        return 1,relevant_news_links
    else:
        return 2, similar_news_links
        
def fetch_article_content(url):
    try:
        article = Article(url)
        article.download()
        article.parse()
        article.nlp()

        # Fallback to current datetime if publish_date is None
        date = article.publish_date or datetime.datetime.now()

        print(f"Title: {article.title}")
        print(f"Authors: {article.authors}")
        print(f"Published Date: {article.publish_date}")
        print(f"Keywords: {article.keywords}")

        return article.summary, article.title, article.keywords, date

    except Exception as e:
        logging.error(f"Error fetching article from {url}: {e}")
        return "", "", [], datetime.datetime.now()  # Return 4 placeholders on failure

    except Exception as e:
        logging.error(f"Error fetching article from {url}: {e}")
        return '','',''

def fetch_article_contents_from_image(image_path):
    check, news_links = detect_news_links(image_path)
    if check == 0:
        summary, title,keywords,date = fetch_article_content(news_links)
        return news_links, summary,title,keywords,date
    elif check == 1:
        earliest_article = {
            "summary": '',
            "title": '',
            "keywords": '',
            "date": datetime.datetime.max,
            "link": ''
        }

        for link in news_links:
            summary, title, keywords, date = fetch_article_content(link)
            if (
                isinstance(date, datetime.datetime)
                and date.replace(tzinfo=None) < earliest_article["date"].replace(tzinfo=None)
            ):                earliest_article.update({
                    "summary": summary,
                    "title": title,
                    "keywords": keywords,
                    "date": date,
                    "link": link
                })

        return (earliest_article['link'],earliest_article["summary"], earliest_article["title"],earliest_article["keywords"], earliest_article["date"])
    else:
        return '','','','',''

def pick_fact_or_news(urls):
    """
    Given a list of URLs, return:
      - all URLs from TRUSTED_FACT_CHECK_SOURCES if any exist;
      - otherwise all URLs from TRUSTED_NEWS_SOURCES.
    """
    # normalize to lowercase for matching
    lower_urls = [(u, u.lower()) for u in urls]

    # find fact-check URLs
    fact_urls = [
        orig for orig, l in lower_urls
        if any(domain in l for domain in TRUSTED_FACT_CHECK_SOURCES)
    ]
    if fact_urls:
        return 0,fact_urls[0]

    # otherwise pick news URLs
    news_urls = [
        orig for orig, l in lower_urls
        if any(domain in l for domain in TRUSTED_NEWS_SOURCES)
    ]
    return 1,news_urls   
        
def fetch_article_contents_from_image_text(image_path):
    image_text = detect_text(image_path)
    retrive_relevant_links = google_search_urls(image_text, 10)
    check, news_links = pick_fact_or_news(retrive_relevant_links)
    if check == 0:
        summary, title,keywords,date = fetch_article_content(news_links)
        return news_links, summary,title,keywords,date
    elif check == 1:
        earliest_article = {
            "summary": '',
            "title": '',
            "keywords": '',
            "date": datetime.datetime.max,
            "link": ''
        }

        for link in news_links:
            summary, title, keywords, date = fetch_article_content(link)
            if (
                isinstance(date, datetime.datetime)
                and date.replace(tzinfo=None) < earliest_article["date"].replace(tzinfo=None)
            ):                earliest_article.update({
                    "summary": summary,
                    "title": title,
                    "keywords": keywords,
                    "date": date,
                    "link": link
                })

        return (earliest_article['link'],earliest_article["summary"], earliest_article["title"],earliest_article["keywords"], earliest_article["date"])
    else:
        return '','','','',''
    
def detect_text(path):
    """Detects text in the file."""
    from google.cloud import vision

    client = vision.ImageAnnotatorClient()

    with open(path, "rb") as image_file:
        content = image_file.read()

    image = vision.Image(content=content)

    response = client.text_detection(image=image)
    annotations = response.text_annotations
    if not annotations:
        return ""  # no text detected

    # The 0th element is the full text block
    full_text = annotations[0].description

    # (Optional) if you wanted each individual piece instead:
    # pieces = [ann.description for ann in annotations[1:]]
    # full_text = " ".join(pieces)
    print(full_text)
    return full_text
    

def detect_web_entities(image_path):
    """Detects web-related information about an image and fetches relevant news articles."""
    client = vision.ImageAnnotatorClient()

    # Read the image
    with open(image_path, "rb") as image_file:
        content = image_file.read()

    image = vision.Image(content=content)

    # Perform web detection
    response = client.web_detection(image=image)
    web_detection = response.web_detection

    web_entities = []
          
    if web_detection.web_entities:
        print("\n{} Web entities found: ".format(len(web_detection.web_entities)))
        for entity in web_detection.web_entities:
            print(f"\n\tScore      : {entity.score}")
            print(f"\tDescription: {entity.description}")
            web_entities.append(entity.description)

    return web_entities

def shorten_summary(summary, model, tokenizer, generation_config):
    prompt = f'Can you shorten this article so that it will be 500 words at most: {summary}. As a response, only give the result.'
    response = model.chat(tokenizer, None, prompt, generation_config)
    return response
def extract_story_number(story_id):
    if isinstance(story_id, str) and story_id.startswith("story_"):
        # Split on '+' and take the first part
        first_part = story_id.split('+')[0]
        return first_part
    return story_id

question = """ <image>
### Context
You are provided with:
- A tweet written by a user
- An image used in the tweet
-(If given) The consensus and examples related to the image
- The branch of the image

Classify the stance of the TWEET toward the IMAGE and return a final label.

### Definitions

- "Misleading": the text supports a claim that contradicts the established context of the image or asserts misinformation.
- "Debunking": the text refutes a misleading interpretation of the image.
- "Unrelated": the text does not refer to or explain the image (off-topic).

### Decision Process

Branch rules for the final label:
- If BRANCH = MISLEADING_IMAGE:
   Debunking → Benign; Misleading or Unrelated → Misleading.
- If BRANCH = BENIGN_IMAGE:
   Unrelated → Benign; Misleading or Debunking → Misleading.
---
### Few-Shot Examples

**Example 1**  
Image: A crowded supermarket checkout line (2022 photo)  
Branch: MISLEADING_IMAGE  
Tweet: "This is proof people were panic-buying before the 2024 election — shelves were empty everywhere!"  
Final Label: Misleading  

**Example 2**  
Image: A crowded supermarket checkout line (2022 photo)  
Branch: MISLEADING_IMAGE  
Tweet: "This picture is from last year during a snowstorm, not related to the election."  
Final Label: Benign  

**Example 3**  
Image: A city skyline at sunset  
Branch: BENIGN_IMAGE  
Tweet: "The government caused this blackout by shutting down the power grid."  
Final Label: Misleading  

**Example 4**  
Image: A city skyline at sunset  
Branch: BENIGN_IMAGE  
Tweet: "Beautiful view of the city tonight."  
Final Label: Benign  
---

### Response Format

Respond with:
**One of the two labels**: Misleading, Benign
"""    

question_2 = """ <image>
### Context
You are provided with:
- A tweet written by a user
- An image used in the tweet

Classify the stance of the TWEET toward the IMAGE and return a final label.

### Definitions

- "Misleading": the text supports a claim that contradicts the established context of the image or asserts misinformation.
- "Debunking": the text refutes a misleading interpretation of the image.
- "Unrelated": the text does not refer to or explain the image (off-topic).

### Few-Shot Examples

**Example 1**  
Image: A crowded supermarket checkout line (2022 photo)  
Branch: MISLEADING_IMAGE  
Tweet: "This is proof people were panic-buying before the 2024 election — shelves were empty everywhere!"  
Final Label: Misleading  

**Example 2**  
Image: A crowded supermarket checkout line (2022 photo)  
Branch: MISLEADING_IMAGE  
Tweet: "This picture is from last year during a snowstorm, not related to the election."  
Final Label: Benign  

**Example 3**  
Image: A city skyline at sunset  
Branch: BENIGN_IMAGE  
Tweet: "The government caused this blackout by shutting down the power grid."  
Final Label: Misleading  

**Example 4**  
Image: A city skyline at sunset  
Branch: BENIGN_IMAGE  
Tweet: "Beautiful view of the city tonight."  
Final Label: Benign  

---
### Response Format

Respond with:
**One of the two labels**: Misleading, Benign
"""    


image_dirs = ["", ""]
tweet_list_a = []
with open('baselines_for_part_a_dataset.json', "r", encoding="utf-8") as f:
    tweet_list_a = json.load(f)

with open('baselines_for_part_b_dataset_2.json', 'r') as f:
    tweet_list_b = json.load(f)
# Open CSV file for writing
i = 0
b=0 
def main(): 
    check = 'llava'  
    updated_tweets = []
    model, proc = load_llava()
    # Reconstruct the dictionary with the new key after 'raw_img'
    i = 0
    for tweet in tweet_list_b:
        new_tweet = OrderedDict()
        full_text = tweet.get('full_text', '')
        # Extract the fields
        raw_img = tweet.get('raw_img', '')
        image_path = image_dirs[0] + raw_img
        if not os.path.exists(image_path):
            print(f"[SKIP] Image not found: {image_path}")
            continue  # Skip to next tweet
        link,summary,title,keywords,date = fetch_article_contents_from_image(image_path)
        generation_config = dict(max_new_tokens=500, do_sample=False)

        #summary = shorten_summary(summary, model, tokenizer, generation_config)
        #summary = "" if pd.isna(row["News Content Vision"]) else row["News Content Vision"]
        #title = '' if pd.isna(row["Title Vision"]) else row["Title Vision"]
        #keywords = "" if pd.isna(news_url) else fetch_article_content(news_url)
        print(image_path)
        print(f'id : {i}, full text: {full_text}, \n image_path: {image_path}')
        #image_web_entities = detect_web_entities(image_path)
        #if link == '' or link == None:
            #link,summary,title,keywords,date= fetch_article_contents_from_image_text(image_path)
        #prompt = f"""{question_2}\nTweet:{full_text}\nAnswer:"""
        prompt = f"""{question}\nBRANCH = BENIGN IMAGE\nTweet:{full_text}\nAnswer:"""
        if check == 'internvl':
            pixel_values = load_image(image_path, max_num=12).to(torch.bfloat16).cuda()
            generation_config = dict(max_new_tokens=250, do_sample=False)
            response = model.chat(tokenizer, pixel_values, prompt, generation_config)
            print(f'User: {prompt}\nAssistant: {response}')
            i = i+1
            for key in tweet:
                new_tweet[key] = tweet[key]
                if key == 'raw_img':
                    new_tweet['internvl_38_whole_system_prompt_2'] = response
            updated_tweets.append(new_tweet)

        if check == 'qwen':
                image = Image.open(image_path).convert("RGB")
                messages = [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image",
                            "image": image_path,
                        },
                        {"type": "text", "text": prompt},
                    ],
                }
                ]
                text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
                image_inputs, _ = process_vision_info(messages)
                inputs = tokenizer(
                    text=[text],
                    images=image_inputs,
                    padding=True,
                    return_tensors="pt",
                )
                inputs = inputs.to('cuda')
                generated_ids = model.generate(**inputs, max_new_tokens=300,  do_sample=False)
                generated_ids_trimmed = [
                    out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
                ]
                response = tokenizer.batch_decode(
                    generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
                )
                print(f'''User: {prompt}\n Assistant: {response}''')
                i = i+1
                for key in tweet:
                    new_tweet[key] = tweet[key]
                    if key == 'raw_img':
                        new_tweet['qwen_whole_system_prompt_2_few_shot'] = response
                updated_tweets.append(new_tweet)

        if check == 'llava':
            msgs = [{"role":"user","content":[{"type":"image"},
                                      {"type":"text","text":prompt}]}]
            inputs = proc.apply_chat_template(msgs, add_generation_prompt=True)
            raw_image = Image.open(image_path).convert("RGB")
            inputs = proc(images=raw_image, text=prompt, return_tensors='pt').to(0, torch.float16)
            output = model.generate(**inputs, max_new_tokens=200, do_sample=False)
            response =proc.decode(output[0][2:], skip_special_tokens=True)
            #response = extract_final_label(response_pre)
            #response = proc.batch_decode(output, skip_special_tokens=True)[0]
            #print("Final Answer:", response)
            #response = proc.batch_decode(output, skip_special_tokens=True)[0]
            print(f'''User: {prompt}\n Assistant: {response}''')
            i = i+1
            extracted_answer = ''
            match = re.search(r'Answer:\s*(.*?)(\n\n|$)', response, re.DOTALL)
            if match:
                extracted_answer = match.group(1).strip()
                print("Extracted Answer:", extracted_answer)
            else:
                print("Answer not found.")
            for key in tweet:
                new_tweet[key] = tweet[key]
                if key == 'raw_img':
                    new_tweet['llava_whole_system_prompt'] = extracted_answer
            updated_tweets.append(new_tweet)

        if check == 'gpt':
            base64_image = encode_image(image_path)
            try:
                resp = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                            {
                                "role": "user",
                                "content": [
                                    {
                                        "type": "text",
                                        "text" : prompt,
                                    },
                                    {
                                        "type": "image_url",
                                        "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"},
                                    },
                                ],
                            }
                        ],
                    max_tokens=100 
                    )
                i = i+1
                response = resp.choices[0].message.content
                print(f'''User: {prompt}\n Assistant: {response}''')
                for key in tweet:
                    new_tweet[key] = tweet[key]
                    if key == 'raw_img':
                        new_tweet['gpt_whole_system_prompt_few_shot'] =  response
                updated_tweets.append(new_tweet)
            except RateLimitError:
                time.sleep(40)  # Wait a second
                resp = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                            {
                                "role": "user",
                                "content": [
                                    {
                                        "type": "text",
                                        "text" : prompt,
                                    },
                                    {
                                        "type": "image_url",
                                        "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"},
                                    },
                                ],
                            }
                        ],
                    max_tokens=100 
                    )
                i = i+1
                print(f'''User: {prompt}\n Assistant: {resp.choices[0].message.content}''')
                response =  resp.choices[0].message.content
                for key in tweet:
                    new_tweet[key] = tweet[key]
                    if key == 'raw_img':            
                        new_tweet['gpt_whole_system_prompt_few_shot'] = response
                updated_tweets.append(new_tweet)
        if check == 'instructblip':
            image =  Image.open(image_path).convert("RGB")
            inputs = processor(images=image, text=prompt,    truncation=True,  max_length=512,  return_tensors="pt").to("cuda")
            outputs = model.generate(
                **inputs,
                do_sample=False,
                num_beams=5,
                max_new_tokens=256,       # <-- replace max_length with this
                min_length=1,
                top_p=0.9,
                repetition_penalty=1.5,
                length_penalty=1.0,
                temperature=1,
            )
            response = processor.batch_decode(outputs, skip_special_tokens=True)[0].strip()
            print(f'''User: {prompt}\n Assistant: {response}''')
            for key in tweet:
                new_tweet[key] = tweet[key]
                if key == 'raw_img':
                    new_tweet['instructblip_whole_system_prompt_few_shot'] =  response
            updated_tweets.append(new_tweet)      
            i = i +1
    with open("baselines_for_part_b_dataset_2.json", "w") as f:
        json.dump(updated_tweets, f, indent=4)
    updated_tweets = []
    for tweet in tweet_list_a:
        new_tweet = OrderedDict()
        full_text = tweet.get('tweet_text', '')
        # Extract the fields
        raw_img = tweet.get('storyid', '')
        storyid = extract_story_number(raw_img)
        image_path = image_dirs[1] + storyid + '.jpg'
        fact = tweet.get('fact')
        refuting = tweet.get('supporting')
        supporting = tweet.get('refuting')
        if not os.path.exists(image_path):
            print(f"[SKIP] Image not found: {image_path}")
            continue  # Skip to next tweet
        #link,summary,title,keywords,date = fetch_article_contents_from_image(image_path)
        generation_config = dict(max_new_tokens=500, do_sample=False)


        #summary = shorten_summary(summary, model, tokenizer, generation_config)
        #summary = "" if pd.isna(row["News Content Vision"]) else row["News Content Vision"]
        #title = '' if pd.isna(row["Title Vision"]) else row["Title Vision"]
        #keywords = "" if pd.isna(news_url) else fetch_article_content(news_url)
        print(image_path)
        print(f'id : {i}, full text: {full_text}, \n image_path: {image_path}')
        #image_web_entities = detect_web_entities(image_path)
        #if link == '' or link == None:
            #link,summary,title,keywords,date= fetch_article_contents_from_image_text(image_path)
        
        
        prompt = f"""{question}\nBRANCH=MISLEADING IMAGE\n
                      Consensus Statement:{fact}\n
                      Supporting Evidence:{supporting}\n
                      Refuting Evidence: {refuting}\nTweet: {full_text}\nAnswer:"""
        
        #prompt = f"""{question_2}\nTweet: {full_text}\nAnswer:"""
    
        if check == 'internvl':
            pixel_values = load_image(image_path, max_num=12).to(torch.bfloat16).cuda()

            #pixel_values = torch.cat((pixel_values1, pixel_values2,pixel_values3,pixel_values4,pixel_values), dim=0).cuda()
                    # Generate model response
            generation_config = dict(max_new_tokens=250, do_sample=False)
            response = model.chat(tokenizer, pixel_values, prompt, generation_config)
            print(f'User: {prompt}\nAssistant: {response}')
            i = i+1
            for key in tweet:
                new_tweet[key] = tweet[key]
                if key == 'max_similarity_score_llama_cls_token':
                    new_tweet['internvl_38_whole_system_prompt_2'] = response
            updated_tweets.append(new_tweet)
        if check == 'qwen':
                image = Image.open(image_path).convert("RGB")
                messages = [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image",
                            "image": image_path,
                        },
                        {"type": "text", "text": prompt},
                    ],
                }
                ]
                text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
                image_inputs, _ = process_vision_info(messages)
                inputs = tokenizer(
                    text=[text],
                    images=image_inputs,
                    padding=True,
                    return_tensors="pt",
                )
                inputs = inputs.to('cuda')
                generated_ids = model.generate(**inputs, max_new_tokens=300,  do_sample=False)
                generated_ids_trimmed = [
                    out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
                ]
                response = tokenizer.batch_decode(
                    generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
                )
                print(f'User: {prompt}\nAssistant: {response}')
                i = i+1
                for key in tweet:
                    new_tweet[key] = tweet[key]
                    if key == 'max_similarity_score_llama_cls_token':
                        new_tweet['qwen_whole_system_prompt_2_few_shot'] = response
                updated_tweets.append(new_tweet)

        if check == 'llava':
            msgs = [{"role":"user","content":[{"type":"image"},
                                      {"type":"text","text":prompt}]}]
            inputs = proc.apply_chat_template(msgs, add_generation_prompt=True)
            raw_image = Image.open(image_path).convert("RGB")
            inputs = proc(images=raw_image, text=prompt, return_tensors='pt').to(0, torch.float16)
            output = model.generate(**inputs, max_new_tokens=200, do_sample=False)
            response =proc.decode(output[0][2:], skip_special_tokens=True)
            #response = extract_final_label(response_pre)

            print("Final Answer:", response)
            #response = proc.batch_decode(output, skip_special_tokens=True)[0]
            print(f'''User: {prompt}\n Assistant: {response}''')
            i = i+1
            extracted_answer = ''
            match = re.search(r'Answer:\s*(.*?)(\n\n|$)', response, re.DOTALL)
            if match:
                extracted_answer = match.group(1).strip()
                print("Extracted Answer:", extracted_answer)
            else:
                print("Answer not found.")
            for key in tweet:
                new_tweet[key] = tweet[key]
                if key == 'max_similarity_score_llama_cls_token':
                    new_tweet['llava_whole_system_prompt'] = extracted_answer
            updated_tweets.append(new_tweet)
        if check == 'gpt':
            base64_image = encode_image(image_path)
            try:
                resp = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                            {
                                "role": "user",
                                "content": [
                                    {
                                        "type": "text",
                                        "text" : prompt,
                                    },
                                    {
                                        "type": "image_url",
                                        "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"},
                                    },
                                ],
                            }
                        ],
                    max_tokens=100 
                    )
                i = i+1
                response = resp.choices[0].message.content
                print(f'''User: {prompt}\n Assistant: {response}''')
                for key in tweet:
                    new_tweet[key] = tweet[key]
                    if key == 'max_similarity_score_llama_cls_token':
                        new_tweet['gpt_whole_system_prompt_few_shot'] = response
                updated_tweets.append(new_tweet)
            except RateLimitError:
                time.sleep(40)  # Wait a second
                resp = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                            {
                                "role": "user",
                                "content": [
                                    {
                                        "type": "text",
                                        "text" : prompt,
                                    },
                                    {
                                        "type": "image_url",
                                        "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"},
                                    },
                                ],
                            }
                        ],
                    max_tokens=100 
                    )
                i = i+1
                response = resp.choices[0].message.content

                print(f'''User: {prompt}\n Assistant: {resp.choices[0].message.content}''')
                for key in tweet:
                    new_tweet[key] = tweet[key]
                    if key == 'max_similarity_score_llama_cls_token':
                        new_tweet['gpt_whole_system_prompt_few_shot'] = response
                updated_tweets.append(new_tweet)
        if check == 'instructblip':
            image =  Image.open(image_path).convert("RGB")
            inputs = processor(images=image, text=prompt,    truncation=True,  max_length=512,  return_tensors="pt").to("cuda")

            outputs = model.generate(
                **inputs,
                do_sample=False,
                num_beams=5,
                max_new_tokens=256,       # <-- replace max_length with this
                min_length=1,
                top_p=0.9,
                repetition_penalty=1.5,
                length_penalty=1.0,
                temperature=1,
            )
            response = processor.batch_decode(outputs, skip_special_tokens=True)[0].strip()
            print(f'''User: {prompt}\n Assistant: {response}''')
            for key in tweet:
                new_tweet[key] = tweet[key]
                if key == 'max_similarity_score_llama_cls_token':
                    new_tweet['instructblip_whole_system_prompt_few_shot'] =  response
            i = i +1

            updated_tweets.append(new_tweet)      
    with open("baselines_for_part_a_dataset.json", "w") as f:
        json.dump(updated_tweets, f, indent=4)  

def extract_final_label(response_pre):
    """
    Extract the last classification label (Misleading or Benign) from model response.
    """
    matches = re.findall(r"Answer:\s*(Misleading|Benign)", response_pre, flags=re.IGNORECASE)
    if matches:
        return matches[-1].strip().capitalize()
    return ''

def load_llava():
    ckpt =  "llava-hf/llava-1.5-7b-hf"
    proc = AutoProcessor.from_pretrained(ckpt, trust_remote_code=True)
    model = LlavaForConditionalGeneration.from_pretrained(
        ckpt, torch_dtype=torch.bfloat16, device_map="auto", trust_remote_code=True, cache_dir = './'
        # attn_implementation="flash_attention_2",   # if installed
    )
    return model,proc

def load_instructblip():
    model = InstructBlipForConditionalGeneration.from_pretrained("Salesforce/instructblip-vicuna-7b",  cache_dir = './',
)
    processor = InstructBlipProcessor.from_pretrained("Salesforce/instructblip-vicuna-7b",cache_dir = './')

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)
    return model, processor
def load_qwen():
    # Assuming you want to use the second GPU (index 1)
    # default: Load the model on the available device(s)
    model = Qwen2VLForConditionalGeneration.from_pretrained(
        "Qwen/Qwen2-VL-7B-Instruct", torch_dtype=torch.bfloat16, device_map="auto", cache_dir = './', token =hf_token
    )
    # We recommend enabling flash_attention_2 for better acceleration and memory saving, especially in multi-image and video scenarios.
    # model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
    #     "Qwen/Qwen2.5-VL-72B-Instruct",
    #     torch_dtype=torch.bfloat16,
    #     attn_implementation="flash_attention_2",
    #     device_map="auto",
    # )

    # default processer
    processor = AutoProcessor.from_pretrained("Qwen/Qwen2-VL-7B-Instruct", use_fast =True ,cache_dir = './', token =hf_token)
    return model,processor
    # The default range for the number of visual tokens per image in the model is 4-16384.
    # You can set min_pixels and max_pixels according to your needs, such as a token range of 256-1280, to balance performance and cost.
    # min_pixels = 256*28*28
    # max_pixels = 1280*28*28
    # processor = AutoProcessor.from_pretrained("Qwen/Qwen2.5-VL-72B-Instruct", min_pixels=min_pixels, max_pixels=max_pixels)
def encode_image(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode("utf-8")
        
def load_internvl():
        path = 'OpenGVLab/InternVL2_5-38B-MPO'
        device_map = split_model('InternVL2_5-38B')
        model = AutoModel.from_pretrained(
            path,
            torch_dtype=torch.bfloat16,
            load_in_8bit=False,
            low_cpu_mem_usage=True,
            use_flash_attn=True,
            trust_remote_code=True,
            cache_dir = './',
            device_map=device_map).eval()
        tokenizer = AutoTokenizer.from_pretrained(path, cache_dir= './', trust_remote_code=True, use_fast=False)
        return model, tokenizer
    

if __name__ == "__main__":
    main()

                            