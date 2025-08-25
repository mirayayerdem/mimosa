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
path = "OpenGVLab/InternVL2_5-38B-MPO"

cache_dir = 'cache_dir'
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
from PIL import Image
import requests
import pandas as pd
import logging
from newspaper import Article
from collections import OrderedDict


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

question = """<image>
### Context
You are provided with:
- A tweet written by a user
- An image used in the tweet
- An article title the image belongs to

Your task is to analyze how the tweet text relates specifically to the image and determine the type of relationship or miscaptioning it might represent.

### Definitions

1. **Narrative-Based Miscaptioning**:  
   The image is relevant to the tweet, but the **text exaggerates or adds interpretive claims** not directly visible. Often used to push narratives (e.g., political bias, fraud) that the image doesn't support on its own.

2. **Evidence-Based Miscaptioning**:  
   The tweet **factually misrepresents the image**, such as stating the wrong location, time, person, or event. The image is used **out of its original context**.

3. **None of them**:  
   The tweet **accurately references the image** without distortion. No false, misleading, or unverifiable claims are made. If the tweet text claims something parallel to the evidence of the image -article title or article summary image belongs to- it belogns to this class.

4. **Asymmetric**:  
   The image and tweet are **not clearly related**. The image neither supports nor contradicts the tweet’s message — they are disconnected.


### Decision Process

Ask: **Does the image meaningfully support the tweet text?**
- If **yes**:
  - Does the tweet add unverifiable or biased interpretation? → Narrative-Based
  - Does the tweet misstate factual details in the image? → Evidence-Based
  - Is the tweet accurate and neutral? → None of them
- If **no**, image and text are disconnected → Asymmetric

---

### Response Format

Respond with:
1. **One of the four labels**: Narrative-Based Miscaptioning, Evidence-Based Miscaptioning, None of them, or Asymmetric  
2. A brief reasoning (2–4 sentences) explaining your classification based on the image and tweet content.
"""    

question_few_shot = """<image>
### Context
You are provided with:
- A tweet written by a user
- An image used in the tweet
- Web entities of the image

Your task is to analyze how the tweet text relates specifically to the  image and determine the type of relationship or miscaptioning it might represent.

### Definitions

1. **Narrative-Based Miscaptioning**:  
   The image is relevant to the tweet, but the **text exaggerates or adds interpretive claims** not directly visible. Often used to push narratives (e.g., political bias, fraud) that the image doesn't support on its own.

2. **Evidence-Based Miscaptioning**:  
   The tweet **factually misrepresents the image**, such as stating the wrong location, time, person, or event. The image is used **out of its original context**.

3. **None of them**:  
   The tweet **accurately references the image** without distortion. No false, misleading, or unverifiable claims are made. If the tweet text claims something parallel to the evidence of the image -article title or article summary image belongs to- it belogns to this class.

4. **Asymmetric**:  
   The image and tweet are **not clearly related**. The image neither supports nor contradicts the tweet’s message — they are disconnected.

---

### Examples

**Example 1**  
Tweet: “And then there's Pennsylvania, where Biden led on Election Night, the votes counted over the next several hours were good for Trump, but the absentee ballots counted after that showed that Biden had won.
To be clear: this isn't fraud, just the slow wheels of democracy turning.”  
Image: image shows matching graph.
→ **Label**: None of them
→ **Reasoning**: The tweet accurately describes the vote counting process and matches the timeline shown in the image. It makes no misleading claims, clearly stating that the process is normal and not fraudulent. The image and text support each other truthfully and informatively.

**Example 2**  
Tweet: “My vote says cancelled.  I voted in person last night.  I don’t have early voting or mail in.  They made me use a Sharpie and it bled completely through. I’m in AZ.  This is voter FRAUD. AZ needs a recount!! @POTUS @PressSec @TuckerCarlson @w_terrence @DavidJHarrisJr @TomiLahren”  
Image: image shows mail-in ballot record.  
→ **Label**:  Evidence-based Miscaptioning
→ **Reasoning**: The tweet claims the user voted in person and did not use mail-in voting, while the image is a record specifically for a mail or early ballot. The image and tweet refer to different voting methods, causing a factual mismatch. The image is reused out of its proper context to support a misleading claim of voter fraud.

**Example 3**  
Tweet: “If you're wondering why Michigan and Wisconsin are flipping and why Trump has been railing against mail-in ballots, here is the picture.
Mail-in greatly favors Democrats. It is also more prone to ballot harvesting and manipulation unfortunately.”  
Image: Image shows vote count
→ **Label**: Narrative-based Miscaptioning
→ **Reasoning**: While the image accurately shows absentee vote counts, the tweet goes beyond the data by claiming mail-in ballots are prone to fraud and manipulation. These claims are not supported by the image itself. The image is used to push a broader, politically charged narrative.

**Example 4**  
Tweet: “Federal Agent Russell Strasser Who Intimidated and Coerced Pennsylvania USPS Whistleblower is a Trump-Hating Biden Supporter.
The tactics used to coerce and intimidate the USPS whistleblower is something you would see in a Communist country.
#StopTheSteal”  
Image:image shows unrelated portraits.
→ **Label**: Asymmetric  
→ **Reasoning**: The image presents individuals’ identities but does not illustrate or confirm the accusations made in the tweet. There is no direct visual evidence of intimidation, coercion, or political affiliation. The tweet and image are loosely connected at best, making this an asymmetric pairing.
---

### Decision Process

Ask: **Does the image meaningfully support the tweet text?**
- If **yes**:
  - Does the tweet add unverifiable or biased interpretation? → Narrative-Based
  - Does the tweet misstate factual details in the image? → Evidence-Based
  - Is the tweet accurate and neutral? → None of them
- If **no**, image and text are disconnected → Asymmetric

---

### Response Format

Respond with:
1. **One of the four labels**: Narrative-Based Miscaptioning, Evidence-Based Miscaptioning, None of them, or Asymmetric  
2. A brief reasoning (2–4 sentences) explaining your classification based on the image and tweet content.
"""
image_path= ''
image_dir = image_path

# Open CSV file for writing
i = 0
b=0 
def main():   
    path = 'OpenGVLab/InternVL2_5-38B-MPO'
    device_map = split_model('InternVL2_5-38B')
    model = AutoModel.from_pretrained(
        path,
        torch_dtype=torch.bfloat16,
        load_in_8bit=False,
        low_cpu_mem_usage=True,
        use_flash_attn=True,
        trust_remote_code=True,
        cache_dir =cache_dir,
        device_map=device_map).eval()
    tokenizer = AutoTokenizer.from_pretrained(path, cache_dir= cache_dir, trust_remote_code=True, use_fast=False)
    updated_tweets = []
    file_path = ''
    output_path = ''
    with open(file_path, 'r') as f:
        tweet_list = json.load(f)

    # Reconstruct the dictionary with the new key after 'raw_img'
    new_dict = OrderedDict()
    i = 0
    
    for tweet in tweet_list[:10]:
        new_tweet = OrderedDict()
        full_text = tweet.get('full_text', '')
        # Extract the fields
        raw_img = tweet.get('raw_img', '')
        image_path = image_dir + raw_img
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
        image_web_entities = detect_web_entities(image_path)
     
        prompt = f"{question_few_shot}\nTweet:{full_text}\nWeb entities:{image_web_entities}\nAnswer:"

        pixel_values = load_image(image_path, max_num=12).to(torch.bfloat16).cuda()

                # Generate model response
        generation_config = dict(max_new_tokens=250, do_sample=False)
        response = model.chat(tokenizer, pixel_values, prompt, generation_config)
        print(f'User: {prompt}\nAssistant: {response}')
        i = i+1
        for key in tweet:
            new_tweet[key] = tweet[key]
            if key == 'raw_img':
                new_tweet['vision_link'] = link
                new_tweet['vision_summary'] = summary
                new_tweet['vision_title'] = title
                new_tweet['vision_keywords'] = keywords
                new_tweet['internvl_38_web_entities_response_reasoning_few_shot'] = response
        updated_tweets.append(new_tweet)
    with open(output_path, "w") as f:
            json.dump(updated_tweets, f, indent=4)
            print('Done')
    
if __name__ == "__main__":
    main()

                            