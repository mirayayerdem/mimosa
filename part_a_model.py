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
import json
import pandas as pd
import os

from sentence_transformers import SentenceTransformer, util

import datetime
path = "OpenGVLab/InternVL2_5-38B-MPO"
cache_dir = 'cache_dir'
api_key = os.environ.get("CLAIMBUSTER_API_KEY")
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
from PIL import Image
import requests
import pandas as pd

# Open CSV file for writing
i = 0
b=0 
def prepare_prompt(row):
    """
    Prepares the input prompt for the stance classification task.
    """
    return f"""You will be provided with a tweet, delimited by triple backticks, related to the given fact, supporting statement, and refuting statement. Indicate whether the tweet supports or refutes regarding the fact.
    Your response should be one of the following: [supporting, refuting]. Do not include any explanation.
    Fact: {row['fact']}
    Supporting Statement: {row['supporting']}
    Refuting Statement: {row['refuting']}
    Tweet: {row['tweet_text']}
    """
def get_stance(row, model, tokenizer):
    prompt_text = prepare_prompt(row)
    generation_config = dict(max_new_tokens=250, do_sample=False)
    response = model.chat(tokenizer, None, prompt_text, generation_config)
    print(response)
    return response

def run_claimspotter(df):
    # Define your API key
    # Initialize a list to store the maximum scores for each claim
    max_scores = []
    for input_claim in df['tweet_text']:
        # Define the endpoint with the claim formatted as part of it
        api_endpoint = f"https://idir.uta.edu/claimbuster/api/v2/score/text/sentences/{input_claim}"
        request_headers = {"x-api-key": api_key}
        # Send the GET request to the API and store the response
        api_response = requests.get(url=api_endpoint, headers=request_headers)
    
        # Initialize a variable to store the max score for this tweet
        max_score = 0
        # Check if the response was successful
        if api_response.status_code == 200:
            # Parse the JSON response
            sentences = api_response.json()
            # Check if 'results' key exists and is a list
            if 'results' in sentences and isinstance(sentences['results'], list):
                # Loop through each sentence to find the maximum score
                for sentence in sentences['results']:
                    if 'score' in sentence and sentence['score'] > max_score:
                        max_score = sentence['score']
            else:
                print("Unexpected response format:", sentences)
                max_score = None  # Set max_score to None if response format is unexpected
        else:
            print(f"Failed to retrieve data for claim: {input_claim}. Status code: {api_response.status_code}")
            max_score = None  # Set max_score to None if the request fails
        # Append the max score for this claim to the max_scores list
        max_scores.append(max_score)

    # Add the max scores as a new column in the DataFrame
    df['claimspotter_score'] = max_scores
    print("Max scores have been added")
    return df



def compute_scores(model, df, query_col, score_col_name):
    """
    For each unique query in `query_col`, compute cosine similarity between the query
    and the associated tweets. Returns a DataFrame with columns: ['tweet_id', score_col_name].
    """
    # Keep only rows that have a query
    sub = df[['tweet_id', 'tweet_text', query_col]].dropna(subset=[query_col]).copy()

    results = []
    query_prompt_name = "s2p_query"  # Sentence-to-passage prompt used by STELLA

    # Group by query (e.g., 'fact', 'supporting', 'refuting')
    for query, grp in sub.groupby(query_col):
        docs = grp['tweet_text'].tolist()
        ids  = grp['tweet_id'].tolist()

        # Encode query and documents
        q_emb   = model.encode(query, convert_to_tensor=True, prompt_name=query_prompt_name, normalize_embeddings=True)
        d_embs  = model.encode(docs,  convert_to_tensor=True,                         normalize_embeddings=True)

        # Cosine similarity: shape (1, N)
        scores = util.cos_sim(q_emb, d_embs)[0]

        # Map back to tweet_ids
        for tid, sc in zip(ids, scores):
            results.append({'tweet_id': tid, score_col_name: round(float(sc), 4)})

    if not results:
        return pd.DataFrame(columns=['tweet_id', score_col_name])

    res_df = pd.DataFrame(results)

    # If a tweet_id appears multiple times for the same query (duplicate rows),
    # choose the max score for stability.
    res_df = res_df.groupby('tweet_id', as_index=False)[score_col_name].max()
    return res_df

from huggingface_hub.constants import HF_HOME, HF_HUB_CACHE

def run_stella(df):
    print("HF_HOME (env):", os.environ["HF_HOME"])    
    print("HF_HUB_CACHE:", HF_HUB_CACHE)
    model = SentenceTransformer(
        "dunzhang/stella_en_400M_v5",
        trust_remote_code=True,device='cuda',config_kwargs={"use_memory_efficient_attention": False, "unpad_inputs": False})


    df = df.copy()
    if 'tweet_id' not in df.columns:
        raise ValueError("`tweet_id` column is required in df.")

    # Filter rows to run Stella on
    mask = df['claimspotter_score'] >= 0.49
    df_sub = df.loc[mask]

    if df_sub.empty:
        df['max_score_stella'] = pd.NA
        return df

    # Compute scores only on the subset
    res_fact  = compute_scores(model, df_sub, 'fact','score_stella_fact')
    res_supporting = compute_scores(model, df_sub, 'supporting','score_stella_supporting')
    res_refuting = compute_scores(model, df_sub, 'refuting', 'score_stella_refuting')

    merged = res_fact.merge(res_supporting, on='tweet_id', how='outer') \
                     .merge(res_refuting,on='tweet_id', how='outer')

    # Row-wise max across available scores
    score_cols = ['score_stella_fact', 'score_stella_supporting', 'score_stella_refuting']
    merged['max_score_stella'] = merged[score_cols].max(axis=1)

    final_df = df.merge(merged[['tweet_id', 'max_score_stella']], on='tweet_id', how='left')
    return final_df

def run_internvl(df):
    path = 'OpenGVLab/InternVL2_5-38B-MPO'
    device_map = split_model('InternVL2_5-38B')  
    model = AutoModel.from_pretrained(
            path,
            torch_dtype=torch.bfloat16,
            load_in_8bit=False,
            low_cpu_mem_usage=True,
            use_flash_attn=True,
            trust_remote_code=True,
            cache_dir=cache_dir,
            device_map=device_map).eval()
    tokenizer = AutoTokenizer.from_pretrained(
            path,
            cache_dir=cache_dir,
            trust_remote_code=True,
            use_fast=False
        )
    out = df.copy()
    threshold = 0.52
    # Create/ensure output column exists
    target_col = 'stance_internvl_api_final'
    if target_col not in out.columns:
        out[target_col] = pd.NA
    if 'max_score_stella' not in out.columns:
        raise ValueError("`max_score_stella` is required in df.")
    mask = out['max_score_stella'].ge(threshold).fillna(False)
    if not mask.any():
        return out

    out.loc[mask, target_col] = out.loc[mask].apply(
        get_stance, args=(model, tokenizer), axis=1
    )
    return out

def main():  
    df = pd.read_csv("/projectnb/llm-stance/multimodal-playground/twitter_dataset/benigns.csv")
    #run claimspotter
    df = run_claimspotter(df)
    #run stella
    df = run_stella(df)
    #run internvl
    df =run_internvl(df)
    df.to_csv("final_results_try.csv", index=False)


if __name__ == "__main__":
    main()

                            