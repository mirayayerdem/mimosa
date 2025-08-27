# mimosa

## Model Codes

- **`part_a_model.py`** – Pipeline for tweets with **misleading images**, including ClaimSpotter, STELLA, and InternVL.  
- **`part_b_model.py`** – Pipeline for tweets with **benign images**, including Google Vision API and InternVL.  
- **`baseline_models_whole_pipeline.py`** – Runs baseline models on the **union dataset** (tweets with both misleading and benign images).  
- **`compute_performance_metics_baselines.py`** – Computes the evaluation scores of baseline models and `\sys`.  
- **`baseline_models_ctd_part_1.ipynb`** – Evaluates performance of the CTD tool on tweets with misleading images.  
- **`llama-70-b model for dataset creation.ipynb`** – Includes baseline models for classification of posts with misleading images; also includes OOC detection and experiments on benign images.  

Please make sure to set the environment variables before running the scripts.
```
export HF_TOKEN=your_huggingface_token
export GOOGLE_API_KEY=your_google_key
export OPENAI_API_KEY=your_openai_key
```
---

## Datasets

- **`all_dataset_misleading.csv`** – Full dataset including images flagged by PixelMod.  
- **`ann_dataset_misleading_images.json`** – Annotated subset of `all_dataset_misleading.csv` (173 tweets), labels: *refuting / supporting / irrelevant*.  
- **`ann_dataset_misleading_images.csv`** – Same annotations as above in CSV format.  
- **`all_dataset_benign.json`** – Full dataset including images not flagged by PixelMod.  
- **`ann_dataset_benign.json`** – Annotated subset of `all_dataset_benign.json` with labels: *symmetric-miscaptioned / symmetric-benign / asymmetric*.  
  - Note: the *misleading image* class is removed during evaluation.  
- **`baselines_for_part_a_dataset`** – Tweets with misleading images plus baseline and `\sys` results.  
- **`baselines_for_part_b_dataset`** – Tweets with benign images plus baseline and `\sys` results.  
- **`image_caption_analysis_intern_vl_78_2_3.csv`** – 300 samples from the NewsClipping dataset with baseline performances (models, prompt selection, web-source selection).  
- **`ann_dataset_benign_image_2.json`** – Annotated tweets with benign images and results from baseline models.  

Raw datasets are uploaded in [Zenodo link](https://zenodo.org/records/16957489) containing following zipped files:
- **`misleading_images_folder.zip `** – Contains images flagged by PixelMod; ⁠ storyid ⁠ keys in ⁠ all_dataset_misleading.csv ⁠ map to these images.
- **⁠`benign_images_folder.zip` ⁠** – Contains images not flagged by PixelMod; ⁠ raw_img ⁠ keys in ⁠ all_dataset_benign.json ⁠ map to these images
