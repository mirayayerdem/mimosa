# mimosa
Model Codes: 

part_a_model.py - the pipeline dealing with tweets containing misleading images including claimspotter, stella, internvl

part_b_model.py - the pipeline dealing with tweets containing benign images including google vision api and internvl

baseline_models_whole_pipeline.py - the pipeline that we run baselines models on the union dataset(tweets with misleading benign image)

compute_performance_metics_baselines - computes the evaluation score of baseline models and \sys

baseline_models_ctd_part_1.ipynb - evaluate performance of ctd tool on tweets with misleading images

llama-70-b model for dataset creation.ipynb - includes various baseline models for the classification of posts with misleading images, it also includes OOC detection and detection of tweets with benign images with baselines

Datasets:

all_dataset_misleading.csv - the whole dataset that includes images flagged by Pixelmod

ann_dataset_misleading_images.json - annotated part of all_dataset_misleading.csv  (173 tweets)(refuting/supporting/irrelevant)

ann_dataset_misleading_images.csv - annotated part of all_dataset_misleading.csv  (173 tweets)(refuting/supporting/irrelevant)

all_dataset_benign.json - the whole dataset that includes images not flagged by Pixelmod

ann_dataset_benign.json - annotated part of all_dataset_benign.json () (symmetric miscaptioned/symmetic-benign/asymmetric/misleading image-we remove this class during evaluation) 

misleading_images_folder.zip - contains the images flagged by pixelmod, 'storyid' keys in all_dataset_misleading.csv points to these images

benign_images_folder.zip - contains the images not flagged by pixelmod, 'raw_img' keys in ll_dataset_benign.json points to these images

baselines_for_part_a_dataset - includes the tweets with misleading images and our system and baseline performance results

baselines_for_part_b_dataset - includes the tweets with misleading images and our system and baseline performance results

image_caption_analysis_intern_vl_78_2_3.csv - includes 300 samples from NewsClipping dataset with our baseline performances (baseline models, prompt selection, web sources 
selection)

ann_dataset_benign_image_2.json - includes annotates tweets with benign images and results of our baseline models
