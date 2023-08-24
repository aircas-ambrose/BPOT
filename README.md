# BPOT: Narrative Order Aware Story Generation via Bidirectional Pretraining Model with Optimal Transport Reward
This is the initial version of BPOT.  

### **Preliminary**
+ environment: we export our environment in ```env.yml```
+ pretraining data: download pretraining data (Temporal Event sequences) from https://github.com/jjasonn0717/TemporalBART.
+ finetuning data: download finetuning data (ROCStories and WritingPrompts) from https://github.com/PlusLabNLP/flashback_gen.

### **Data Processing**
+ For pretraining data, utilize ```TempRel``` from EventPlus https://github.com/PlusLabNLP/EventPlus to detect the event trigger for each sentence, then utilize ECONET from https://github.com/PlusLabNLP/ECONET to get the temporal relationship between each pairwise event. The temporal relationship is used as temporal prompts in bidirectional pretraining stage.
+ For finetuning data, split each sentence in original story and set each sentence as new key in dict. For example, for an entriy story ```"Dan's parents were overweight. Dan was overweight as well. The doctors told his parents it was unhealthy. His parents understood and decided to make a change. The got themselves and Dan on a diet."``` Then set dict_name["sentence1"] = ```"Dan's parents were overweight."```, dict_name["sentence2"] = ```"Dan was overweight as well."```, dict_name["sentence3"] = ```"The doctors told his parents it was unhealthy."```, dict_name["sentence4"] = ```"The doctors told his parents it was unhealthy."```, dict_name["sentence_5"] = ```"The got themselves and Dan on a diet".``` Note that this is only necessary for ROCStories, the WritingPrompts datasets itself is sharded.
+ Place the pretraining data in directory ```./data/pretrain_data```
+ Place the processed finetuning data in directory ```./data```

### **Pretraining Stage**
+ The pretrained code is adapted from https://github.com/zhufq00/mcnc/. We modify its data processing (masking paradigm) and some pretraining strategies.
+ To perform bidirectional pretraining, first ```cd "./mcnc/configs/"``` and modify the ```Bidirectional_bart_base_pretraining.yaml``` to your own configuration. Then ```bash run.sh```
+ Note that the bidirectional pretrianing model is used to finetune for both RocStories and WritingPrompts datasets. 

### **Finetuning Stage**
#### ROCStories
+ For vanilla version, it doesn't include bidirectional pretraining and reinforcemnet leanring with optimal transport-based reward. To train vanilla model, execute ```bash run_story_generation_pipeline.sh```
+ For bidirectional pretraining version, execute ```bash run_story_generation_pipeline.sh``` Note that you have to ensure to set the path of pretraining checkpoint in ```run_story_generation_pipeline.sh``` (the item "load_model_dir")
+ For reinforcement learning with optimal transport reward version, execute ```bash run_story_generation_pipeline_otrl.sh``` 
+ For BPOT version, execute ```bash run_story_generation_pipeline_otrl.sh``` Note that you have to ensure to set the path of pretraining checkpoint in ```run_story_generation_pipeline_otrl.sh``` (the item "load_model_dir")
+ The log of training process can be found in the directory ```./logs_final```
#### WritingPrompts
+ For vanilla version, it doesn't include bidirectional pretraining and reinforcemnet leanring with optimal transport-based reward. To train vanilla model, execute ```bash run_story_generation_pipeline_wp.sh```
+ For bidirectional pretraining version, execute ```bash run_story_generation_pipeline_wp.sh``` Note that you have to ensure to set the path of pretraining checkpoint in ```run_story_generation_pipeline_wp.sh``` (the item "load_model_dir")
+ In WritingPrompts datasets, the number of sentences in each story is different so that the batch IPOT algorithm can't be used. Therefore, we modify the corresponding part in ```run_story_generation_pipeline_otrl.py``` to get "run_story_generation_pipeline_otrl_wp.py".
+ For reinforcement learning with optimal transport reward version, execute ```bash run_story_generation_pipeline_otrl_wp.sh```
+ For BPOT version, execute ```bash run_story_generation_pipeline_otrl_wp.sh``` Note that you have to ensure to set the path of pretraining checkpoint in ```run_story_generation_pipeline_otrl_wp.sh``` (the item "load_model_dir")
+ The log of training process can be found in the directory ```./logs_final```

### **Inference Stage** 
+ For ROCStories ```bash eval_story_generation_pipleine.sh``` Note that change the item "suffix" for inferring different models
+ For WritingPrompts ```bash eval_story_generation_pipeline_wp.sh``` Note that change the item "suffix" for inferring different models
+ The generated results for ROCStories and WritingPrompts can be found in the directory ```./generation``` and  ```./generation_wp``` respectively.

### **Comparative Experiments**
+ The comparative experiments for bidirectional pretraining is conducted by changing the masking paradigm in Pretraining Stage. It is convenient for you to set the parameter ```ablation_event``` and ```ablation_timing``` in ```./mcmc/configs/Bidirectional_bart_base_pretraining.yaml``` and rerun the pretraining stage to realize it.
+ The comparative experiments for Optimal Transport-based reward is conducted by changing the construction of rewards. For story-level RL reward, change the code in line744 and rerun the ```bash run_story_generation_pipeline_otrl.sh``` to realize it. For Naive Rl reward, ```bash run_story_generation_piepline_rl_sentence.sh``` to realize it.
