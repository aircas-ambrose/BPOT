task="story_generation"
lrs=(1e-4)
batch=(64)
seed=5      # set the random seed
device="7"   # set the gpu id
event_num=0
model=""     # set the path of pretrained bart-based model in huggingface
model_name="facebook/bart-base"

# suffix="gen_no_struct_pipeline_story_input_wp_max500_event_script"
# suffix="gen_with_rel_pipeline_story_input_wp_max500"
# suffix="gen_with_rel_output_pipeline_story_input_event_script_update_otrl_sentence_mean_try_2.5_0.25_5_wp_max500"
# suffix="gen_with_rel_output_pipeline_pretrain_story_input_event_script_update_otrl_sentence_mean_try_2.5_0.25_5_wp_max500"

root="../output_final/${task}/"

for l in "${lrs[@]}"
do
  for s in "${batch[@]}"
  do
      python eval_story_generation_pipeline.py \
      --data_dir "../data/" \
      --model ${model} \
      --topk_sample \
      --task_name  ${task} \
      --gen_storyline_len 512 \
      --file_suffix "_story_generation_wp_baseline_final.json" \
      --device_num ${device} \
      --eval_batch_size 2 \
      --num_train_epochs 10 \
      --learning_rate ${l} \
      --input_event_num ${event_num} \
      --seed ${seed} \
      --model_dir "${root}/${model_name}_batch_${s}_lr_${l}_seed_${seed}_event_${event_num}_${suffix}/"
  done
done
