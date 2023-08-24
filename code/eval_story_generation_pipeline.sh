task="story_generation"
lrs=(5e-5)
batch=(10)
seed=5     # set the random seed
device="1"   # set the device id
event_num=0
model=""   # set the path of pretrained-bart-based model
model_name="facebook/bart-base"

# suffix='gen_with_rel_output_pipeline_story_input_event_script'
# suffix='gen_with_rel_output_pipeline_pretrain_story_input_event_script'
# suffix="gen_with_rel_output_pipeline_story_input_event_script_update_otrl_sentence_mean_try_2.5_0.25_5_input_new_similarity_0.7"
# suffix="gen_with_rel_output_pipeline_pretrain_story_input_event_script_update_otrl_sentence_mean_try_2.5_0.25_5_input_new_similarity_0.7"
# suffix="gen_with_rel_output_pipeline_story_input_update_event_script_rl_sentence"

root="../output_final/${task}/"   # 测试的普通版本就放这里

for l in "${lrs[@]}"
do
  for s in "${batch[@]}"
  do
      python eval_story_generation_pipeline.py \
      --data_dir "../data/" \
      --model ${model} \
      --task_name  ${task} \
      --file_suffix "_story_generation_all_complete_final.json" \
      --device_num ${device} \
      --eval_batch_size ${s} \
      --num_train_epochs 10 \
      --learning_rate ${l} \
      --input_event_num ${event_num} \
      --seed ${seed} \
      --model_dir "${root}/${model_name}_batch_${s}_lr_${l}_seed_${seed}_event_${event_num}_${suffix}/"
  done
done