task="story_generation"
lrs=(5e-5)
batch=(10)
seeds=(5)   # set the random seed
device="0"  # set the gou id
model=""   # the path for pretrained bart-base model in huggingface
model_name="facebook/bart-base"
event_num=0
alpha=2.5
beta=0.25
gamma=0.7
iteration=5
root="../output_final/"
load_model_dir=""    # the path for bidirectional pretraining path

# suffix="gen_with_rel_output_pipeline_story_input_event_script_update_otrl_sentence_mean_try_2.5_0.25_5_input_new_similarity_0.7"
# suffix="gen_with_rel_output_pipeline_pretrain_story_input_event_script_update_otrl_sentence_mean_try_2.5_0.25_5_input_new_similarity_0.7"

for l in "${lrs[@]}"
do
  for s in "${batch[@]}"
  do
	    for seed in "${seeds[@]}"
	    do
          nohup python run_story_generation_pipeline_otrl.py \
          --data_dir "../data/" \
          --model ${model} \
          --save_model \
          --task_name  ${task} \
          --file_suffix "_new_full_stop_story_generation_all_complete.json" \
          --device_num ${device} \
          --train_batch_size ${s} \
          --eval_batch_size ${s} \
          --num_train_epochs 10 \
          --max_seq_length 72 \
          --do_train \
          --do_eval \
          --alpha ${alpha} \
          --beta ${beta} \
          --gamma ${gamma} \
          --iteration ${iteration} \
          --input_event_num ${event_num} \
          --learning_rate ${l} \
          --seed ${seed} \
          --output_dir "${root}${task}/${model_name}_batch_${s}_lr_${l}_seed_${seed}_event_${event_num}_${suffix}" \
          > ../logs_final/${task}/${model_name}_batch_${s}_lr_${l}_seed_${seed}_event_${event_num}_${suffix} 2>&1 &
      done
    done
done