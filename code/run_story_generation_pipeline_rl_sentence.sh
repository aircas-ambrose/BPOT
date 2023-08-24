task="story_generation"
lrs=(5e-5)
batch=(10)
seeds=(5)
device="0"
model="/workspace/huggingface/hub/models--facebook--bart-base/snapshots/84358834e73de6a82c22cec1d90eb45ef4f6eba5"
model_name="facebook/bart-base"
event_num=0
root="../output_final/"

suffix="gen_with_rel_output_pipeline_story_input_update_event_script_rl_sentence"   # 采用event_script的方式

for l in "${lrs[@]}"
do
  for s in "${batch[@]}"
  do
	    for seed in "${seeds[@]}"
	    do
          nohup python run_story_generation_pipeline_rl_sentence.py \
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
          --input_event_num ${event_num} \
          --learning_rate ${l} \
          --seed ${seed} \
          --output_dir "${root}${task}/${model_name}_batch_${s}_lr_${l}_seed_${seed}_event_${event_num}_${suffix}" \
          > ../logs_final/${task}/${model_name}_batch_${s}_lr_${l}_seed_${seed}_event_${event_num}_${suffix} 2>&1 &
      done
    done
done
