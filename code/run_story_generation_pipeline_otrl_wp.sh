task="story_generation"
lrs=(2.5e-4)
batch=(64)
seeds=(5)     # set random seed
device="0"    # set gpu id
model=""      # set the pretrained bart-base in huggingface
model_name="facebook/bart-base"
event_num=0
alpha=2.5
beta=0.25
gamma=0.7
iteration=5
root="../output_test/"

load_model_dir=""     # the path for bidirectional pretraining model
# suffix="gen_with_rel_output_pipeline_story_input_event_script_update_otrl_sentence_mean_try_2.5_0.25_5_wp_max500"
# suffix="gen_with_rel_output_pipeline_pretrain_story_input_event_script_update_otrl_sentence_mean_try_2.5_0.25_5_wp_max500"

for l in "${lrs[@]}"
do
  for s in "${batch[@]}"
  do
	    for seed in "${seeds[@]}"
	    do
          nohup python run_story_generation_pipeline_otrl_wp.py \
          --data_dir "../data/" \
          --model ${model} \
          --gradient_accumulation_steps 8 \
          --save_model \
          --task_name  ${task} \
          --gen_storyline_len 512 \
          --file_suffix "_story_generation_wp_max500.json" \
          --device_num ${device} \
          --train_batch_size ${s} \
          --eval_batch_size 8 \
          --num_train_epochs 10 \
          --do_train \
          --do_eval \
          --alpha ${alpha} \
          --beta ${beta} \
          --gamma ${gamma} \
          --iteration ${iteration} \
          --input_event_num ${event_num} \
          --learning_rate ${l} \
          --seed ${seed} \
          --output_dir "${root}/${task}/${model_name}_batch_${s}_lr_${l}_seed_${seed}_event_${event_num}_${suffix}" \
          > ../log_result/${task}/${model_name}_batch_${s}_lr_${l}_seed_${seed}_event_${event_num}_${suffix} 2>&1 &
      done
    done
done