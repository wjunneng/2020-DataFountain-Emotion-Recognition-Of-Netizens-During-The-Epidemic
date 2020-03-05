DATE=$(date +%Y%m%d)
mkdir models/$DATE
export CUDA_VISIBLE_DEVICES=0
for((i=0;i<5;i++));  
do

python run_bert.py \
--config_name bert_config.json \
--model_type bert \
--model_name_or_path premodels/chinese_roberta_wwm_ext_pytorch \
--do_train \
--do_eval \
--do_test \
--data_dir eda/data_$i \
--output_dir models/$DATE/model_bert$i \
--max_seq_length 32 \
--split_num 3 \
--lstm_hidden_size 512 \
--lstm_layers 3 \
--lstm_dropout 0.1 \
--eval_steps 200 \
--per_gpu_train_batch_size 4 \
--gradient_accumulation_steps 4 \
--per_gpu_eval_batch_size 32 \
--learning_rate 5e-6 \
--adam_epsilon 1e-6 \
--weight_decay 0.01 \
--train_steps 200

done  





