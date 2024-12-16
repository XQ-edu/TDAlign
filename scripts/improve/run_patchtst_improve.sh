features=M
model_name=PatchTST
checkpoints=~/checkpoints/improve/PatchTST
results=~/results/improve/PatchTST

gpu=0
itr=5
seq_len=336

for pred_len in 96 192 336 720; do
  log_dir="~/logs/improve/${model_name}/ETTh1"
  mkdir -p "$log_dir"
  log_file="${log_dir}/${model_name}_ETTh1_${seq_len}_${pred_len}.log"
  echo "Running: Model: $model_name, Data: ETTh1, Seq Length: $seq_len, Pred Length: $pred_len"
  python -u ~/run_longExp.py \
    --is_training 1 \
    --root_path ~/datasets/all_six_datasets/ETT-small \
    --data_path ETTh1.csv \
    --checkpoints $checkpoints/ETTh1 \
    --results $results/ETTh1 \
    --model_id ${model_name}_ETTh1_${seq_len}_${pred_len} \
    --model $model_name \
    --data ETTh1 \
    --features $features \
    --seq_len $seq_len \
    --pred_len $pred_len \
    --des 'Exp' \
    --gpu $gpu \
    --itr $itr \
    --enc_in 7 \
    --e_layers 3 \
    --n_heads 4 \
    --d_model 16 \
    --d_ff 128 \
    --dropout 0.3\
    --fc_dropout 0.3 \
    --head_dropout 0 \
    --patch_len 16 \
    --stride 8 \
    --train_epochs 150 \
    --patience 100 \
    --lradj type8 \
    --learning_rate 0.0001 \
    --batch_size 128 \
    --task "improve" \
  > "$log_file" 2>&1        
done

for pred_len in 96 192 336 720; do
  log_dir="~/logs/improve/${model_name}/ETTh2"
  mkdir -p "$log_dir"
  log_file="${log_dir}/${model_name}_ETTh2_${seq_len}_${pred_len}.log"
  echo "Running: Model: $model_name, Data: ETTh2, Seq Length: $seq_len, Pred Length: $pred_len"
  python -u ~/run_longExp.py \
    --is_training 1 \
    --root_path ~/datasets/all_six_datasets/ETT-small \
    --data_path ETTh2.csv \
    --checkpoints $checkpoints/ETTh2 \
    --results $results/ETTh2 \
    --model_id ${model_name}_ETTh2_${seq_len}_${pred_len} \
    --model $model_name \
    --data ETTh2 \
    --features $features \
    --seq_len $seq_len \
    --pred_len $pred_len \
    --des 'Exp' \
    --gpu $gpu \
    --itr $itr \
    --enc_in 7 \
    --e_layers 3 \
    --n_heads 4 \
    --d_model 64 \
    --d_ff 32 \
    --dropout 0.3 \
    --fc_dropout 0.3 \
    --head_dropout 0 \
    --patch_len 16 \
    --stride 8 \
    --train_epochs 150 \
    --patience 100 \
    --lradj type8 \
    --learning_rate 0.005 \
    --batch_size 256 \
    --task "improve" \
  > "$log_file" 2>&1         
done

for pred_len in 96 192 336 720; do
  log_dir="~/logs/improve/${model_name}/ETTm1"
  mkdir -p "$log_dir"
  log_file="${log_dir}/${model_name}_ETTm1_${seq_len}_${pred_len}.log"
  echo "Running: Model: $model_name, Data: ETTm1, Seq Length: $seq_len, Pred Length: $pred_len"
  python -u ~/run_longExp.py \
    --is_training 1 \
    --root_path ~/datasets/all_six_datasets/ETT-small \
    --data_path ETTm1.csv \
    --checkpoints $checkpoints/ETTm1 \
    --results $results/ETTm1 \
    --model_id ${model_name}_ETTm1_${seq_len}_${pred_len} \
    --model $model_name \
    --data ETTm1 \
    --features $features \
    --seq_len $seq_len \
    --pred_len $pred_len \
    --des 'Exp' \
    --gpu $gpu \
    --itr $itr \
    --enc_in 7 \
    --e_layers 3 \
    --n_heads 16 \
    --d_model 128 \
    --d_ff 256 \
    --dropout 0.2 \
    --fc_dropout 0.2 \
    --head_dropout 0 \
    --patch_len 16 \
    --stride 8 \
    --pct_start 0.4 \
    --train_epochs 150 \
    --patience 20 \
    --lradj TST \
    --learning_rate 0.0001 \
    --batch_size 128 \
    --task "improve" \
  > "$log_file" 2>&1        
done

for pred_len in 96 192 336 720; do
  log_dir="~/logs/improve/${model_name}/ETTm2"
  mkdir -p "$log_dir"
  log_file="${log_dir}/${model_name}_ETTm2_${seq_len}_${pred_len}.log"
  echo "Running: Model: $model_name, Data: ETTm2, Seq Length: $seq_len, Pred Length: $pred_len"
  python -u ~/run_longExp.py \
    --is_training 1 \
    --root_path ~/datasets/all_six_datasets/ETT-small \
    --data_path ETTm2.csv \
    --checkpoints $checkpoints/ETTm2 \
    --results $results/ETTm2 \
    --model_id ${model_name}_ETTm2_${seq_len}_${pred_len} \
    --model $model_name \
    --data ETTm2 \
    --features $features \
    --seq_len $seq_len \
    --pred_len $pred_len \
    --des 'Exp' \
    --gpu $gpu \
    --itr $itr \
    --e_layers 3 \
    --n_heads 16 \
    --d_model 128 \
    --d_ff 256 \
    --dropout 0.2 \
    --fc_dropout 0.2 \
    --head_dropout 0 \
    --patch_len 16 \
    --stride 8 \
    --pct_start 0.4 \
    --train_epochs 150 \
    --patience 20 \
    --lradj TST \
    --learning_rate 0.0001 \
    --batch_size 128 \
    --task "improve" \
  > "$log_file" 2>&1        
done

for pred_len in 96 192 336 720; do
  log_dir="~/logs/improve/${model_name}/weather"
  mkdir -p "$log_dir"
  log_file="${log_dir}/${model_name}_weather_${seq_len}_${pred_len}.log"
  echo "Running: Model: $model_name, Data: weather, Seq Length: $seq_len, Pred Length: $pred_len"
  python -u ~/run_longExp.py \
    --is_training 1 \
    --root_path ~/datasets/all_six_datasets/weather \
    --data_path weather.csv \
    --checkpoints $checkpoints/weather \
    --results $results/weather \
    --model_id ${model_name}_weather_${seq_len}_${pred_len} \
    --model $model_name \
    --data custom \
    --features $features \
    --seq_len $seq_len \
    --pred_len $pred_len \
    --des 'Exp' \
    --gpu $gpu \
    --itr $itr \
    --enc_in 21 \
    --e_layers 3 \
    --n_heads 16 \
    --d_model 128 \
    --d_ff 256 \
    --dropout 0.2 \
    --fc_dropout 0.2 \
    --head_dropout 0 \
    --patch_len 16 \
    --stride 8 \
    --train_epochs 150 \
    --patience 20 \
    --lradj type8 \
    --learning_rate 0.0001 \
    --batch_size 128 \
    --task "improve" \
  > "$log_file" 2>&1        
done

for pred_len in 96 192 336 720; do
  log_dir="~/logs/improve/${model_name}/electricity"
  mkdir -p "$log_dir"
  log_file="${log_dir}/${model_name}_electricity_${seq_len}_${pred_len}.log"
  echo "Running: Model: $model_name, Data: electricity, Seq Length: $seq_len, Pred Length: $pred_len"
  python -u ~/run_longExp.py \
    --is_training 1 \
    --root_path ~/datasets/all_six_datasets/electricity \
    --data_path electricity.csv \
    --checkpoints $checkpoints/electricity \
    --results $results/electricity \
    --model_id ${model_name}_electricity_${seq_len}_${pred_len} \
    --model $model_name \
    --data custom \
    --features $features \
    --seq_len $seq_len \
    --pred_len $pred_len \
    --des 'Exp' \
    --gpu $gpu \
    --itr $itr \
    --enc_in 321 \
    --e_layers 3 \
    --n_heads 16 \
    --d_model 128 \
    --d_ff 256 \
    --dropout 0.2 \
    --fc_dropout 0.2 \
    --head_dropout 0 \
    --patch_len 16 \
    --stride 8 \
    --pct_start 0.2 \
    --train_epochs 150 \
    --patience 10 \
    --lradj TST \
    --learning_rate 0.0001 \
    --batch_size 32 \
    --task "improve" \
  > "$log_file" 2>&1        
done

seq_len=104

for pred_len in 24 36 48 60; do
  log_dir="~/logs/improve/${model_name}/illness"
  mkdir -p "$log_dir"
  log_file="${log_dir}/${model_name}_illness_${seq_len}_${pred_len}.log"
  echo "Running: Model: $model_name, Data: illness, Seq Length: $seq_len, Pred Length: $pred_len"
  python -u ~/run_longExp.py \
    --is_training 1 \
    --root_path ~/datasets/all_six_datasets/illness \
    --data_path national_illness.csv \
    --checkpoints $checkpoints/illness \
    --results $results/illness \
    --model_id ${model_name}_illness_${seq_len}_${pred_len} \
    --model $model_name \
    --data custom \
    --features $features \
    --seq_len $seq_len \
    --pred_len $pred_len \
    --des 'Exp' \
    --gpu $gpu \
    --itr $itr \
    --enc_in 7 \
    --e_layers 3 \
    --n_heads 4 \
    --d_model 16 \
    --d_ff 128 \
    --dropout 0.3 \
    --fc_dropout 0.3 \
    --head_dropout 0 \
    --patch_len 24 \
    --stride 2 \
    --pct_start 0.2 \
    --train_epochs 150 \
    --patience 100 \
    --lradj constant \
    --learning_rate 0.0025 \
    --batch_size 16 \
    --task "improve" \
  > "$log_file" 2>&1        
done
