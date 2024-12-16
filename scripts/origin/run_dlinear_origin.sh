features=M
model_name=DLinear
checkpoints=~/checkpoints/origin/DLinear
results=~/results/origin/DLinear

gpu=0
itr=5
seq_len=336

for pred_len in 96 192 336 720; do
  log_dir="~/logs/origin/${model_name}/ETTh1"
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
    --train_epochs 10 \
    --patience 3 \
    --lradj type1 \
    --learning_rate 0.005 \
    --batch_size 32 \
    --task "origin" \
  > "$log_file" 2>&1      
done

for pred_len in 96 192 336 720; do
  log_dir="~/logs/origin/${model_name}/ETTh2"
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
    --train_epochs 10 \
    --patience 3 \
    --lradj type1 \
    --learning_rate 0.05 \
    --batch_size 32 \
    --task "origin" \
  > "$log_file" 2>&1      
done

for pred_len in 96 192 336 720; do
  log_dir="~/logs/origin/${model_name}/ETTm1"
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
    --train_epochs 10 \
    --patience 3 \
    --lradj type1 \
    --learning_rate 0.0001 \
    --batch_size 8 \
    --task "origin" \
  > "$log_file" 2>&1      
done

for pred_len in 96 192; do
  log_dir="~/logs/origin/${model_name}/ETTm2"
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
    --enc_in 7 \
    --train_epochs 10 \
    --patience 3 \
    --lradj type1 \
    --learning_rate 0.001 \
    --batch_size 32 \
    --task "origin" \
  > "$log_file" 2>&1      
done

for pred_len in 336 720; do
  log_dir="~/logs/origin/${model_name}/ETTm2"
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
    --enc_in 7 \
    --train_epochs 10 \
    --patience 3 \
    --lradj type1 \
    --learning_rate 0.01 \
    --batch_size 32 \
    --task "origin" \
  > "$log_file" 2>&1      
done

for pred_len in 96 192 336 720; do
  log_dir="~/logs/origin/${model_name}/weather"
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
    --train_epochs 10 \
    --patience 3 \
    --lradj type1 \
    --learning_rate 0.0001 \
    --batch_size 32 \
    --task "origin" \
  > "$log_file" 2>&1      
done

for pred_len in 96 192 336 720; do
  log_dir="~/logs/origin/${model_name}/electricity"
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
    --train_epochs 10 \
    --patience 3 \
    --lradj type1 \
    --learning_rate 0.001 \
    --batch_size 16 \
    --task "origin" \
  > "$log_file" 2>&1      
done

seq_len=104
for pred_len in 24 36 48 60; do
  log_dir="~/logs/origin/${model_name}/illness"
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
    --train_epochs 10 \
    --patience 3 \
    --lradj type1 \
    --learning_rate 0.01 \
    --batch_size 32 \
    --task "origin" \
  > "$log_file" 2>&1      
done