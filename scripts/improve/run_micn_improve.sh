features=M
model_name=MICN
checkpoints=~/checkpoints/improve/MICN
results=~/results/improve/MICN

gpu=0
itr=5
seq_len=96
label_len=96

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
    --label_len $label_len \
    --pred_len $pred_len \
    --des 'Exp' \
    --gpu $gpu \
    --itr $itr \
    --enc_in 7 \
    --dec_in 7 \
    --c_out 7 \
    --mode regre \
    --conv_kernel 12 16 \
    --d_layers 1 \
    --d_model 512 \
    --train_epochs 100 \
    --patience 3 \
    --lradj type7 \
    --learning_rate 0.001 \
    --batch_size 256 \
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
    --label_len $label_len \
    --pred_len $pred_len \
    --des 'Exp' \
    --gpu $gpu \
    --itr $itr \
    --enc_in 7 \
    --dec_in 7 \
    --c_out 7 \
    --mode regre \
    --conv_kernel 12 16 \
    --d_layers 1 \
    --d_model 512 \
    --train_epochs 100 \
    --patience 3 \
    --lradj type7 \
    --learning_rate 0.005 \
    --batch_size 64 \
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
    --label_len $label_len \
    --pred_len $pred_len \
    --des 'Exp' \
    --gpu $gpu \
    --itr $itr \
    --enc_in 7 \
    --dec_in 7 \
    --c_out 7 \
    --mode regre \
    --conv_kernel 12 16 \
    --d_layers 1 \
    --d_model 512 \
    --freq t \
    --train_epochs 100 \
    --patience 3 \
    --lradj type7 \
    --learning_rate 0.001 \
    --batch_size 16 \
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
    --label_len $label_len \
    --pred_len $pred_len \
    --des 'Exp' \
    --gpu $gpu \
    --itr $itr \
    --enc_in 7 \
    --dec_in 7 \
    --c_out 7 \
    --mode regre \
    --conv_kernel 12 16 \
    --d_layers 1 \
    --d_model 512 \
    --freq t \
    --train_epochs 100 \
    --patience 3 \
    --lradj type7 \
    --learning_rate 0.001 \
    --batch_size 32 \
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
    --label_len $label_len \
    --pred_len $pred_len \
    --des 'Exp' \
    --gpu $gpu \
    --itr $itr \
    --enc_in 21 \
    --dec_in 21 \
    --c_out 21 \
    --mode regre \
    --conv_kernel 12 16 \
    --d_layers 1 \
    --d_model 512 \
    --freq t \
    --train_epochs 100 \
    --patience 3 \
    --lradj type7 \
    --learning_rate 0.001 \
    --batch_size 32 \
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
    --label_len $label_len \
    --pred_len $pred_len \
    --des 'Exp' \
    --gpu $gpu \
    --itr $itr \
    --enc_in 321 \
    --dec_in 321 \
    --c_out 321 \
    --mode regre \
    --conv_kernel 12 16 \
    --d_layers 1 \
    --d_model 512 \
    --train_epochs 100 \
    --patience 3 \
    --lradj type7 \
    --learning_rate 0.001 \
    --batch_size 32 \
    --task "improve" \
  > "$log_file" 2>&1        
done


seq_len=36
label_len=36

for pred_len in 24 36; do
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
    --label_len $label_len \
    --pred_len $pred_len \
    --des 'Exp' \
    --gpu $gpu \
    --itr $itr \
    --enc_in 7 \
    --dec_in 7 \
    --c_out 7 \
    --mode regre \
    --conv_kernel 18 12 \
    --d_layers 1 \
    --d_model 64 \
    --freq d \
    --train_epochs 100 \
    --patience 3 \
    --lradj type7 \
    --learning_rate 0.005 \
    --batch_size 8 \
    --task "improve" \
  > "$log_file" 2>&1      
done

for pred_len in 48 60; do
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
    --label_len $label_len \
    --pred_len $pred_len \
    --des 'Exp' \
    --gpu $gpu \
    --itr $itr \
    --enc_in 7 \
    --dec_in 7 \
    --c_out 7 \
    --mode regre \
    --conv_kernel 18 12 \
    --d_layers 1 \
    --d_model 64 \
    --freq d \
    --train_epochs 100 \
    --patience 3 \
    --lradj type7 \
    --learning_rate 0.005 \
    --batch_size 16 \
    --task "improve" \
  > "$log_file" 2>&1      
done