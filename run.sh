exp='sund_corr2'
nh=256
nl=10
model='sund_corr'
input_dim=28 #28, 30, 34, 

python scripts/train.py n_epochs=50 \
    learning_rate=5e-4 \
    device=cuda \
    batch_size=32 \
    exp_name=${exp} \
    model=${model} \
    model.weight_decay=0.01 \
    network.hidden_dim=${nh} \
    network.n_layers=${nl} \
    network.input_dim=${input_dim}

#python scripts/forecast.py \
#    exp_name=${exp} \
#    model=${model} \
#    network.hidden_dim=${nh} \
#    network.n_layers=${nl} \
#    network.input_dim=${input_dim} \
#    +suite=val
    

#python scripts/evaluate.py \
#    --prediction_dir data/forecasts/${exp} \
#    --to_json \
#    --output_path scores/${exp}.json