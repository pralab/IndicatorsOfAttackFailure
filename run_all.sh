for MODEL in kwta distillation pang tws das guo dnr
do
    echo "Computing indicators for model $MODEL"
    python -m src.ioaf_demo --model $MODEL --samples 10
done