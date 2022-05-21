for MODEL in kwta distillation pang tws
do
    echo "Computing indicators for model $MODEL"
    python3 -m src.ioaf_demo --model $MODEL --samples 2
done