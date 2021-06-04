for MODEL in 0 1 2 3
do
    echo "Computing indicators for model $MODEL"
    python3 -m src.ioaf_demo --model $MODEL --samples 10
done