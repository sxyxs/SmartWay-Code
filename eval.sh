CUDA_VISIBLE_DEVICES=0 PYTHONPATH=$(pwd) python run.py \
    --exp_name evaluation \
    --run-type eval \
    --exp-config run_VLNBERT.yaml \
    SIMULATOR_GPU_IDS "[0]" \
    TORCH_GPU_ID 0 \
    TORCH_GPU_IDS "[0]" \
    EVAL.SPLIT rand100 > "evaluation.log" 2>&1 &