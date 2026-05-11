ssh -L 1455:localhost:1455 featurize@workspace.featurize.cn -p 51118
51118为端口，自行修改

npm i -g @openai/codex

环境
conda activate mix
或使用 /home/sunl/miniconda3/envs/mix/bin/python
  featurize port export 6006
source /home/featurize/work/envs/mix_backup/bin/activate

  cd /home/featurize/work/mix_v1
  CUDA_VISIBLE_DEVICES=0 python train_open_vocab_v2.py \
    --config config/train_scannet_v2_full_multi_gpu.yaml