# run on cuda:1, if use 0 and 2 then CUDA_VISIBLE_DEVICES=0,2 
CUDA_VISIBLE_DEVICES=1 python -m open_clip_train.main \
  --train-data "../mydataset/sample_data/my_sample.tar" \
  --pretrained "/projects/chimera/nobackup/wliu25/ckpts/pretrain/ckpt_lshort_bs128/eval/training_99999/teacher_checkpoint.pth" \
  --dataset-type webdataset \
  --warmup 0 \
  --batch-size 2 \
  --lr 1e-5 \
  --wd 0.0 \
  --epochs 1 \
  --workers 0 \
  --model ViT-DinoV2 \
  --train-num-samples 2 \
  --logs ./logs_debug \
  --report-to none