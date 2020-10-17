# GAN_Mask-RCNN

Please refer to the follwing links for installation intructions and usage:

  - **maskrcnn-benchmark**: [facebookresearch/maskrcnn-benchmark](https://github.com/facebookresearch/maskrcnn-benchmark)
  - **PreciseRoiPooling**: [vacancy/PreciseRoIPooling](https://github.com/vacancy/PreciseRoIPooling)
  
### Multi-GPU training
We follow the facebook-benchmark command for multi-gpu training. For more information on how to change the config file for different number of gpus, please check the maskrcnn-benchmark link above. (Note that our config is designed to work on 2 GPUs)

```bash
export NGPUS=2
python -m torch.distributed.launch --nproc_per_node=$NGPUS train.py --config-file "configs/e2e_mask_rcnn_R_101_FPN_1x_phone.yaml" 
```
