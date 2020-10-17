# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
r"""
Basic training script for PyTorch
"""

# Set up custom environment before nearly anything else is imported
# NOTE: this should be the first import (no not reorder)
from maskrcnn_benchmark.utils.env import setup_environment  # noqa F401 isort:skip

import argparse
import os

import torch
torch.multiprocessing.set_sharing_strategy('file_system')
from maskrcnn_benchmark.config import cfg
from maskrcnn_benchmark.data import make_data_loader
from maskrcnn_benchmark.solver import make_lr_scheduler
from maskrcnn_benchmark.solver import make_optimizer
from maskrcnn_benchmark.engine.inference import inference
from maskrcnn_benchmark.engine.trainer import do_train
from maskrcnn_benchmark.modeling.detector import build_detection_model
from maskrcnn_benchmark.utils.checkpoint import DetectronCheckpointer
from maskrcnn_benchmark.utils.collect_env import collect_env_info
from maskrcnn_benchmark.utils.comm import synchronize, get_rank, get_world_size
from maskrcnn_benchmark.utils.imports import import_file
from maskrcnn_benchmark.utils.logger import setup_logger
from maskrcnn_benchmark.utils.miscellaneous import mkdir, save_config
from .gan_mrcnn import (MaskDiscriminator, BoxDiscriminator, CombinedDiscriminator, 
                        GAN_RCNN, Mask_RCNN, TensorboardLogger)

import logging
import time
import torch.distributed as dist                   
import datetime
from tqdm import tqdm
from maskrcnn_benchmark.utils.metric_logger import MetricLogger

# See if we can use apex.DistributedDataParallel instead of the torch default,
# and enable mixed-precision via apex.amp
try:
    from apex import amp
except ImportError:
    raise ImportError('Use APEX for multi-precision via apex.amp')

def reduce_loss_dict(loss_dict):
    """
    Reduce the loss dictionary from all processes so that process with rank
    0 has the averaged results. Returns a dict with the same fields as
    loss_dict, after reduction.
    """
    world_size = get_world_size()
    if world_size < 2:
        return loss_dict
    with torch.no_grad():
        loss_names = []
        all_losses = []
        for k in sorted(loss_dict.keys()):
            loss_names.append(k)
            all_losses.append(loss_dict[k])
        all_losses = torch.stack(all_losses, dim=0)
        dist.reduce(all_losses, dst=0)
        if dist.get_rank() == 0:
            # only main process gets accumulated, so only divide by
            # world_size in this case
            all_losses /= world_size
        reduced_losses = {k: v for k, v in zip(loss_names, all_losses)}
    return reduced_losses

def make_D_optimizer(cfg, model):
    params = []
    for key, value in model.named_parameters():
        if not value.requires_grad:
            continue
        lr = cfg.SOLVER.BASE_LR/5.0
        weight_decay = cfg.SOLVER.WEIGHT_DECAY
        if "bias" in key:
            lr = cfg.SOLVER.BASE_LR * cfg.SOLVER.BIAS_LR_FACTOR/5.0
            weight_decay = cfg.SOLVER.WEIGHT_DECAY_BIAS
        params += [{"params": [value], "lr": lr, "weight_decay": weight_decay}]

    optimizer = torch.optim.SGD(params, lr, momentum=cfg.SOLVER.MOMENTUM)
    return optimizer

def train(cfg, local_rank, distributed, d_path=None):

    MaskDnet = MaskDiscriminator(nc=256)
    BBoxDnet = BoxDiscriminator(nc=256, ndf=64)
    Dnet = CombinedDiscriminator(MaskDnet, BBoxDnet)
    model = Mask_RCNN(cfg)
    g_rcnn = GAN_RCNN(model, Dnet)

    device = torch.device(cfg.MODEL.DEVICE)
    g_rcnn.to(device)

    g_optimizer = make_optimizer(cfg, model)
    d_optimizer = make_D_optimizer(cfg, Dnet)

    g_scheduler = make_lr_scheduler(cfg, g_optimizer)
    d_scheduler = make_lr_scheduler(cfg, d_optimizer)
    # model.BoxDnet = BBoxDnet

    # Initialize mixed-precision training
    use_mixed_precision = cfg.DTYPE == "float16"
    amp_opt_level = 'O1' if use_mixed_precision else 'O0'
    model, g_optimizer = amp.initialize(model, g_optimizer, opt_level=amp_opt_level)
    Dnet, d_optimizer = amp.initialize(Dnet, d_optimizer, opt_level=amp_opt_level)

    if distributed:
        g_rcnn = torch.nn.parallel.DistributedDataParallel(
                    g_rcnn, device_ids=[local_rank], output_device=local_rank,
                    # this should be removed if we update BatchNorm stats
                    broadcast_buffers=False,
                )

    arguments = {}
    arguments["iteration"] = 0

    output_dir = cfg.OUTPUT_DIR

    save_to_disk = get_rank() == 0
    checkpointer = DetectronCheckpointer(
        cfg, model, g_optimizer, g_scheduler, output_dir, save_to_disk
    )

    extra_checkpoint_data = checkpointer.load(cfg.MODEL.WEIGHT)

    arguments.update(extra_checkpoint_data)

    d_checkpointer = DetectronCheckpointer(
        cfg, Dnet, d_optimizer, d_scheduler, output_dir, save_to_disk
    )

    if d_path:
        d_checkpointer.load(d_path, use_latest=False)

    data_loader = make_data_loader(
            cfg,
            is_train=True,
            is_distributed=distributed,
            start_iter=arguments["iteration"],
        )

    test_period = cfg.SOLVER.TEST_PERIOD
    data_loader_val = make_data_loader(cfg, is_train=False, is_distributed=distributed, is_for_period=True)

    checkpoint_period = cfg.SOLVER.CHECKPOINT_PERIOD

    ## START TRAINING
    logger = logging.getLogger("maskrcnn_benchmark.trainer")
    logger.info("Start training")

    meters = TensorboardLogger(
            log_dir=cfg.OUTPUT_DIR + "/tensorboardX",
            start_iter=arguments['iteration'],
            delimiter="  ")

    max_iter = len(data_loader)
    start_iter = arguments["iteration"]
    g_rcnn.train()
    start_training_time = time.time()
    end = time.time()

    iou_types = ("bbox",)
    if cfg.MODEL.MASK_ON:
        iou_types = iou_types + ("segm",)

    dataset_names = cfg.DATASETS.TEST

    for iteration, (images, targets, _) in enumerate(data_loader, start_iter):

        if any(len(target) < 1 for target in targets):
            logger.error(f"Iteration={iteration + 1} || Image Ids used for training {_} || targets Length={[len(target) for target in targets]}" )
            continue
        data_time = time.time() - end
        iteration = iteration + 1
        arguments["iteration"] = iteration

        images = images.to(device)
        targets = [target.to(device) for target in targets]

        g_loss_dict, d_loss_dict = g_rcnn(images, targets)

        g_losses = sum(loss for loss in g_loss_dict.values())
        d_losses = sum(loss for loss in d_loss_dict.values())

        # reduce losses over all GPUs for logging purposes
        g_loss_dict_reduced = reduce_loss_dict(g_loss_dict)
        g_losses_reduced = sum(loss for loss in g_loss_dict_reduced.values())
        
        d_loss_dict_reduced = reduce_loss_dict(d_loss_dict)
        d_losses_reduced = sum(loss for loss in d_loss_dict_reduced.values())
        
        meters.update(total_g_loss=g_losses_reduced, **g_loss_dict_reduced)
        meters.update(total_d_loss=d_losses_reduced, **d_loss_dict_reduced)

        g_optimizer.zero_grad()
        # Note: If mixed precision is not used, this ends up doing nothing
        # Otherwise apply loss scaling for mixed-precision recipe
        with amp.scale_loss(g_losses, g_optimizer) as g_scaled_losses:
            g_scaled_losses.backward()
        g_optimizer.step()
        g_scheduler.step()
        
        
        d_optimizer.zero_grad()
        # Note: If mixed precision is not used, this ends up doing nothing
        # Otherwise apply loss scaling for mixed-precision recipe
        with amp.scale_loss(d_losses, d_optimizer) as d_scaled_losses:
            d_scaled_losses.backward()
        d_optimizer.step()
        d_scheduler.step()

        batch_time = time.time() - end
        end = time.time()
        meters.update(time=batch_time, data=data_time)

        eta_seconds = meters.time.global_avg * (max_iter - iteration)
        eta_string = str(datetime.timedelta(seconds=int(eta_seconds)))

        if iteration % 20 == 0 or iteration == max_iter:
            logger.info(
                meters.delimiter.join(
                    [
                        "eta: {eta}",
                        "iter: {iter}",
                        "{meters}",
                        "lr: {lr:.6f}",
                        "max mem: {memory:.0f}",
                    ]
                ).format(
                    eta=eta_string,
                    iter=iteration,
                    meters=str(meters),
                    lr=g_optimizer.param_groups[0]["lr"],
                    memory=torch.cuda.max_memory_allocated() / 1024.0 / 1024.0,
                )
            )
        if iteration % checkpoint_period == 0:
            checkpointer.save("model_{:07d}".format(iteration), **arguments)
            d_checkpointer.save("dnet_{:07d}".format(iteration), **arguments)
            
        if data_loader_val is not None and test_period > 0 and iteration % test_period == 0:
            meters_val = MetricLogger(delimiter="  ")
            synchronize()
            _ = inference(  # The result can be used for additional logging, e. g. for TensorBoard
                model,
                # The method changes the segmentation mask format in a data loader,
                # so every time a new data loader is created:
                make_data_loader(cfg, is_train=False, is_distributed=False, is_for_period=True),
                dataset_name="[Validation]",
                iou_types=iou_types,
                box_only=False if cfg.MODEL.RETINANET_ON else cfg.MODEL.RPN_ONLY,
                device=cfg.MODEL.DEVICE,
                expected_results=cfg.TEST.EXPECTED_RESULTS,
                expected_results_sigma_tol=cfg.TEST.EXPECTED_RESULTS_SIGMA_TOL,
                output_folder=cfg.OUTPUT_DIR,
            )
            synchronize()
            model.train()
            with torch.no_grad():
                # Should be one image for each GPU:
                for iteration_val, (images_val, targets_val, _) in enumerate(tqdm(data_loader_val)):
                    images_val = images_val.to(device)
                    targets_val = [target.to(device) for target in targets_val]
                    loss_dict = model(images_val, targets_val)
                    losses = sum(loss for loss in loss_dict.values())
                    loss_dict_reduced = reduce_loss_dict(loss_dict)
                    losses_reduced = sum(loss for loss in loss_dict_reduced.values())
                    meters_val.update(loss=losses_reduced, **loss_dict_reduced)
            synchronize()
            logger.info(
                meters_val.delimiter.join(
                    [
                        "[Validation]: ",
                        "eta: {eta}",
                        "iter: {iter}",
                        "{meters}",
                        "lr: {lr:.6f}",
                        "max mem: {memory:.0f}",
                    ]
                ).format(
                    eta=eta_string,
                    iter=iteration,
                    meters=str(meters_val),
                    lr=g_optimizer.param_groups[0]["lr"],
                    memory=torch.cuda.max_memory_allocated() / 1024.0 / 1024.0,
                )
            )
        
        if iteration == max_iter:
            checkpointer.save("model_final", **arguments)

    total_training_time = time.time() - start_training_time
    total_time_str = str(datetime.timedelta(seconds=total_training_time))
    logger.info(
        "Total training time: {} ({:.4f} s / it)".format(
            total_time_str, total_training_time / (max_iter)
        )
    )

def main():
    parser = argparse.ArgumentParser(description="PyTorch Object Detection Training")
    parser.add_argument(
        "--config-file",
        default="",
        metavar="FILE",
        help="path to config file",
        type=str,
    )
    parser.add_argument("--local_rank", type=int, default=0)
    parser.add_argument(
        "--skip-test",
        dest="skip_test",
        help="Do not test the final model",
        action="store_true",
    )
    parser.add_argument(
        "opts",
        help="Modify config options using the command-line",
        default=None,
        nargs=argparse.REMAINDER,
    )
    parser.add_argument("--d_ckpt", 
        help="The path to the checkpoint for test, default is the latest checkpoint.",
        default=None
    )

    args = parser.parse_args()

    num_gpus = int(os.environ["WORLD_SIZE"]) if "WORLD_SIZE" in os.environ else 1
    args.distributed = num_gpus > 1

    if args.distributed:
        torch.cuda.set_device(args.local_rank)
        torch.distributed.init_process_group(
            backend="nccl", init_method="env://"
        )
        synchronize()

    ## SETUP
    cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    cfg.freeze()

    output_dir = cfg.OUTPUT_DIR
    if output_dir:
        mkdir(output_dir)

    logger = setup_logger("maskrcnn_benchmark", output_dir, get_rank())
    logger.info("Using {} GPUs".format(num_gpus))
    logger.info(args)

    logger.info("Collecting env info (might take some time)")
    logger.info("\n" + collect_env_info())

    logger.info("Loaded configuration file {}".format(args.config_file))
    with open(args.config_file, "r") as cf:
        config_str = "\n" + cf.read()
        logger.info(config_str)
    logger.info("Running with config:\n{}".format(cfg))

    output_config_path = os.path.join(cfg.OUTPUT_DIR, 'config.yml')
    logger.info("Saving config into: {}".format(output_config_path))
    # save overloaded model config in the output directory
    save_config(cfg, output_config_path)

    model = train(cfg, args.local_rank, args.distributed, args.d_ckpt)

if __name__ == "__main__":
    main()