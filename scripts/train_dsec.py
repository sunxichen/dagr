# avoid matlab error on server
import os
os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'

# HDF5: avoid missing VOL/filter plugin issues in worker processes
for k in ('HDF5_VOL_CONNECTOR', 'HDF5_PLUGIN_PATH'):
    os.environ.pop(k, None)

import hdf5plugin
import h5py

import torch
import math
import torch.distributed as dist
import tqdm
import wandb
from pathlib import Path
import argparse

from torch_geometric.data import DataLoader
from torch.utils.data import Subset
# --- [ADDED] Mixed Precision Imports ---
from torch.cuda.amp import autocast, GradScaler
# ---------------------------------------

from dagr.utils.logging import Checkpointer, set_up_logging_directory, log_hparams
from dagr.utils.buffers import DetectionBuffer
from dagr.utils.args import FLAGS
from dagr.utils.learning_rate_scheduler import LRSchedule

from dagr.data.augment import Augmentations
from dagr.utils.buffers import format_data
from dagr.data.dsec_data import DSEC

from dagr.model.networks.dagr import DAGR
from dagr.model.networks.ema import ModelEMA

# torch.autograd.set_detect_anomaly(True)
def gradients_broken(model):
    valid_gradients = True
    for name, param in model.named_parameters():
        if param.grad is not None:
            # valid_gradients = not (torch.isnan(param.grad).any() or torch.isinf(param.grad).any())
            valid_gradients = not (torch.isnan(param.grad).any())
            if not valid_gradients:
                break
    return not valid_gradients

def fix_gradients(model):
    for name, param in model.named_parameters():
        if param.grad is not None:
            # param.grad = torch.nan_to_num(param.grad, nan=0.0)
            param.grad.nan_to_num_(nan=0.0)


def train(loader: DataLoader,
          model: torch.nn.Module,
          ema: ModelEMA,
          scheduler: torch.optim.lr_scheduler.LambdaLR,
          optimizer: torch.optim.Optimizer,
          args: argparse.ArgumentParser,
          scaler: GradScaler, # --- [ADDED] scaler arg
          run_name=""):

    model.train()

    total_loss_sum = 0.0
    num_steps = 0

    iterator = loader
    if getattr(args, 'is_main_process', True):
        iterator = tqdm.tqdm(loader, desc=f"Training {run_name}")

    accum_steps = getattr(args, 'accum_steps', 1)
    step_in_accum = 0
    optimizer.zero_grad(set_to_none=True)

    printed_unused_once = False
    for i, data in enumerate(iterator):
        data = data.cuda(non_blocking=True)
        data = format_data(data)

        # --- [MODIFIED] AMP Autocast Context ---
        with autocast():
            model_outputs = model(data)

            loss_dict = {k: v for k, v in model_outputs.items() if "loss" in k}
            loss = loss_dict.pop("total_loss")

            loss = loss / accum_steps
        # ---------------------------------------
        
        # torch.autograd.set_detect_anomaly(True)
        # --- [MODIFIED] Scaled Backward ---
        scaler.scale(loss).backward()
        # ----------------------------------

        # Debug: list parameters without gradients
        # if (not printed_unused_once) and getattr(args, 'debug_unused_params', False) and getattr(args, 'is_main_process', True):
        #     try:
        #         # unwrap ddp if needed
        #         model_for_debug = model.module if hasattr(model, 'module') else model
        #         unused = []
        #         for idx, (name, p) in enumerate(model_for_debug.named_parameters()):
        #             if p.requires_grad and (p.grad is None):
        #                 unused.append((idx, name, tuple(p.shape)))
        #         if len(unused) == 0:
        #             print("[UnusedParamDebug] All parameters received gradients in this iteration.")
        #         else:
        #             print("[UnusedParamDebug] Parameters without gradient (index, name, shape):")
        #             for idx, name, shape in unused[:256]:  # cap print length
        #                 print(f"  {idx}: {name} {shape}")
        #             if len(unused) > 256:
        #                 print(f"  ... and {len(unused) - 256} more")
        #         printed_unused_once = True
        #     except Exception as e:
        #         print(f"[UnusedParamDebug][WARN] failed to enumerate unused params: {repr(e)}")

        step_in_accum += 1
        if step_in_accum == accum_steps:
            # --- [MODIFIED] Unscale before clip ---
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_value_(model.parameters(), args.clip)
            fix_gradients(model)
            
            # --- [MODIFIED] Scaler Step ---
            scaler.step(optimizer)
            scaler.update()
            # ------------------------------
            
            scheduler.step()
            optimizer.zero_grad(set_to_none=True)
            step_in_accum = 0

            if getattr(args, 'is_main_process', True):
                ema.update(model.module if getattr(args, 'distributed', False) else model)

        training_logs = {f"training/loss/{k}": v for k, v in loss_dict.items()}
        if getattr(args, 'is_main_process', True):
            try:
                current_lr = scheduler.get_last_lr()[-1]
            except Exception:
                current_lr = optimizer.param_groups[0]['lr']
            wandb.log({"training/loss": float(loss.item()) * accum_steps, "training/lr": current_lr, **training_logs})

        # accumulate for epoch mean
        total_loss_sum += float(loss.item()) * accum_steps
        num_steps += 1 if step_in_accum == 0 else 0

        # if torch.cuda.is_available():
        #     torch.cuda.empty_cache()

    # return mean loss for console print
    mean_loss = total_loss_sum / max(1, num_steps)
    return mean_loss

def run_test(loader: DataLoader,
         model: torch.nn.Module,
         dry_run_steps: int=-1,
         dataset="gen1"):

    model.eval()

    # Unwrap Subset to access base dataset attributes
    base_dataset = loader.dataset
    while isinstance(base_dataset, Subset):
        base_dataset = base_dataset.dataset

    mapcalc = DetectionBuffer(height=base_dataset.height, width=base_dataset.width, classes=base_dataset.classes)

    for i, data in enumerate(tqdm.tqdm(loader)):
        data = data.cuda()
        data = format_data(data)

        # --- [MODIFIED] Autocast Inference ---
        with autocast():
            detections, targets = model(data)
        # -------------------------------------
        
        if i % 10 == 0:
            torch.cuda.empty_cache()

        mapcalc.update(detections, targets, dataset, data.height[0], data.width[0])

        if dry_run_steps > 0 and i == dry_run_steps:
            break

    torch.cuda.empty_cache()

    return mapcalc

if __name__ == '__main__':
    import torch_geometric
    import random
    import numpy as np

    seed = 42
    torch_geometric.seed.seed_everything(seed)
    torch.random.manual_seed(seed)
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    args = FLAGS()

    # Distributed init (torchrun)
    args.distributed = bool(getattr(args, 'distributed', False))
    local_rank = 0
    rank = 0
    world_size = 1
    if args.distributed:
        local_rank = int(os.environ.get('LOCAL_RANK', 0))
        rank = int(os.environ.get('RANK', 0))
        world_size = int(os.environ.get('WORLD_SIZE', 1))
        torch.cuda.set_device(local_rank)
        dist.init_process_group(backend='nccl')
    args.is_main_process = (rank == 0)

    # Set up logging directory and wandb only on main process
    if args.is_main_process:
        output_directory = set_up_logging_directory(args.dataset, args.task, args.output_directory, exp_name=args.exp_name)
        log_hparams(args)
        run_id = wandb.run.id
    else:
        # receive run_id from rank0
        run_id = None
    if args.distributed:
        obj_list = [run_id]
        dist.broadcast_object_list(obj_list, src=0)
        run_id = obj_list[0]
    if not args.is_main_process:
        # reconstruct same output directory path
        base_dir = Path(args.output_directory) / args.dataset / args.task
        output_directory = base_dir / run_id
        output_directory.mkdir(parents=True, exist_ok=True)

    augmentations = Augmentations(args)

    print("init datasets")
    dataset_path = args.dataset_directory / args.dataset

    # train_dataset = DSEC(root=dataset_path, split="train", transform=augmentations.transform_training, debug=False,
    #                      min_bbox_diag=15, min_bbox_height=10)
    # test_dataset = DSEC(root=dataset_path, split="val", transform=augmentations.transform_testing, debug=False,
    #                     min_bbox_diag=15, min_bbox_height=10)
    # --- 修正：强制 scale=4 以减少 VRAM 占用 ---
    # 注意：这仅用于 24GB 显存的 OOM 测试
    forced_scale = 4
    print(f"\033[93mWARNING: Forcing data scale to {forced_scale} to fit in 24GB VRAM.\033[0m")
    
    train_dataset = DSEC(root=dataset_path, split="train", transform=augmentations.transform_training, debug=False,
                         min_bbox_diag=15, min_bbox_height=10, scale=forced_scale)
    test_dataset = DSEC(root=dataset_path, split="val", transform=augmentations.transform_testing, debug=False,
                        min_bbox_diag=15, min_bbox_height=10, scale=forced_scale)
    # --- 修正结束 ---

    # Experiment trend modes: fast / mid / full
    # fast: train+val use subsets; evaluate every epoch on fast subset, no full eval
    # mid: train full; eval every epoch on fast subset AND every 3 epochs on full
    # full: train full; eval only every 3 epochs on full (no fast subset)

    EXP_TREND = getattr(args, 'exp_trend', 'fast')
    TRAIN_SUB_N = 5000
    VAL_SUB_N = 1000

    use_train_subset = (EXP_TREND == 'fast')
    use_val_fast_subset = (EXP_TREND in ('fast', 'mid'))
    full_eval_every_n_epochs = 3 if EXP_TREND in ('mid', 'full') else None

    if use_train_subset and len(train_dataset) > TRAIN_SUB_N:
        train_indices = list(range(TRAIN_SUB_N))
        train_dataset = Subset(train_dataset, train_indices)
        print(f"[FastTrend] Using train subset: {len(train_indices)} / original", flush=True)

    # Always have full validation dataset
    test_dataset_full = test_dataset
    test_dataset_fast = None
    if use_val_fast_subset:
        if len(test_dataset) > VAL_SUB_N:
            val_indices = list(range(VAL_SUB_N))
            test_dataset_fast = Subset(test_dataset, val_indices)
            print(f"[FastTrend] Using val subset: {len(val_indices)} / original for frequent eval", flush=True)
        else:
            test_dataset_fast = test_dataset

    # Per-GPU batch semantics: args.batch_size is per-device
    world_size = world_size if args.distributed else 1
    per_device_batch = int(args.batch_size)
    total_batch = per_device_batch * world_size
    accum_steps = getattr(args, 'accum_steps', 1)
    args.accum_steps = int(accum_steps)

    if args.distributed:
        from torch.utils.data.distributed import DistributedSampler
        train_sampler = DistributedSampler(train_dataset, num_replicas=world_size, rank=rank, shuffle=True, drop_last=True)
        train_loader = DataLoader(train_dataset, follow_batch=['bbox', 'bbox0'], batch_size=per_device_batch, shuffle=False, sampler=train_sampler, num_workers=4, drop_last=True, pin_memory=True, persistent_workers=True)
    else:
        train_loader = DataLoader(train_dataset, follow_batch=['bbox', 'bbox0'], batch_size=per_device_batch, shuffle=True, num_workers=4, drop_last=True, pin_memory=True, persistent_workers=True)
    effective_iters_per_epoch = int(math.ceil(len(train_loader) / max(1, accum_steps)))

    # Build validation loaders only on main process
    test_loader_fast = None
    if args.is_main_process:
        if test_dataset_fast is not None:
            sampler = np.random.permutation(np.arange(len(test_dataset_fast)))
            test_loader_fast = DataLoader(test_dataset_fast, sampler=sampler, follow_batch=['bbox', 'bbox0'], batch_size=per_device_batch, shuffle=False, num_workers=4, drop_last=True, pin_memory=True, persistent_workers=True)

        test_loader_full = DataLoader(test_dataset_full, sampler=np.random.permutation(np.arange(len(test_dataset_full))), follow_batch=['bbox', 'bbox0'], batch_size=per_device_batch, shuffle=False, num_workers=4, drop_last=True, pin_memory=True, persistent_workers=True)

    print("init net")
    # load a dummy sample to get height, width
    model = DAGR(args, height=test_dataset.height, width=test_dataset.width)

    num_params = sum([np.prod(p.size()) for p in model.parameters()])
    print(f"Training with {num_params} number of parameters.")

    # print index->name map to correlate with DDP error indices
    # if getattr(args, 'print_param_index_map', False):
    #     try:
    #         for idx, (name, p) in enumerate(model.named_parameters()):
    #             print(f"[ParamIndexMap] {idx}: {name} {tuple(p.shape)}")
    #     except Exception as e:
    #         print(f"[ParamIndexMap][WARN] failed: {repr(e)}")

    device = torch.device(f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    if args.distributed:
        model = torch.nn.parallel.DistributedDataParallel(
            model,
            device_ids=[local_rank],
            output_device=local_rank,
            # find_unused_parameters=True,
        )
    ema = ModelEMA(model.module if args.distributed else model)

    nominal_batch_size = 64
    lr = args.l_r * np.sqrt(args.batch_size) / np.sqrt(nominal_batch_size)
    optimizer = torch.optim.AdamW(list(model.parameters()), lr=lr, weight_decay=args.weight_decay)

    # --- [ADDED] GradScaler for AMP ---
    scaler = GradScaler()
    # ----------------------------------

    lr_func = LRSchedule(warmup_epochs=.3,
                         num_iters_per_epoch=effective_iters_per_epoch,
                         tot_num_epochs=args.tot_num_epochs)

    lr_scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer=optimizer, lr_lambda=lr_func)

    model_for_saving = model.module if args.distributed else model
    checkpointer = Checkpointer(output_directory=output_directory,
                                model=model_for_saving, optimizer=optimizer,
                                scheduler=lr_scheduler, ema=ema,
                                args=args)

    if args.is_main_process:
        start_epoch = checkpointer.restore_if_existing(output_directory, resume_from_best=False)

    start_epoch = 0
    if "resume_checkpoint" in args:
        start_epoch = checkpointer.restore_checkpoint(args.resume_checkpoint, best=False)
        print(f"Resume from checkpoint at epoch {start_epoch}")

    # Warmup evaluation only on main process (skip when --no_eval to avoid early OOM)
    if args.is_main_process and not getattr(args, 'no_eval', False):
        with torch.no_grad():
            if EXP_TREND == 'full':
                warmup_loader = test_loader_full
            else:
                warmup_loader = test_loader_fast if test_loader_fast is not None else test_loader_full
            mapcalc = run_test(warmup_loader, ema.ema, dry_run_steps=1, dataset=args.dataset)
            metrics_pre = mapcalc.compute()
            summary_keys = ['mAP', 'mAP_50', 'mAP_75', 'mAP_S', 'mAP_M', 'mAP_L']
            summary_parts = []
            for k in summary_keys:
                if k in metrics_pre:
                    try:
                        summary_parts.append(f"{k}={metrics_pre[k]:.4f}")
                    except Exception:
                        summary_parts.append(f"{k}={metrics_pre[k]}")
            if summary_parts:
                print("[Eval][Warmup] " + ", ".join(summary_parts), flush=True)

    if args.is_main_process:
        print("starting to train")
    for epoch in range(0, args.tot_num_epochs):
        if args.distributed:
            train_sampler.set_epoch(epoch)
        mean_loss = train(train_loader, model, ema, lr_scheduler, optimizer, args, scaler=scaler, run_name=(wandb.run.name if args.is_main_process else ""))
        if args.is_main_process:
            try:
                current_lr = lr_scheduler.get_last_lr()[-1]
            except Exception:
                current_lr = optimizer.param_groups[0]['lr']
            print(f"[Train][Epoch {epoch}] loss={mean_loss:.6f}, lr={current_lr:.6g}", flush=True)
            checkpointer.checkpoint(epoch, name=f"last_model")

            # Evaluation scheduling per exp_trend (only rank0)
            with torch.no_grad():
                if EXP_TREND == 'fast':
                    if (not getattr(args, 'no_eval', False)) and test_loader_fast is not None:
                        mapcalc = run_test(test_loader_fast, ema.ema, dataset=args.dataset)
                        metrics = mapcalc.compute()
                        checkpointer.process(metrics, epoch)
                elif EXP_TREND == 'mid':
                    if not getattr(args, 'no_eval', False):
                        if (epoch % 3 == 0):
                            mapcalc = run_test(test_loader_full, ema.ema, dataset=args.dataset)
                        else:
                            mapcalc = run_test(test_loader_fast, ema.ema, dataset=args.dataset)
                        metrics = mapcalc.compute()
                        checkpointer.process(metrics, epoch)
                elif EXP_TREND == 'full':
                    if (not getattr(args, 'no_eval', False)) and (epoch % 3 == 0):
                        mapcalc = run_test(test_loader_full, ema.ema, dataset=args.dataset)
                        metrics = mapcalc.compute()
                        checkpointer.process(metrics, epoch)

    if args.distributed:
        dist.destroy_process_group()
