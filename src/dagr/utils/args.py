import argparse
import yaml

from pathlib import Path


def BASE_FLAGS():
    parser = argparse.ArgumentParser("")
    parser.add_argument('--dataset_directory', type=Path, default=argparse.SUPPRESS, help="Path to the directory containing the dataset.")
    parser.add_argument('--output_directory', type=Path, default=argparse.SUPPRESS, help="Path to the logging directory.")
    parser.add_argument("--checkpoint", type=Path, default=argparse.SUPPRESS, help="Path to the directory containing the checkpoint.")
    parser.add_argument("--img_net", default=argparse.SUPPRESS, type=str)
    parser.add_argument("--img_net_checkpoint", type=Path, default=argparse.SUPPRESS)
    # --- Modified for 3-branch hybrid backbone (Fused, RGB, MAD) ---
    parser.add_argument("--mad_flow_checkpoint", type=Path, default=argparse.SUPPRESS,
                        help="Path to the pre-trained EVFlowNet checkpoint for the MAD branch.")
    parser.add_argument("--no_load_mad_flow", action="store_true",
                        help="Do not load a pre-trained EVFlowNet checkpoint; use random weights instead for debugging.")
    parser.add_argument("--use_checkpointing", action="store_true",
                        help="Use activation checkpointing to save VRAM at the cost of computation.")
    # --- Modified end ---

    parser.add_argument("--config", type=Path, default="../config/detection.yaml")
    parser.add_argument("--use_image", action="store_true")
    parser.add_argument("--no_events", action="store_true")
    parser.add_argument("--pretrain_cnn", action="store_true")
    parser.add_argument("--keep_temporal_ordering", action="store_true")
    parser.add_argument("--debug_unused_params", action="store_true", help="Print parameters that did not receive gradients after backward() on the first iteration")

    # task params
    parser.add_argument("--task", default=argparse.SUPPRESS, type=str)
    parser.add_argument("--dataset", default=argparse.SUPPRESS, type=str)

    # graph params
    parser.add_argument('--radius', default=argparse.SUPPRESS, type=float)
    parser.add_argument('--time_window_us', default=argparse.SUPPRESS, type=int)
    parser.add_argument('--max_neighbors', default=argparse.SUPPRESS, type=int)
    parser.add_argument('--n_nodes', default=argparse.SUPPRESS, type=int)

    # learning params
    parser.add_argument('--batch_size', default=argparse.SUPPRESS, type=int)

    # network params
    parser.add_argument("--activation", default=argparse.SUPPRESS, type=str, help="Can be one of ['Hardshrink', 'Hardsigmoid', 'Hardswish', 'ReLU', 'ReLU6', 'SoftShrink', 'HardTanh']")
    parser.add_argument("--edge_attr_dim", default=argparse.SUPPRESS, type=int)
    parser.add_argument("--aggr", default=argparse.SUPPRESS, type=str)
    parser.add_argument("--kernel_size", default=argparse.SUPPRESS, type=int)
    parser.add_argument("--pooling_aggr", default=argparse.SUPPRESS, type=str)

    parser.add_argument("--base_width", default=argparse.SUPPRESS, type=float)
    parser.add_argument("--after_pool_width", default=argparse.SUPPRESS, type=float)
    parser.add_argument('--net_stem_width', default=argparse.SUPPRESS, type=float)
    parser.add_argument("--yolo_stem_width", default=argparse.SUPPRESS, type=float)
    parser.add_argument("--num_scales", default=argparse.SUPPRESS, type=int)
    parser.add_argument('--pooling_dim_at_output', default=argparse.SUPPRESS)
    parser.add_argument('--weight_decay', default=argparse.SUPPRESS, type=float)
    parser.add_argument('--clip', default=argparse.SUPPRESS, type=float)

    parser.add_argument('--aug_p_flip', default=argparse.SUPPRESS, type=float)

    # SNN backbone options
    parser.add_argument("--use_snn_backbone", action="store_true", help="Enable SNN backbone defined by a YAML config")
    parser.add_argument("--snn_yaml_path", default=argparse.SUPPRESS, type=str, help="Path to SNN YAML config file")
    parser.add_argument("--snn_scale", default=argparse.SUPPRESS, type=str, help="Model scale for SNN backbone (e.g., s/m/l)")
    parser.add_argument("--snn_temporal_bins", type=int, default=4, help="Temporal bins T for SNN voxelization (default: 4)")
    parser.add_argument("--backbone_type", default=argparse.SUPPRESS, type=str,
                        help="Backbone selector: set to 'sdtv3' to use Spike-Driven Transformer V3")
    parser.add_argument("--sdt_T", type=int, default=argparse.SUPPRESS, help="Time steps for SDT-V3 (default: 4)")
    parser.add_argument("--sdt_repeat_static", action="store_true",
                        help="Repeat static BCHW input across T for SDT-V3")
    parser.add_argument("--sdt_input_t_first", action="store_true",
                        help="Set if SDT-V3 inputs are already (T,B,C,H,W); default assumes (B,T,C,H,W)")
    parser.add_argument("--sdt_checkpoint", action="store_true",
                        help="Force activation checkpointing inside SDT-V3 backbone")
    parser.add_argument("--sdt_in_channels", type=int, default=argparse.SUPPRESS,
                        help="Input channels for SDT-V3 (2 for events, 3 for RGB)")
    parser.add_argument("--sdt_depths", nargs="+", type=int, default=argparse.SUPPRESS,
                        help="Depths (number of blocks) for each stage of SDT-V3 (e.g., 2 2 6 2)")
    parser.add_argument("--sdt_num_heads", type=int, default=argparse.SUPPRESS,
                        help="Number of attention heads for SDT-V3 (default: 8)")
    parser.add_argument("--sdt_mlp_ratio", type=float, default=argparse.SUPPRESS,
                        help="MLP expansion ratio for SDT-V3 (default: 4.0)")
    parser.add_argument("--sdt_embed_dim", nargs="+", type=int, default=argparse.SUPPRESS,
                        help="Embedding dims list for SDT-V3 (len>=4, e.g., 128 256 512 640)")
    parser.add_argument("--sdt_norm", type=float, default=argparse.SUPPRESS,
                        help="Spike normalization (lens) used by SDT-V3 (default: 4.0)")

    return parser

def FLAGS():
    parser = BASE_FLAGS()

    # learning params
    parser.add_argument('--aug_trans', default=argparse.SUPPRESS, type=float)
    parser.add_argument('--aug_zoom', default=argparse.SUPPRESS, type=float)
    parser.add_argument('--exp_name', default=argparse.SUPPRESS, type=str)
    parser.add_argument('--l_r', default=argparse.SUPPRESS, type=float)
    parser.add_argument('--no_eval', action="store_true")
    parser.add_argument('--tot_num_epochs', default=argparse.SUPPRESS, type=int)
    # distributed training switch (torchrun + DDP)
    parser.add_argument('--distributed', action="store_true")
    parser.add_argument('--print_param_index_map', action='store_true', help='Print an index->name map of model parameters on rank0 after model build')

    parser.add_argument('--run_test', action="store_true")

    # experiment trend: controls data subsampling and evaluation frequency
    parser.add_argument('--exp_trend', default='fast', type=str, choices=['fast', 'mid', 'full'])

    parser.add_argument('--num_interframe_steps', type=int, default=10)

    args = parser.parse_args()

    if args.config != "":
        args = parse_config(args, args.config)

    args.dataset_directory = Path(args.dataset_directory)
    args.output_directory = Path(args.output_directory)

    if "checkpoint" in args:
        args.checkpoint = Path(args.checkpoint)
    
    # --- Modified for 3-branch hybrid backbone (Fused, RGB, MAD) ---
    if getattr(args, 'mad_flow_checkpoint', None) is not None:
        args.mad_flow_checkpoint = Path(args.mad_flow_checkpoint)
    # --- Modified end ---

    return args

def FLOPS_FLAGS():
    parser = BASE_FLAGS()

    # for flop eval
    parser.add_argument("--check_consistency", action="store_true")
    parser.add_argument("--dense", action="store_true")

    # for runtime eval
    args = parser.parse_args()

    if args.config != "":
        args = parse_config(args, args.config)

    args.dataset_directory = Path(args.dataset_directory)
    args.output_directory = Path(args.output_directory)

    if "checkpoint" in args:
        args.checkpoint = Path(args.checkpoint)

    # --- Modified for 3-branch hybrid backbone (Fused, RGB, MAD) ---
    if getattr(args, 'mad_flow_checkpoint', None) is not None:
        args.mad_flow_checkpoint = Path(args.mad_flow_checkpoint)
    # --- Modified end ---

    return args


def parse_config(args: argparse.ArgumentParser, config: Path):
    with config.open() as f:
        config = yaml.load(f, Loader=yaml.SafeLoader)
        for k, v in config.items():
            # Prefer command-line flags over YAML. Only set values not provided via CLI.
            if not hasattr(args, k):
                setattr(args, k, v)
        return args
