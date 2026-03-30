import os
import cv2
import random
import numpy as np
import torch
import argparse
import sys
from pathlib import Path
from shutil import copyfile
from src.config import Config
import src.muralnet

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from repo_layout import get_realistic_data_root, pipeline_result_root, resolve_data_subdir

def main(mode=None):
    r"""starts the model

    Args:
        mode (int): 1: train, 2: test, 3: eval, reads from config file if not specified
    """

    config = load_config(mode)


    # cuda visble devices
    os.environ['CUDA_VISIBLE_DEVICES'] = ','.join(str(e) for e in config.GPU)


    # init device
    print("torch.cuda.is_available:")
    print(torch.cuda.is_available())
    if torch.cuda.is_available():
        # config.DEVICE = torch.device("cuda")
        config.DEVICE = torch.device("cuda:{}".format(config.GPU[0]))
        # torch.backends.cudnn.benchmark = True   # cudnn auto-tuner
    else:
        config.DEVICE = torch.device("cpu")



    # set cv2 running threads to 1 (prevents deadlocks with pytorch dataloader)
    cv2.setNumThreads(0)


    # initialize random seed
    torch.manual_seed(config.SEED)
    torch.cuda.manual_seed_all(config.SEED)
    np.random.seed(config.SEED)
    random.seed(config.SEED)

    # build the model and initialize
    model = src.muralnet.MuralNet(config)
    model.load()
    print("model loaded")

    # model training
    if config.MODE == 1:
        config.print()
        print('\nstart training...\n')
        model.train()

    # model test
    elif config.MODE == 2:
        print('\nstart testing...\n')
        model.test()

    # eval mode
    else:
        print('\nstart eval...\n')
        model.eval()


def load_config(mode=None):
    r"""loads model config

    Args:
        mode (int): 1: train, 2: test, 3: eval, reads from config file if not specified
    """

    pipeline_root = Path(__file__).resolve().parent
    default_checkpoints_path = pipeline_root / 'checkpoints'
    parser = argparse.ArgumentParser()
    parser.add_argument('--path', '--checkpoints', type=str, default=str(default_checkpoints_path), help='model checkpoints path (default: ./checkpoints)')
    parser.add_argument('--model',default=2, type=int, choices=[2],)
    parser.add_argument('--data-root', type=Path, default=get_realistic_data_root(), help='shared mural dataset root')
    parser.add_argument('--input-subdir', type=str, default='damaged_albedo', help='dataset subdir used for model input')
    parser.add_argument('--edge-subdir', type=str, default='edges', help='dataset subdir used for edge guidance')
    parser.add_argument('--mask-subdir', type=str, default='masks', help='dataset subdir used for masks')
    parser.add_argument('--output-subdir', type=str, default='result-damaged', help='result subdirectory under pipelines/muralnet/result')
    parser.add_argument('--input', type=str, default=None, help='path to the input images directory or an input image')
    parser.add_argument('--mask', type=str, default=None, help='path to the masks directory or a mask file')
    parser.add_argument('--edge', type=str, default=None, help='path to the edges directory or an edge file')
    parser.add_argument('--merged_output', type=str, default=None, help='path to the output directory')
    # parser.add_argument('--output', type=str, default='./checkpoints/test/output', help='path to the output directory')

    args = parser.parse_args()
    config_path = os.path.join(args.path, 'config.yml')
    default_input = str(resolve_data_subdir(args.data_root, args.input_subdir))
    default_mask = str(resolve_data_subdir(args.data_root, args.mask_subdir))
    default_edge = str(resolve_data_subdir(args.data_root, args.edge_subdir))
    default_output = str(pipeline_result_root(pipeline_root) / args.output_subdir)

    # create checkpoints path if does't exist
    if not os.path.exists(args.path):
        os.makedirs(args.path)

    # copy config template if does't exist
    if not os.path.exists(config_path):
        template_path = default_checkpoints_path / 'config.yml'
        if not template_path.exists():
            raise FileNotFoundError(f"Could not find MuralNet config template at {template_path}")
        copyfile(template_path, config_path)

    # load config file
    config = Config(config_path)

    # train mode
    if mode == 1:
        config.MODE = 1
        if args.model:
            config.MODEL = args.model

    # test mode
    elif mode == 2:
        config.MODE = 2
        config.MODEL = args.model if args.model is not None else 3
        config.INPUT_SIZE = 512

        config.TEST_FLIST = args.input if args.input is not None else default_input
        config.TEST_MASK_FLIST = args.mask if args.mask is not None else default_mask
        config.TEST_EDGE_FLIST = args.edge if args.edge is not None else default_edge
        #
        # if args.output is not None:
        #     config.RESULTS_ORI = args.output

        config.RESULTS = args.merged_output if args.merged_output is not None else default_output

    # eval mode
    elif mode == 3:
        config.MODE = 3
        config.MODEL = args.model if args.model is not None else 3

    return config


if __name__ == "__main__":
    print(torch.cuda.is_available())
    main()
