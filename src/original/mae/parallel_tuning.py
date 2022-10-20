import os
import argparse
import subprocess
import utils
from multiprocessing.pool import ThreadPool


def generate_command(args):
    command = ''
    for k, v in args.__dict__.items():
        if isinstance(v, bool):
            if v == True:
                command += f'--{k} '
            else:
                continue
        elif v == None or v == '':
            continue
        else:
            command += f'--{k} {v} '
    return command


def run_tuning(args, tuning_dataset):
    # command = f'''
    # PYTHONPATH="./" python ./chem/tuning.py --model_file {args.model_file} --split "scaffold" --num_workers 2 --epochs {args.epochs} --name {args.name} --evaluation_mode linear --batch_size 32 --use_schedule --device {args.device} --num_seeds {args.num_seeds} --scheme_prefix {args.scheme_prefix} --mix_gnn --tuning_dataset {args.tuning_dataset}
    # '''
    command = f'python ./tuning.py {generate_command(args)} --tuning_dataset {tuning_dataset}'
    log_path = f'./results/{args.name}/tuning_print_log.txt'
    utils.print_time_info(f'Start Tuning {tuning_dataset}')
    with open(log_path, 'a') as f:
        output = subprocess.call(command, shell=True, stdout=f, stderr=f)
        utils.print_time_info(f'{tuning_dataset} subprocess output: {output}')
    utils.print_time_info(f'Finished tuning {tuning_dataset}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='PyTorch implementation of pre-training of graph neural networks')
    parser.add_argument('--device', type=int, default=0,
                        help='which gpu to use if any (default: 0)')
    parser.add_argument('--batch_size', type=int, default=32,
                        help='input batch size for training (default: 32)')
    parser.add_argument('--epochs', type=int, default=50,
                        help='number of epochs to train (default: 50)')
    parser.add_argument('--lr', type=float, default=0.001,
                        help='learning rate (default: 0.001)')
    parser.add_argument('--decay', type=float, default=0,
                        help='weight decay (default: 0)')
    parser.add_argument('--num_layer', type=int, default=5,
                        help='number of GNN message passing layers (default: 5).')
    parser.add_argument('--emb_dim', type=int, default=300,
                        help='embedding dimensions (default: 300)')
    parser.add_argument('--dropout_ratio', type=float, default=0.5,
                        help='dropout ratio (default: 0.5)')
    parser.add_argument('--graph_pooling', type=str, default="mean",
                        help='graph level pooling (sum, mean, max, set2set, attention)')
    parser.add_argument('--JK', type=str, default="last",
                        help='how the node features across layers are combined. last, sum, max or concat')
    parser.add_argument('--model_file', type=str, default='',
                        help='filename to read the model (if there is any)')
    parser.add_argument('--gnn_type', type=str, default="gin")
    parser.add_argument('--seed', type=int, default=42,
                        help="Seed for splitting dataset.")
    parser.add_argument('--split', type=str, default="species",
                        help='Bio dataset: Random or species split; Chem dataset: random or scaffold or random_scaffold.')
    parser.add_argument('--num_workers', type=int, default=2,
                        help='number of workers for dataset loading')
    parser.add_argument('--use_schedule', action="store_true",
                        default=False, help='Use learning rate scheduler?')
    parser.add_argument('--name', type=str, help='experiment name')
    parser.add_argument('--scheme_prefix', type=str,
                        default=None, help='The name for tuning logs.')
    parser.add_argument('--dataset', type=str, default="chem",
                        help='bio or chem. The domain of the pretrained model.')
    parser.add_argument('--tuning_dataset', type=str, default=None,
                        help='Used only for CHEM dataset. The dataset used for fine-tuning.')
    parser.add_argument('--skip_evaluation', action='store_true',
                        help='Skip evaluation to speed up training.')
    parser.add_argument('--eval_train', action='store_true',
                        help='Evaluate the training dataset or not.')
    # number of random seeds
    parser.add_argument('--num_seeds', type=int, default=10)
    parser.add_argument('--start_seed', type=int, default=0)
    parser.add_argument('--graph_trans', action='store_true', default=False)
    args = parser.parse_args()
    
    tunining_dataset_list = ["muv", "bbbp", "clintox", "hiv", "toxcast", "tox21", "sider", "bace"]
    
    if not os.path.exists('results/{}'.format(args.name)):
        os.makedirs('results/{}'.format(args.name))
    print(str(args))
    num = 2  # set to the number of workers you want (it defaults to the cpu count of your machine)
    tp = ThreadPool(num)
    for tuning_dataset in tunining_dataset_list:
        tp.apply_async(run_tuning, (args, tuning_dataset))
    tp.close()
    tp.join()
    
    