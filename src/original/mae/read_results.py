import os
import re
import argparse
import numpy as np


def print_std(accs, stds, categories, append_mean=False):
    category_line = ' '.join(categories)
    if append_mean:
        category_line += ' Mean'
    
    line = ''
    if stds is None:
        for acc in accs:
            line += '{:0.1f} '.format(acc)
    else:
        for acc, std in zip(accs, stds):
            line += '{:0.1f}±{:0.1f} '.format(acc, std)
    
    if append_mean:
        line += '{:0.1f}'.format(sum(accs) / len(accs))
    print(category_line)
    print(line)


def read_bio(args):
    args.begin = max(1, args.begin)
    with open(args.path, 'r') as f:
        lines = f.readlines()
    lines = lines[args.begin-1:args.end]
    lines = [line.strip() for line in lines]
    lines = [line for line in lines if len(line) > 0]

    regex = re.compile('Seed-(\d); Epoch-(\d+); 0 ([.\d]+) ([.\d]+) ([.\d]+)')
    test_hard_acc_list = []
    for line in lines:
        result = regex.match(line)
        if result:
            seed = int(result.group(1))
            test_easy_acc = float(result.group(4)) * 100
            test_hard_acc = float(result.group(5)) * 100
            test_hard_acc_list.append(test_hard_acc)
    test_hard_acc_list = np.asarray(test_hard_acc_list)
    print(test_hard_acc_list)
    print('{:0.1f}±{:0.1f}'.format(test_hard_acc_list.mean(), test_hard_acc_list.std()))


def read_chem(args):
    args.begin = max(1, args.begin)
    with open(args.path, 'r') as f:
        lines = f.readlines()
        lines = lines[args.begin-1:args.end]
        lines = [line.strip().split() for line in lines]
        lines = [line for line in lines if len(line) > 0]

    model2lines = {}
    for line in lines:
        if len(line) == 5:
            dataset, model_path, seed, val_roc, test_roc = line
        elif len(line) == 4:
            dataset, seed, val_roc, test_roc = line
            model_path = ''
        else:
            raise NotImplementedError
        if model_path not in model2lines:
            model2lines[model_path] = []
        model2lines[model_path].append(line)
    for model, lines in model2lines.items():
        dataset2rocs = {}
        for line in lines:
            if len(line) == 5:
                dataset, model_path, seed, val_roc, test_roc = line
            elif len(line) == 4:
                dataset, seed, val_roc, test_roc = line
                model_path = ''
            else:
                raise NotImplementedError
            test_roc = float(test_roc)
            if dataset in dataset2rocs:
                dataset2rocs[dataset][seed] = test_roc
            else:
                dataset2rocs[dataset] = {seed: test_roc}

        dataset2results = {}
        for dataset, rocs in dataset2rocs.items():
            rocs = np.asarray(list(rocs.values()))
            roc_mean = rocs.mean()
            roc_std = rocs.std()
            dataset2results[dataset] = (roc_mean, roc_std)
        
        dataset_list = ['BBBP', 'Tox21', 'ToxCast',
                        'SIDER', 'ClinTox', 'MUV', 'HIV', 'BACE']
        dataset_list = [dataset.lower() for dataset in dataset_list]
        rocs = [dataset2results.get(dataset, [0,0])[0]*100 for dataset in dataset_list]
        stds = [dataset2results.get(dataset, [0,0])[1]*100 for dataset in dataset_list]
        print(f'Performance for {model}')
        print_std(rocs, stds, dataset_list, True)


def read_chem_dir(args):
    log_dir = args.path
    acc_mean_list = []
    acc_std_list = []
    dataset_list = ['bbbp', 'tox21', 'toxcast', 'sider', 'clintox', 'muv', 'hiv', 'bace']
    for dataset in dataset_list:
        log_name = f'linear_{dataset}_log_100.txt' if args.scheme_prefix is None else f'{args.scheme_prefix}_linear_{dataset}_log_100.txt'
        log_path = os.path.join(log_dir, log_name)
        if not os.path.exists(log_path):
            acc_mean_list.append(0)
            acc_std_list.append(0)
            continue
        with open(log_path, 'r') as f:
            lines = f.readlines()
            lines = [line.strip().split() for line in lines]
        test_acc_list = []
        for line in lines:
            if len(line) < 3:
                continue
            if line[0] == 'Epoch' and line[1] == f'{args.epoch}:' and len(line) >= 4:
                test_acc = float(line[-1])
                test_acc_list.append(test_acc)
        if args.select_half:
            test_acc_list = np.asarray(test_acc_list[:5])
            test_acc_mean = np.mean(test_acc_list[:5])
        else:
            test_acc_list = np.asarray(test_acc_list)
            test_acc_mean = np.mean(test_acc_list)
        acc_std = np.std(test_acc_list)
        test_acc_mean = round(100 * test_acc_mean, 1)
        acc_std = round(100 * acc_std, 1)
        acc_mean_list.append(test_acc_mean)
        acc_std_list.append(acc_std)
    print_std(acc_mean_list, acc_std_list, dataset_list, True)

def read_chem_dir_val(args):
    log_dir = args.path
    acc_mean_list = []
    acc_std_list = []
    dataset_list = ['bbbp', 'tox21', 'toxcast', 'sider', 'clintox', 'muv', 'hiv', 'bace']
    for dataset in dataset_list:
        log_name = f'linear_{dataset}_log_100.txt' if args.scheme_prefix is None else f'{args.scheme_prefix}_linear_{dataset}_log_100.txt'
        log_path = os.path.join(log_dir, log_name)
        if not os.path.exists(log_path):
            acc_mean_list.append(0)
            acc_std_list.append(0)
            continue
        with open(log_path, 'r') as f:
            lines = f.readlines()
            lines = [line.strip().split() for line in lines]
        test_acc_list = []
        best_val = 0
        best_test = -1
        for line in lines:
            if len(line) < 3:
                continue
            if line[0] == 'Epoch' and len(line) >= 4:
                if int(line[1][:-1]) == 1:
                    if best_test > 0:
                        test_acc_list.append(best_test)
                    best_val = 0
                    best_test = 0
                val_acc = float(line[-2])
                test_acc = float(line[-1])
                if val_acc > best_val:
                    best_val = val_acc
                    best_test = test_acc
        test_acc_list = np.asarray(test_acc_list) # shape = [10, 100]
        test_acc_mean = np.mean(test_acc_list)
        acc_std = np.std(test_acc_list)
        test_acc_mean = round(100 * test_acc_mean, 1)
        acc_std = round(100 * acc_std, 1)
        acc_mean_list.append(test_acc_mean)
        acc_std_list.append(acc_std)
    print_std(acc_mean_list, acc_std_list, dataset_list, True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Performance Reader')
    parser.add_argument('--dataset', type=str, default='chem',
                        help='The output comes from which dataset?')
    parser.add_argument('--path', type=str, help='Performance Log')
    parser.add_argument('--epoch', type=int, default=100)
    parser.add_argument('--mode', type=str,)
    parser.add_argument('--scheme_prefix', type=str, default=None)
    parser.add_argument('--begin', type=int, default=0, help='the line in which you begin to count')
    parser.add_argument('--end', type=int, default=100000, help='the line in which you stop to count')
    parser.add_argument('--select_half', default=False, action='store_true')
    args = parser.parse_args()

    if args.dataset == 'bio':
        read_bio(args)
    elif args.dataset == 'chem':
        if args.mode == 'dir':
            read_chem_dir(args)
        elif args.mode == 'log':
            read_chem(args)
        elif args.mode == 'read_dir':
            print(f'Log: {args.path} -- {args.scheme_prefix}')
            for epoch in [10, 20, 30, 40, 50, 60, 70, 80, 90, 100]:
                args.epoch = epoch
                print(f'Epoch {epoch}:', end='')
                read_chem_dir(args)
        elif args.mode == 'read_dir_val':
            print(f'Log: {args.path} -- {args.scheme_prefix}')
            read_chem_dir_val(args)
        else:
            raise NotImplementedError
    else:
        raise NotImplementedError()
