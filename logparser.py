#!/scratch/db4045/capstone_env/bin/python
# coding: utf-8

import os
import sys
import pandas as pd
import numpy as np
import yaml
import click


def logreader(fname):
    with open(fname, 'r') as f:
        line = f.readline()
        if not len(line):
            print("Empty file!")
            return
        while len(line):
            res = line.strip('\n')
            if len(res):
                yield res
            line = f.readline()

def get_epoch_loss(line):
    assert line.startswith('Train Epoch:')
    parts = line.split('Alphas:')
    mainline = parts[0].strip()
    mparts = mainline.split(' ')
    epoch = int(mparts[2])
    loss = float(mparts[-1])
#    if len(parts) > 1:
#        alphas = yaml.load(parts[1].strip())
#    else:
#        alphas = {}
#    return epoch, loss, alphas
    return epoch, loss, {}


def get_test_accuracy_loss(line):
    assert line.startswith('Test set:')
    parts = line.split(' ')
    loss = float(parts[4].strip(','))
    accuracy = float(parts[-1][1:-1])
    return accuracy, loss


def get_experiment_details(exp):
    items = {t.split('=')[0]: t.split('=')[1] for t in exp['name'].split('~')}
    items['seed'] = int(items['seed'])
    items['sparsity'] = int(items['sparsity'])
    items['epochs'] = exp['epochs']
    items['seed-model'] = exp['seed-model'] if 'seed-model' in exp else None
    return items


def get_task_id(line):
    return int(line.split(' ')[-1])


def summarize_details(expno, details, task, epoch, tloss, loss, accuracy, num_masks):
    return (details['id'], expno, details['seed'], details['sparsity'], task, epoch, tloss, loss, accuracy, num_masks, details['seed-model'], )


def logparser(fname, is_basis=False, num_masks=None):
    if is_basis and not num_masks:
        raise ValueError("Cannot have basis experiement with num masks. Please set value for num_masks")
    if not is_basis and num_masks:
        raise ValueError("Cannot have seed experiement with num masks. Do not set a value for num_masks")
    data = [('ID', 'exp_no', 'seed_val', 'sparsity', 'task', 'epochno', 'train_loss', 'test_loss', 'accuracy', 'num_masks', 'seed_model')]
    testdata = [('ID', 'exp_no', 'seed_val', 'sparsity', 'task', 'epochno', 'train_loss', 'test_loss', 'accuracy', 'num_masks', 'seed_model')]
    print("Starting parse for: {}".format(fname))
    it = logreader(fname)
    try:
        firstline = next(it)
        if not firstline.startswith('[{'):
            raise ValueError("Log doesn't start with experiments description!")
        experiments = yaml.load(firstline)
        line = next(it)
        print("Found {} experiments in logfile".format(len(experiments)))
        for expno, exp in enumerate(experiments):
            print("Parsing experiement {}".format(expno))
            details = get_experiment_details(exp)
            exp_data = []
            assert line.startswith('=> Reading YAML config'), "Line: {}".format(line)
            while not line.startswith('Testing'):
                while not line.startswith('Task '):
                    line = next(it)
                task = get_task_id(line)
                line = next(it)
                for epoch in range(1, details['epochs']+1):
                    tloss = 0.0
                    while not line.startswith('Test set: Average loss:'):
                        assert line.startswith('Train Epoch'), "Line: {}".format(line)
                        epoch_log, tloss, alphas = get_epoch_loss(line)
                        assert epoch_log == epoch, "Epochs don't line up for {}, {}, {} \n {}".format(exp, epoch, epoch_log, line)
                        line = next(it)
                    accuracy, loss = get_test_accuracy_loss(line)
                    exp_data.append(summarize_details(expno, details, task, epoch, tloss, loss, accuracy, num_masks))
                    line = next(it)
            while not line.startswith('Testing '):
                line = next(it)
            test_data = []
            while not line.startswith('=> Reading YAML config'):
                if line.startswith('Testing '):
                    taskidx = int(line.split(' ')[1].strip(':'))
                    line = next(it)
                    accuracy = float(line.split(' ')[-1].strip('(').strip(')').strip('%'))
                    test_data.append(summarize_details(expno, details, taskidx, -1, -1, -1, accuracy, num_masks))
                line = next(it)
            data += exp_data
            testdata += test_data
    except StopIteration as stp:
        pass
    it.close()
    data += exp_data
    testdata += test_data
    testdf = pd.DataFrame(data=testdata[1:], columns=testdata[0])
    testdf = testdf.drop(['train_loss', 'test_loss', 'epochno'], axis=1)
    return pd.DataFrame(data=data[1:], columns=data[0]), testdf


@click.command()
@click.option("--log-file", type=str, required=True, help="Full path to log file")
@click.option("--output-dir", type=str, required=True, help="Output file directory")
@click.option("--is-basis", required=False, default=False, help="Is it a basis experiment?", is_flag=True)
@click.option("--num-masks", type=int, required=False, default=None, help="Num masks if basis experiemnt")
def run(log_file, output_dir, is_basis, num_masks):
    if not os.path.exists(log_file):
        raise ValueError("File not found!: {}".format(log_file))

    df, test = logparser(log_file, is_basis, num_masks)
    outfile = os.path.join(output_dir, '{}_results.csv'.format('_'.join(os.path.basename(log_file).split('_')[:-1])))
    outfile_test = os.path.join(output_dir, '{}_test.csv'.format('_'.join(os.path.basename(log_file).split('_')[:-1])))
    print("Finished parsing log. Writing to: {}".format(outfile))
    df.to_csv(outfile)
    print("Writing to: {}".format(outfile_test))
    test.to_csv(outfile_test)


if __name__ == "__main__":
    run()
