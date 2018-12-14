import argparse, sys, os, shutil, time, random,warnings, csv
from random import sample
from tqdm import tqdm
import numpy as np
from sklearn import metrics
import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
from torch.optim.lr_scheduler import MultiStepLR
from torch.utils.data import DataLoader

from model import MTCGCNN
from data import collate_pool, get_train_val_test_loader
from data import CIFData
from plotter import plotMultiGraph, plotGraph


UNDEFINED_INF = 1000000
USE_WEIGHTED_LOSS = False

parser = argparse.ArgumentParser(description='Crystal gated neural networks')
parser.add_argument('modelpath', help='path to the trained model.')
parser.add_argument('cifpath', help='path to the directory of CIF files.')
parser.add_argument(
 '-b',
 '--batch-size',
 default=256,
 type=int,
 metavar='N',
 help='mini-batch size (default: 256)')
parser.add_argument(
 '-j',
 '--workers',
 default=0,
 type=int,
 metavar='N',
 help='number of data loading workers (default: 0)')
parser.add_argument('--disable-cuda', action='store_true', help='Disable CUDA')
parser.add_argument(
 '--print-freq',
 '-p',
 default=10,
 type=int,
 metavar='N',
 help='print frequency (default: 10)')

args = parser.parse_args(sys.argv[1:])
if os.path.isfile(args.modelpath):
    print("=> loading model params '{}'".format(args.modelpath))
    model_checkpoint = torch.load(
     args.modelpath, map_location=lambda storage, loc: storage)
    model_args = argparse.Namespace(**model_checkpoint['args'])
    print("=> loaded model params '{}'".format(args.modelpath))
else:
    print("=> no model params found at '{}'".format(args.modelpath))

args.cuda = not args.disable_cuda and torch.cuda.is_available()
best_mae_error = 1e10

FloatTensor = torch.cuda.FloatTensor
# if torch.cuda.is_available(
# ) else torch.FloatTensor


def main():
    global args, model_args, best_mae_error
    # print(FloatTensor)
    # load data
    dataset = CIFData(args.cifpath)
    collate_fn = collate_pool
    test_loader = DataLoader(
     dataset,
     batch_size=args.batch_size,
     shuffle=True,
     num_workers=args.workers,
     collate_fn=collate_fn,
     pin_memory=args.cuda)

    # build model
    structures, targets, _ = dataset[0]
    orig_atom_fea_len = structures[0].shape[-1]
    nbr_fea_len = structures[1].shape[-1]
    n_p = len(targets)
    properties_loss_weight = torch.ones(n_p)
    model = MTCGCNN(
     orig_atom_fea_len,
     nbr_fea_len,
     atom_fea_len=model_args.atom_fea_len,
     n_conv=model_args.n_conv,
     h_fea_len=model_args.h_fea_len,
     n_p=n_p,
     n_hp=model_args.n_hp,
     dropout=model_args.dropout)

    if args.cuda:
        model.cuda()

    properties_loss_weight = torch.ones(n_p)

    if model_args.weights is not None:
        USE_WEIGHTED_LOSS = True
        properties_loss_weight = FloatTensor(model_args.weights)
        print('Using weights: ', properties_loss_weight)

    collate_fn = collate_pool
    # Only training loader needs to be differentiated, val/test only use full dataset
    # obtain target value normalizer
    if len(dataset) < 2000:
        warnings.warn('Dataset has less than 2000 data points. '
         'Lower accuracy is expected. ')
        sample_data_list = [dataset[i] for i in tqdm(range(len(dataset)))]
    else:
        sample_data_list = [dataset[i] for i in
          tqdm(random.sample(range(len(dataset)), 2000))]
    _, sample_target, _ = collate_pool(sample_data_list)
    normalizer = Normalizer(sample_target)

    if args.cuda:
        criterion = ModifiedMSELoss().cuda()
    else:
        criterion = ModifiedMSELoss()
    if model_args.optimizer == 'SGD':
        optimizer = optim.SGD(
         model.parameters(),
         model_args.lr,
         momentum=model_args.momentum,
         weight_decay=model_args.weight_decay)
    elif model_args.optimizer == 'Adam':
        optimizer = optim.Adam(
         model.parameters(),
         model_args.lr,
         weight_decay=model_args.weight_decay)
    else:
        raise NameError('Only SGD or Adam is allowed as optimizer')



    # optionally resume from a checkpoint
    if os.path.isfile(args.modelpath):
        print("=> loading model '{}'".format(args.modelpath))
        checkpoint = torch.load(
        args.modelpath, map_location=lambda storage, loc: storage)
        model.load_state_dict(checkpoint['state_dict'])
        normalizer.load_state_dict(checkpoint['normalizer'])
        print("=> loaded model '{}' (epoch {}, validation {})".format(
         args.modelpath, checkpoint['epoch'], checkpoint['best_error']))
    else:
        print("=> no model found at '{}'".format(args.modelpath))

    # validate(test_loader, model,n_p, criterion, normalizer, test=True)
    validate(test_loader, model, criterion, normalizer, n_p, properties_loss_weight, test=True, print_checkpoints=True)

def validate(val_loader,
 model,
 criterion,
 normalizer,
 n_p,
 properties_loss_weight,
 test=False,
 print_checkpoints=False):
    batch_time = AverageMeter()
    losses = AverageMeter()
    avg_errors = AverageMeter()

    # error_vector is an average error we see per property. Its dim is (1,n_p)
    error_vector = AverageMeter(is_tensor=True, dimensions=[1, n_p])
    # loss_vector is an average loss we see per property. Its dim is (1,n_p)
    loss_vector = AverageMeter(is_tensor=True, dimensions=[1, n_p])
    if test:
        test_targets = []
        test_preds = []
        test_cif_ids = []

    # switch to evaluate mode
    model.eval()

    end = time.time()
    for i, (input, targets, batch_cif_ids) in enumerate(val_loader):
        batch_size = targets.shape[0]
        with torch.no_grad():
            if args.cuda:
                input_var = (Variable(input[0].cuda(async=True)),
                 Variable(input[1].cuda(async=True)),
                 input[2].cuda(async=True), [
                  crys_idx.cuda(async=True)
                  for crys_idx in input[3]
                 ])
                targets = targets.cuda(async=True)
            else:
                input_var = (Variable(input[0]), Variable(input[1]), input[2],
                 input[3])
        targets_normed = normalizer.norm(targets)
        with torch.no_grad():
            if args.cuda:
                targets_var = Variable(targets_normed.cuda(async=True))
                properties_loss_weight = properties_loss_weight.cuda(
                 async=True)
            else:
                targets_var = Variable(targets_normed)

        # compute output
        output, _ = model(*input_var)
        if USE_WEIGHTED_LOSS:
            mse_loss = [
             properties_loss_weight[i] * criterion(
              output[:, i], targets_var[:, i]) for i in range(n_p)
            ]
            loss = np.sum(mse_loss) / n_p
        else:
            mse_loss = [
             criterion(output[:, i], targets_var[:, i]) for i in range(n_p)
            ]  # for individual properties
            loss = np.sum(mse_loss) / n_p
        mse_vec = torch.stack(mse_loss).detach()

        # measure accuracy and record loss
        if model_args.metric == 'mae':
            error = mae(normalizer.denorm(output.data), targets)
        elif model_args.metric == 'rmse':
            error = torch.sqrt(FloatTensor(mse_loss))
        error_vector.update(error, batch_size)
        loss_vector.update(mse_vec, batch_size)
        avg_error = torch.mean(error)
        losses.update(loss.data.item(), batch_size)
        avg_errors.update(avg_error, batch_size)
        if test:
            test_pred = normalizer.denorm(output.data)
            test_target = targets
            test_preds += test_pred.tolist()
            test_targets += test_target.tolist()
            test_cif_ids += batch_cif_ids

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if print_checkpoints:
            if i % args.print_freq == 0:
                print('Test: [{0}/{1}]\t'
                   'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                   'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                   'ERR {avg_errors.val:.3f} ({avg_errors.avg:.3f})'.format(
                 i,
                 len(val_loader),
                 batch_time=batch_time,
                 loss=losses,
                 avg_errors=avg_errors))

    if test:
        star_label = '**'
        with open(arg.cifpath + '/test_results.csv', 'w') as f:
            writer = csv.writer(f)
            for cif_id, targets, preds in zip(test_cif_ids, test_targets,
             test_preds):
                writer.writerow((cif_id, targets, preds))
        print('Test error per property:',
           error_vector.avg.cpu().numpy().squeeze())
        print('Test loss per property:',
           loss_vector.avg.cpu().numpy().squeeze())
    else:
        star_label = '*'
    print(' {star} ERR {avg_errors.avg:.3f} LOSS {avg_loss.avg:.3f}'.format(
     star=star_label, avg_errors=avg_errors, avg_loss=losses))
    return avg_errors.avg, error_vector, loss_vector



class Normalizer(object):
    """Normalize a Tensor and restore it later."""

    def __init__(self, tensor):
        """
        Tensor is taken as a sample to calculate the mean and std.
        The tensor is of dim (N, n_p) where N is the sample size each with n_p columns
        and the normalization is done across a column of values. So, mean is a tensor
        of dim (n_p)
        """
        self.columns = tensor.shape[1]  # =n_i
        self.mean = FloatTensor([torch.mean(tensor[:, i]) for i in range(self.columns)]).cuda()
        self.std = FloatTensor([torch.std(tensor[:, i]) for i in range(self.columns)]).cuda()
        self.mean = self.mean.cuda()
        self.std = self.std.cuda()
    def norm(self, tensor):
        # print(tensor.is_cuda)
        # print(self.mean.cuda().is_cuda)
        # print(self.std.cuda().is_cuda)
        return (tensor - self.mean.cuda()) / self.std.cuda()

    def denorm(self, normed_tensor):
        return normed_tensor * self.std.cuda() + self.mean.cuda()

    def state_dict(self):
        return {'mean': self.mean, 'std': self.std}

    def load_state_dict(self, state_dict):
        self.mean = state_dict['mean']
        self.std = state_dict['std']


def mae(prediction, target):
    """
    Computes the mean absolute error between prediction and target. If target has float('inf')
    then modifies the calculation accordingly to avoid that

    Parameters
    ----------

    prediction: torch.Tensor (N, n_p)
    target: torch.Tensor (N, n_p)

    Returns
    -------
    torch.Tensor (n_p)
    """
    n_p = target.shape[1]
    return FloatTensor([torch.sum(torch.abs(sanitize(prediction[:,i], target[:,i], return_diff=True)))\
       for i in range(n_p)])/target.shape[0]


def class_eval(prediction, target):
    prediction = np.exp(prediction.numpy())
    target = target.numpy()
    pred_label = np.argmax(prediction, axis=1)
    target_label = np.squeeze(target)
    if prediction.shape[1] == 2:
        precision, recall, fscore, _ = metrics.precision_recall_fscore_support(
         target_label, pred_label, average='binary')
        auc_score = metrics.roc_auc_score(target_label, prediction[:, 1])
        accuracy = metrics.accuracy_score(target_label, pred_label)
    else:
        raise NotImplementedError
    return accuracy, precision, recall, fscore, auc_score


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


class ModifiedMSELoss(torch.nn.Module):
    """
    Given the output and target Variables (each of dim (N, 1)) this finds the
    MSE of these variables.
    """

    # The modification is to ignore the specific rows which
    # have float('inf'). For those rows, just set the value to be zero

    def __init__(self):
        super(ModifiedMSELoss, self).__init__()

    def forward(self, output, target):
        [output_c, target] = sanitize(output, target)
        # now return the MSE Loss
        loss = nn.MSELoss()
        return loss(output_c, target)


def sanitize(input_var, reference, return_diff=False):
    """
    Given two tensor/Variable vectors [dim (k)], sanitize the input_var and reference to zero out the indices
    where the reference has inf. If there are no inf values, return the vectors as is.
    The return_diff is basically a lazy hack to vectorize the mae calculation (Nothing fancy).
    """
    # find indices where float('inf') is present. To facilitate this, clamp to some high values first
    reference = torch.clamp(reference, max=UNDEFINED_INF)
    idx = (reference == UNDEFINED_INF).nonzero()
    idx = idx.view(-1)
    # if idx is valid (i.e. there is some inf values),
    # then replace these indices with zero in both the tensors
    input_var_c = input_var
    if len(idx):
        reference.index_fill_(0, idx, 0)
        input_var_c.index_fill_(0, idx, 0)
    if return_diff:
        return input_var_c - reference
    else:
        return input_var_c, reference


class AverageMeter(object):
    """
    Computes and stores the average and current value. Accomodates both numbers and tensors.
    If the input to be monitored is a tensor, also need the dimensions/shape of the tensor.
    Also, for tensors, it keeps a column wise count for average, sum etc.
    """

    def __init__(self, is_tensor=False, dimensions=None):
        if is_tensor and dimensions is None:
            print("Bad definition of AverageMeter!")
            sys.exit(1)
        self.is_tensor = is_tensor
        self.dimensions = dimensions
        self.reset()

    def reset(self):
        self.count = 0
        if self.is_tensor:
            self.val = torch.zeros(self.dimensions).type(FloatTensor)
            self.avg = torch.zeros(self.dimensions).type(FloatTensor)
            self.sum = torch.zeros(self.dimensions).type(FloatTensor)
        else:
            self.val = 0
            self.avg = 0
            self.sum = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count




def save_checkpoint(state, is_best, filename='checkpoint.pth.tar'):
    torch.save(state, filename)
    if is_best:
        shutil.copyfile(filename, 'model_best.pth.tar')


if __name__ == '__main__':
    main()
