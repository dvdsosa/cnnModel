from __future__ import print_function

import os
import sys
import argparse
import time
import math

#import tensorboard_logger as tb_logger
import torch
import torch.backends.cudnn as cudnn
import matplotlib.pyplot as plt
from torchmetrics import Accuracy, Precision, Recall, F1Score
from torcheval.metrics.functional import (
    multiclass_accuracy,
    multiclass_precision,
    multiclass_recall,
    multiclass_f1_score,
    multiclass_confusion_matrix
)
from main_ce import set_loader
from util import AverageMeter
from util import adjust_learning_rate, warmup_learning_rate, accuracy
from util import set_optimizer, save_model
from networks.resnet_big import SupConResNet, LinearClassifier

from email_me import NotifyUser

try:
    import apex
    from apex import amp, optimizers
    print("APEX is available.")
except ImportError:
    print("APEX is not available.")
    pass

def parse_option():
    parser = argparse.ArgumentParser('argument for training')

    parser.add_argument('--print_freq', type=int, default=10,
                        help='print frequency')
    parser.add_argument('--save_freq', type=int, default=50,
                        help='save frequency')
    parser.add_argument('--batch_size', type=int, default=32,
                        help='batch_size')
    parser.add_argument('--num_workers', type=int, default=8,
                        help='num of workers to use')
    parser.add_argument('--epochs', type=int, default=100,
                        help='number of training epochs')

    # optimization
    parser.add_argument('--learning_rate', type=float, default=0.0125,
                        help='learning rate')
    parser.add_argument('--lr_decay_epochs', type=str, default='60,75,90',
                        help='where to decay lr, can be a list')
    parser.add_argument('--lr_decay_rate', type=float, default=0.2,
                        help='decay rate for learning rate')
    parser.add_argument('--weight_decay', type=float, default=0,
                        help='weight decay')
    parser.add_argument('--momentum', type=float, default=0.9,
                        help='momentum')

    # model dataset
    parser.add_argument('--model', type=str, default='seresnext50timm')
    parser.add_argument('--dataset', type=str, default='path',
                        choices=['cifar10', 'cifar100', 'path'], help='dataset')
    parser.add_argument('--mean', type=str, default='0.0418, 0.0353, 0.0409', help='mean of dataset in path in form of str tuple')
    parser.add_argument('--std', type=str, default='0.0956, 0.0911, 0.0769', help='std of dataset in path in form of str tuple')
    parser.add_argument('--data_folder', type=str, default='/home/dsosatr/tesis/DYB-linearHead/train', help='path to custom train dataset')
    parser.add_argument('--val_folder', type=str, default='/home/dsosatr/tesis/DYB-linearHead/test', help='path to custom test dataset')
    parser.add_argument('--size', type=int, default=224, help='parameter for RandomResizedCrop')

    # other setting
    parser.add_argument('--cosine', action='store_true',
                        help='using cosine annealing')
    parser.add_argument('--warm', action='store_true',
                        help='warm-up for large batch training')

    parser.add_argument('--ckpt', type=str, default='',
                        help='path to pre-trained model')

    opt = parser.parse_args()

    # check if dataset is path that passed required arguments
    if opt.dataset == 'path':
        assert opt.data_folder is not None \
            and opt.mean is not None \
            and opt.std is not None

    # set the path according to the environment
    if opt.data_folder is None:
        opt.data_folder = './datasets'

    # set the path according to the environment
    if opt.val_folder is None:
        opt.val_folder = ''

    iterations = opt.lr_decay_epochs.split(',')
    opt.lr_decay_epochs = list([])
    for it in iterations:
        opt.lr_decay_epochs.append(int(it))

    opt.model_name = '{}_{}_lr_{}_decay_{}_bsz_{}'.\
        format(opt.dataset, opt.model, opt.learning_rate, opt.weight_decay,
               opt.batch_size)
    opt.model_path = './save/Linear/{}_models'.format(opt.dataset)
    opt.tb_path = './save/Linear/{}_tensorboard'.format(opt.dataset)

    if opt.cosine:
        opt.model_name = '{}_cosine'.format(opt.model_name)

    # warm-up for large-batch training,
    if opt.warm:
        opt.model_name = '{}_warm'.format(opt.model_name)
        opt.warmup_from = 0
        opt.warm_epochs = 10
        if opt.cosine:
            eta_min = opt.learning_rate * (opt.lr_decay_rate ** 3)
            opt.warmup_to = eta_min + (opt.learning_rate - eta_min) * (
                    1 + math.cos(math.pi * opt.warm_epochs / opt.epochs)) / 2
        else:
            opt.warmup_to = opt.learning_rate

    opt.tb_folder = os.path.join(opt.tb_path, opt.model_name)
    if not os.path.isdir(opt.tb_folder):
        os.makedirs(opt.tb_folder)

    opt.save_folder = os.path.join(opt.model_path, opt.model_name + "_head")
    #if not os.path.isdir(opt.save_folder):
    #    os.makedirs(opt.save_folder)

    if opt.dataset == 'cifar10':
        opt.n_cls = 10
    elif opt.dataset == 'cifar100':
        opt.n_cls = 100
    elif opt.dataset == 'path':
        opt.n_cls = 87
    else:
        raise ValueError('dataset not supported: {}'.format(opt.dataset))

    return opt

# Assuming 'model' is your pre-trained model and 'classifier' is your new classifier layer
class ResNet50timm(torch.nn.Module):
    def __init__(self, model, classifier):
        super(ResNet50timm, self).__init__()
        self.encoder = model.encoder  # Use the encoder from the pre-trained model
        self.classifier = classifier  # Use your new classifier

    def forward(self, x):
        x = self.encoder(x)  # Pass input through the encoder
        x = x.view(x.size(0), -1)  # Flatten the output for the classifier
        x = self.classifier(x)  # Pass the output through the classifier
        return x

def set_model(opt):
    if torch.cuda.is_available():
        #  model = resnet18().cpu()
        custom_model = SupConResNet(name=opt.model)
        custom_model = custom_model.cuda()
        #custom_model.load_state_dict(torch.load(file_path)['model'])
        custom_model.load_state_dict(torch.load('pesos/resnet50.pth')['model'])

        classifier = LinearClassifier(name=opt.model, num_classes=87)
        classifier = classifier.cuda()
        #classifier.load_state_dict(torch.load(head_path)['model'])
        classifier.load_state_dict(torch.load('pesos/resnet50_head.pth')['model'])
        
        #print(model)
        #print("Press Enter to continue.")
        #input()
        criterion = torch.nn.CrossEntropyLoss().cuda()

        model = ResNet50timm(custom_model, classifier)
    else:
        raise NotImplementedError('This code requires GPU')

    return model, criterion

def myAccuracy(output, target, inferred_classes, gt_classes, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""

    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        # store the inferred and ground truth classes into a list iteratively
        inferred_classes = torch.cat((inferred_classes, pred[0].cpu().flatten().float()), dim=0)
        gt_classes = torch.cat((gt_classes, target.cpu().flatten().float()), dim=0)

        res = []
        for k in topk:
            correct_k = correct[:k].contiguous().view(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res, inferred_classes, gt_classes

def compute_metrics(ground_truth_classes_vector, inferred_classes_vector):
    """Computes the accuracy (micro), precision, recall and f1-metrics (the last three using macro)."""
    num_classes = 87

    # convertir a enteros
    ground_truth_classes_vector = ground_truth_classes_vector.long()
    inferred_classes_vector = inferred_classes_vector.long()

    # Generate confusion matrix
    confusion_matrix = multiclass_confusion_matrix(inferred_classes_vector, ground_truth_classes_vector, num_classes)
    #print(confusion_matrix)

    # Calculate the number of instances per class
    instances_per_class = confusion_matrix.sum(axis=1)
    # Print the number of instances per class
    #print("Number of instances per class:", instances_per_class)

    # Plot the number of instances per class
    classes = [f'{i+1}' if (i+1) % 3 == 0 else '' for i in range(num_classes)]
    classes[0] = '1'
    plt.bar(range(num_classes), instances_per_class.numpy())
    plt.xlabel('Classes')
    plt.ylabel('Number of Instances')
    plt.title('Number of Instances per Class')
    plt.xticks(range(num_classes), classes, rotation=0)  # Rotate class labels for better readability
    plt.grid(True)
    #plt.show(block=False)
    #plt.figure()  # Create a new figure

    # Calculate the number of inferred instances per class
    inferred_per_class = confusion_matrix.sum(axis=0)
    # Print the number of inferred instances per class
    #print("Number of inferred instances per class:", inferred_per_class)

    # Plot the number of inferred instances per class
    plt.bar(range(num_classes), inferred_per_class.numpy())
    plt.xlabel('Classes')
    plt.ylabel('Number of Inferred Instances')
    plt.title('Number of Inferred Instances per Class')
    plt.xticks(range(num_classes), classes, rotation=0)  # Use the same classes variable for x-axis labels
    plt.grid(True)
    #plt.show()

    #print("Inferred Classes Vector:", inferred_classes_vector_torch.tolist())
    #print("Ground Truth Classes Vector:", ground_truth_classes_vector_torch.tolist())

    print('-' * 50)

    # calcular métricas mediante torchmetrics
    # https://lightning.ai/docs/torchmetrics/stable/
    accuracies = Accuracy(task='multiclass', num_classes=num_classes, average='micro', top_k=1)
    precision = Precision(task='multiclass', num_classes=num_classes, average='macro')
    recall = Recall(task='multiclass', num_classes=num_classes, average='macro')
    f1 = F1Score(task='multiclass', num_classes=num_classes, average='macro')

    acc = accuracies(inferred_classes_vector, ground_truth_classes_vector)
    prec = precision(inferred_classes_vector, ground_truth_classes_vector)
    rec = recall(inferred_classes_vector, ground_truth_classes_vector)
    f1_score = f1(inferred_classes_vector, ground_truth_classes_vector)

    # imprime los resultados
    print(f'Accuracy: {acc*100:.2f}%')
    print(f'Precision: {prec*100:.2f}%')
    print(f'Recall: {rec*100:.2f}%')
    print(f'F1 Score: {f1_score*100:.2f}%')

    print('-' * 50)

    # calcular métricas mediante torcheval
    # https://pytorch.org/torcheval/stable/
    acc = multiclass_accuracy(inferred_classes_vector, ground_truth_classes_vector, average="micro", num_classes=num_classes, k=1)
    prec = multiclass_precision(inferred_classes_vector, ground_truth_classes_vector, average="macro", num_classes=num_classes)
    rec = multiclass_recall(inferred_classes_vector, ground_truth_classes_vector, average="macro", num_classes=num_classes)
    f1_score= multiclass_f1_score(inferred_classes_vector, ground_truth_classes_vector, average="macro", num_classes=num_classes)

    # imprime los resultados
    print(f'Accuracy: {acc*100:.2f}%')
    print(f'Precision: {prec*100:.2f}%')
    print(f'Recall: {rec*100:.2f}%')
    print(f'F1 Score: {f1_score*100:.2f}%')

    print('-' * 50)

def validate(val_loader, model, criterion, opt):
    """Validates the backbone and head of the model."""

    inferred_classes = torch.tensor([], dtype=torch.float32)
    gt_classes = torch.tensor([], dtype=torch.float32)

    model.eval()

    batch_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()

    end = time.time()
    for idx, (images, labels) in enumerate(val_loader):
        images = images.float().cuda()
        labels = labels.cuda()
        bsz = labels.shape[0]

        # forward
        output = model(images)
        loss = criterion(output, labels)

        # update metric
        losses.update(loss.item(), bsz)
        acc1, inferred_classes, gt_classes = myAccuracy(output, labels, inferred_classes=inferred_classes, gt_classes=gt_classes,  topk=(1,))
        top1.update(acc1[0], bsz)

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

    print('Test: [{0}/{1}]\t'
        'Time {batch_time:.3f} \t'
        'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
        'Acc@1 {top1.val[0]:.3f} ({top1.avg[0]:.3f})'.format(
            idx, len(val_loader), 
            batch_time=batch_time.val, loss=losses, top1=top1))

    print(' * Acc@1 {top1.avg[0]:.3f}'.format(top1=top1))

    compute_metrics(gt_classes, inferred_classes)

    return losses.avg, top1.avg

def main():
    best_acc = 0
    best_epoch = 0
    opt = parse_option()

    # build data loader
    _, val_loader = set_loader(opt)

    # build model
    model, criterion = set_model(opt)

    # eval dataset
    loss, val_acc = validate(val_loader, model, criterion, opt)

    epoch = 1

    if val_acc > best_acc:
        best_acc = val_acc[0]
        best_epoch = epoch
        print('Best accuracy at epoch {} is {:.2f}'.format(best_epoch, best_acc))
    else:
        raise Exception("error in accuracies")

if __name__ == '__main__':
    try:
        main()
    except Exception as e:
        print(f"An error occurred: {e}")
        # Add the code you want to execute when main() fails here
        message = f'Script execution failed!\n ' \
            f'Error: {e}'
