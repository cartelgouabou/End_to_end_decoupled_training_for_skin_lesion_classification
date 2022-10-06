import argparse
# '''
parser = argparse.ArgumentParser(description='PyTorch ISIC2019 CNN Training')
parser.add_argument('--use_cuda', type=bool, default=True, help='device to train on')
parser.add_argument('--distribution', type=bool, default=True, help='Visualize data distribution')
parser.add_argument('--num_epochs', type=int, default=100, help='Number of epochs to train on') #100
parser.add_argument('--param', default='test', type=str, help='test')
parser.add_argument('--ratio_finetune', default=32, type=float, help='ratio of last layer to finetune') #32
parser.add_argument('--weighting_type', default='None', type=str, help='data sampling strategy for train loader')
parser.add_argument('--start_hm_epoch', default=50, type=int, help='start epoch for HM Loss')
parser.add_argument('--max_thresh', default=0.1, type=float,   help='max thresh for outliers')
parser.add_argument('--hm_delay_type', default='epoch', type=str, help='data sampling strategy for train loader')
parser.add_argument('--loss_type', default="CE", type=str, help='loss type')
parser.add_argument('--max_lr', '--learning-rate', default=0.001, type=float, help='define the maximum learning rate during cyclical learning. For more details check the backup implementation in  this file',dest='max_lr')
parser.add_argument('-b','--batch_size', default=128, type=int, help='define batch size') #128
parser.add_argument('--delay', default=15, type=int,help='number of epoch to patience before early stopping the training if no improvement of balanced accuracy') #128
parser.add_argument('--delay_hm', default=10, type=int,help='number of epoch to patience before starting to train with hmLoss')
parser.add_argument('--num_runs', default=10, type=int, help='number of runs to launch ')
parser.add_argument('--dataset_dir', default='/dataset/', type=str, help='ISIC dataset path ')
parser.add_argument('--model_path', default='/checkpoint/', type=str, help='checkpoint path ')
parser.add_argument('--history_path', default='/history/', type=str, help='history path ')
parser.add_argument('--result_dir', default='/results/', type=str, help='results path ')
parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                    help='momentum')
parser.add_argument('--wd', '--weight-decay', default=2e-4, type=float,
                    metavar='W', help='weight decay (default: 1e-4)',
                    dest='weight_decay')
parser.add_argument('-j', '--workers', default=4, type=int, metavar='N',
                    help='number of data loading workers (default: 4)')
parser.add_argument('--delta', default=10000000, type=int, help='delta parameter for HMLoss')




