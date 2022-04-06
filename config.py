import os
import sys
import warnings
from sklearn.preprocessing import StandardScaler

class DefaultConfig(object):
    seed = 666
    device = 7
    
    batch_size = 28

    rdrop = False

    # dataset = 'vggDataset'
    # model = 'densenet121'
    dataset = 'dpnDataset'
    data_size = 126
    model = 'cust2'  # DPN92_3D
    scaler = StandardScaler()

    vgg16 = {
        'pretrained': False
    }

    vggDataset = {
        'raw_path': 'datasets/raw',
        'processed_path': 'datasets/processed/vgg_processed.npz'
    }

    # dpnDataset = {
    #     'prefix': '/home/lyj_11921026/liuqinxian/NAT/NAT/datasets/processed/32cube',
    #     'label': '/home/lyj_11921026/liuqinxian/NAT/NAT/datasets/raw/neo12.txt',
    #     'cube': 32
    # }

    Adam = {
        'lr': 1e-4,
        'weight_decay': 1e-1
    }

    SGD = {
        'lr': 1e-5,
        'momentum': 0.9
    }

    StepLR = {
        'step_size': 40,
        'gamma': 0.1
    }
    
    epochs = 1000

    save_path = 'record/cust2_rdrop0.25'  # 12345nbn_lr2_wd1
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    def parse(self, kwargs):
        '''
        customize configuration by input in terminal
        '''
        for k, v in kwargs.items():
            if not hasattr(self, k):
                warnings.warn('Warning: opt has no attribute %s' % k)
            setattr(self, k, v)

    def output(self):
        print('user config:')
        for k, v in self.__class__.__dict__.items():
            if not k.startswith('__'):
                print(k, getattr(self, k))


class Logger(object):
    def __init__(self, file_name='Default.log'):

        self.terminal = sys.stdout
        self.log = open(file_name, 'a')

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)

    def flush(self):
        pass