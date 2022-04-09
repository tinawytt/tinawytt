import os
from unittest import result
os.environ["CUDA_VISIBLE_DEVICES"] = "0,6"
import fnmatch
import random
from abc import ABCMeta, abstractmethod
import torch
from torch.utils.data import DataLoader, Dataset
from skimage.transform import resize
from trixi.util.pytorchutils import set_seed
import numpy as np
import pickle

import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau, StepLR
from collections import OrderedDict
# from networks.UNet3D import UNet3D
from trixi.util import Config
from trixi.experiment.pytorchexperiment import PytorchExperiment
from torch import nn
from trixi.util.config import update_from_sys_argv
from trixi.logger.experiment.pytorchexperimentlogger import PytorchExperimentLogger
from trixi.util import ResultLogDict, SourcePacker
import sys
import torch.backends.cudnn as cudnn

cudnn.benchmark = True

def get_config():
    # Set your own path, if needed.
    data_root_dir = '/home/wwh/wyt/master/BraTS2020_TrainingData/'  # The path where the downloaded dataset is stored.

    c = Config(
        update_from_argv=True,  # If set 'True', it allows to update each configuration by a cmd/terminal parameter.

        # Train parameters
        # num_classes=3,
        num_classes=4,
        in_channels=4,
        # batch_size=8,
        batch_size=4,
        patch_size=64,
        n_epochs=20,
        learning_rate=0.0002,
        fold=0,  # The 'splits.pkl' may contain multiple folds. Here we choose which one we want to use.

        device="cuda",  # 'cuda' is the default CUDA device, you can use also 'cpu'. For more information, see https://pytorch.org/docs/stable/notes/cuda.html

        # Logging parameters
        name='Basic_Unet',
        author='tinawytt',  # Author of this project
        plot_freq=10,  # How often should stuff be shown in visdom
        append_rnd_string=False,  # Appends a random string to the experiment name to make it unique.
        start_visdom=True,  # You can either start a visom server manually or have trixi start it for you.

        do_instancenorm=True,  # Defines whether or not the UNet does a instance normalization in the contracting path
        do_load_checkpoint=False,
        checkpoint_dir='',

        
        base_dir='/home/wwh/wyt/master/',  # Where to log the output of the experiment.

        data_root_dir=data_root_dir,  # The path where the downloaded dataset is stored.
        data_dir=data_root_dir,  # This is where your training and validation data is stored
        data_test_dir=data_root_dir,  # This is where your test data is stored

        split_dir=data_root_dir,  # This is where the 'splits.pkl' file is located, that holds your splits.

        # execute a segmentation process on a specific image using the model
        model_dir=os.path.join('/home/wwh/wyt/master/', ''),  # the model being used for segmentation
    )

    print(c)
    return c

def load_dataset(base_dir, pattern='*.npz', keys=None):
    fls = []
    files_len = []
    dataset = []
    i = 0
    for root, dirs, files in os.walk(base_dir):
        
        for filename in sorted(fnmatch.filter(files, pattern)):

            if keys is not None and filename[:-4] in keys:
                npz_file = os.path.join(root, filename)
                numpy_array = np.load(npz_file)['data'] # （5 x x x）
                
                fls.append(npz_file)
                files_len.append(numpy_array.shape[1])

                dataset.extend([i])
                

                i += 1

    return fls, files_len, dataset

class SlimDataLoaderBase(object):
    def __init__(self, data, batch_size, number_of_threads_in_multithreaded=None):
        __metaclass__ = ABCMeta
        self.number_of_threads_in_multithreaded = number_of_threads_in_multithreaded
        self._data = data
        self.batch_size = batch_size
        self.thread_id = 0

    def set_thread_id(self, thread_id):
        self.thread_id = thread_id

    def __iter__(self):
        return self

    def __next__(self):
        return self.generate_train_batch()

    @abstractmethod
    def generate_train_batch(self):
        '''override this
        Generate your batch from self._data .Make sure you generate the correct batch size (self.BATCH_SIZE)
        '''
        pass


class NumpyDataLoader(SlimDataLoaderBase):
    def __init__(self, base_dir, mode="train", batch_size=16, num_batches=10000000,
                 seed=None, file_pattern='*.npz', label=1, input=(0,), keys=None):

        shorter_keys=[]
        for key in keys:
            arr=key.split('/')
            
            shorter_keys.append(arr[len(arr)-1])
        
        keys=shorter_keys
        self.files, self.file_len, self.dataset = load_dataset(base_dir=base_dir, pattern=file_pattern, keys=keys )
        
        
        super(NumpyDataLoader, self).__init__(self.dataset, batch_size, num_batches)

        self.batch_size = batch_size

        self.use_next = True
        if mode == "train":
            self.use_next = False
        # else:
        #     print("mode=validate")

        self.idxs = list(range(0, len(self.dataset)))
        
        self.data_len = len(self.dataset)

        self.num_batches = min((self.data_len // self.batch_size)+10, num_batches)

        if isinstance(label, int):
            label = (label,)
        self.input = input
        
        self.label = label
        
        self.np_data = np.asarray(self.dataset)
        

    def reshuffle(self):
        print("Reshuffle...")
        random.shuffle(self.idxs)
        print("Initializing... this might take a while...")

    def generate_train_batch(self):
        open_arr = random.sample(self._data, self.batch_size)
        
        return self.get_data_from_array(open_arr)

    def __len__(self):
        n_items = min(self.data_len // self.batch_size, self.num_batches)
        return n_items

    def __getitem__(self, item):
        idxs = self.idxs
        data_len = len(self.dataset)
        np_data = self.np_data

        if item > len(self):
            raise StopIteration()
        if (item * self.batch_size) == data_len:
            raise StopIteration()

        start_idx = (item * self.batch_size) % data_len
        stop_idx = ((item + 1) * self.batch_size) % data_len

        if ((item + 1) * self.batch_size) == data_len:
            stop_idx = data_len

        if stop_idx > start_idx:
            idxs = idxs[start_idx:stop_idx]
        else:
            raise StopIteration()

        open_arr = np_data[idxs]
        return self.get_data_from_array(open_arr)

    def get_data_from_array(self, open_array):
        data = []
        fnames = []
        idxs = []
        labels = []
        
        for idx in open_array:
            fn_name = self.files[idx]
            
            numpy_array = np.load(fn_name)['data']
            
            data.append(numpy_array[0:4])   # 'None' keeps the dimension list()list(self.input)

            if self.label is not None:
                labels.append(numpy_array[4])   # 'None' keeps the dimension list()

            fnames.append(self.files[idx])
            idxs.append(idx)
        

        ret_dict = {'data': data, 'fnames': fnames, 'idxs': idxs}
        if self.label is not None:
            ret_dict['seg'] = labels

        return ret_dict

class WrappedDataset(Dataset):
    def __init__(self, dataset, transform=None):
        self.transform = transform
        self.dataset = dataset

        self.is_indexable = False
        if hasattr(self.dataset, "__getitem__") and not (hasattr(self.dataset, "use_next") and self.dataset.use_next is True):
            print("use next=",self.dataset.use_next)
            self.is_indexable = True#train

    def __getitem__(self, index):

        if not self.is_indexable:
            item = next(self.dataset)
        else:
            # print("indexable=",self.is_indexable)
            item = self.dataset[index]
        # item = self.transform(**item)
        
        old_data=item['data']
        old_seg=item['seg']
        
        new_shape=(128,128,128)
        result_list=[]
        # print(type(old_data)) 
        # print(len(old_data))
        # print(np.unique(old_seg[0])) # (-1 0 1 2 4)
        print("==resizing data==")
        for i in range(len(old_data)):
            print('here',i)
            result_array=np.zeros((4,128,128,128),dtype=old_data[i].dtype)
            for m in range(0,4):
                
                result_element = np.zeros(new_shape, dtype=old_data[i].dtype)
                # print(old_data[i][m].shape)
                result_element= resize(old_data[i][m].astype(float), new_shape, order=3, clip=True, anti_aliasing=False)
                # print("after",result_element.shape)
                result_array[m]=result_element
            result_list.append(result_array)
            print("after1case in batch")
            
            # print('here')
        item['data']=result_list
        result_list=[]
        result_element = np.zeros(new_shape, dtype=old_seg[0].dtype)
        unique_labels = np.unique(old_seg[0])
        print("==resizing segmentations==")
        for m in range(len(old_seg)):
            for i, c in enumerate(unique_labels):
                
                mask = old_seg[m] == c
                # print(mask.shape)
                reshaped_multihot = resize(mask.astype(float), new_shape, order=1, mode="edge", clip=True, anti_aliasing=False)
                # print("after",reshaped_multihot.shape)
                # print(np.unique(reshaped_multihot))
                result_element[reshaped_multihot >= 0.5] = c
            
            result_list.append(result_element)
        item['seg']=result_list
        # print(np.unique(result_list[0]))
        return item

    def __len__(self):
        return int(self.dataset.num_batches)


class MultiThreadedDataLoader(object):
    def __init__(self, data_loader,  num_processes, transform=None, **kwargs):

        self.cntr = 1
        self.ds_wrapper = WrappedDataset(data_loader, transform)

        self.generator = DataLoader(self.ds_wrapper, batch_size=1, shuffle=False, sampler=None, batch_sampler=None,
                                    num_workers=num_processes, pin_memory=True, drop_last=False,
                                    worker_init_fn=self.get_worker_init_fn())

        self.num_processes = num_processes
        self.iter = None

    def get_worker_init_fn(self):
        def init_fn(worker_id):
            set_seed(worker_id + self.cntr)

        return init_fn

    def __iter__(self):
        self.kill_iterator()
        self.iter = iter(self.generator)
        return self.iter

    def __next__(self):
        if self.iter is None:
            self.iter = iter(self.generator)
        return next(self.iter)

    def renew(self):
        self.cntr += 1
        self.kill_iterator()
        self.generator.worker_init_fn = self.get_worker_init_fn()
        self.iter = iter(self.generator)

    def kill_iterator(self):
        try:
            if self.iter is not None:
                self.iter._shutdown_workers()
                for p in self.iter.workers:
                    p.terminate()
        except:
            print("Could not kill Dataloader Iterator")

class NumpyDataSet(object):
    """
    TODO
    """
    def __init__(self, base_dir, mode="train", batch_size=16, num_batches=10000000, seed=None, num_processes=2, num_cached_per_queue=2 * 4, target_size=128,
                 file_pattern='*.npz', label=1, input=(0,), do_reshuffle=True, keys=None):#8*4->2*4  8->2

        data_loader = NumpyDataLoader(base_dir=base_dir, mode=mode, batch_size=batch_size, num_batches=num_batches, seed=seed, file_pattern=file_pattern,
                                      input=input, label=label, keys=keys) # 取列表里所有的数据

        self.data_loader = data_loader
        self.batch_size = batch_size
        self.do_reshuffle = do_reshuffle
        self.number_of_slices = 1

        self.transforms = None
        self.augmenter = MultiThreadedDataLoader(data_loader, num_processes,
        num_cached_per_queue=num_cached_per_queue, seeds=seed, shuffle=do_reshuffle)
        

    def __len__(self):
        return len(self.data_loader)

    def __iter__(self):
        if self.do_reshuffle:
            self.data_loader.reshuffle()
        self.augmenter.renew()
        return self.augmenter

    def __next__(self):
        return next(self.augmenter)

class CrossentropyND(torch.nn.CrossEntropyLoss):
    """
    Network has to have NO NONLINEARITY!
    """
    def forward(self, inp, target):
        target = target.long()
        num_classes = inp.size()[1]

        i0 = 1
        i1 = 2

        while i1 < len(inp.shape): # this is ugly but torch only allows to transpose two axes at once
            inp = inp.transpose(i0, i1)
            i0 += 1
            i1 += 1

        inp = inp.contiguous()
        inp = inp.view(-1, num_classes)

        target = target.view(-1,)

        return super(CrossentropyND, self).forward(inp, target)

def softmax_helper(x):
    rpt = [1 for _ in range(len(x.size()))]
    rpt[1] = x.size(1)
    x_max = x.max(1, keepdim=True)[0].repeat(*rpt)
    e_x = torch.exp(x - x_max)
    return e_x / e_x.sum(1, keepdim=True).repeat(*rpt)   

class DC_and_CE_loss(nn.Module):
    def __init__(self, soft_dice_kwargs, ce_kwargs, aggregate="sum"):
        super(DC_and_CE_loss, self).__init__()
        self.aggregate = aggregate
        self.ce = CrossentropyND(**ce_kwargs)
        self.dc = SoftDiceLoss(apply_nonlin=softmax_helper, **soft_dice_kwargs)

    def forward(self, net_output, target):
        dc_loss = self.dc(net_output, target)
        ce_loss = self.ce(net_output, target)
        if self.aggregate == "sum":
            result = ce_loss + dc_loss
        else:
            raise NotImplementedError("nah son") # reserved for other stuff (later)
        print(result)
        return result

class SoftDiceLoss(nn.Module):
    def __init__(self, smooth=1., apply_nonlin=None, batch_dice=False, do_bg=True, smooth_in_nom=True, background_weight=1, rebalance_weights=None):
        """
        hahaa no documentation for you today
        :param smooth:
        :param apply_nonlin:
        :param batch_dice:
        :param do_bg:
        :param smooth_in_nom:
        :param background_weight:
        :param rebalance_weights:
        """
        super(SoftDiceLoss, self).__init__()
        if not do_bg:
            assert background_weight == 1, "if there is no bg, then set background weight to 1 you dummy"
        self.rebalance_weights = rebalance_weights
        self.background_weight = background_weight
        if smooth_in_nom:
            self.smooth_in_nom = smooth
        else:
            self.smooth_in_nom = 0
        self.do_bg = do_bg
        self.batch_dice = batch_dice
        self.apply_nonlin = apply_nonlin
        self.smooth = smooth
        self.y_onehot = None

    def forward(self, x, y):
        with torch.no_grad():
            y = y.long()
        shp_x = x.shape
        shp_y = y.shape
        # print("1",x.shape,y.shape) # pred, target
        # torch.Size([1, 3, 128, 128, 128]) torch.Size([128, 128, 128])
        if self.apply_nonlin is not None:
            x = self.apply_nonlin(x)
        if len(shp_x) != len(shp_y):
            y = y.view((shp_y[0], 1, *shp_y[1:]))
        # now x and y should have shape (B, C, X, Y(, Z))) and (B, 1, X, Y(, Z))), respectively   
        # torch.Size([1, 3, 128, 128, 128]) torch.Size([128, 1, 128, 128])
        y_onehot = torch.zeros(shp_x)
        if x.device.type == "cuda":
            y_onehot = y_onehot.cuda(x.device.index)
        # print("2",y_onehot.shape,y.shape) 
        # y[y == -1] = 0 # -1改成0
        # y[y == 4] = 3 # 4改成3
        # print(torch.unique(y)) # 0,1,2,3
        '''
        下面一行报错
        错误原因:输出通道是3个,分别代表3个类别索引为0、1、2,标签中是0、1、2、4这4个值,所以scatter_会往不存在的第4个通道赋值。
        解决方案:将标签4改成3,-1改成0,num_classes设成4表示4中类别(包括背景),scatter_可以正常进行one_hot编码,self.do_bg设为false(默认false)表示不考虑背景损失,会在后面计算损失前去除背景通道
        github仓库里也有这个错误的说明:make sure you updated num_classes in your config file. The value of num_classes should always equal the number of your labels including background.
        '''
        y_onehot.scatter_(1, y, 1) # 这里输出“/pytorch/aten/src/ATen/native/cuda/ScatterGatherKernel.cu:276: operator(): block: [10711,0,0], thread: [31,0,0] Assertion `idx_dim >= 0 && idx_dim < index_size && "index out of bounds"` failed.”
        print(y_onehot.shape)
        # 打印batch中第一个样例
        # f = open('../../onehot.txt', 'w')
        # y_onehot_np = y_onehot.long().cpu().numpy()
        # for c in range(y_onehot_np.shape[2]):
        #     for d in range(y_onehot_np.shape[3]):
        #         for e in range(y_onehot.shape[4]):
        #             # print('---', end="")
        #             print("\r\n", file=f)
        #             for b in range(y_onehot_np.shape[1]):
        #                 # print(y_onehot_np[0][b][c][d][e], end=" ")
        #                 print(y_onehot_np[0][b][c][d][e], end=" ",file=f)
            
                            
        if not self.do_bg:
            '''
            在这里去掉背景通道,不考虑背景的损失（又看了一下，这一步后计算损失时也不是完全不考虑，就是尽量让背景部分与其它类别远离）
            '''
            x = x[:, 1:]
            # print('=======')
            # print(np.unique(y_onehot[0].numpy()))
            # print('=======')
            y_onehot = y_onehot[:, 1:]
        if not self.batch_dice:
            if self.background_weight != 1 or (self.rebalance_weights is not None):
                raise NotImplementedError("nah son")
            l = soft_dice(x, y_onehot, self.smooth, self.smooth_in_nom)
        else:
            l = soft_dice_per_batch_2(x, y_onehot, self.smooth, self.smooth_in_nom,
                                      background_weight=self.background_weight,
                                      rebalance_weights=self.rebalance_weights)
        return l


def soft_dice_per_batch(net_output, gt, smooth=1., smooth_in_nom=1., background_weight=1):
    axes = tuple([0] + list(range(2, len(net_output.size()))))
    intersect = sum_tensor(net_output * gt, axes, keepdim=False)
    denom = sum_tensor(net_output + gt, axes, keepdim=False)
    weights = torch.ones(intersect.shape)
    weights[0] = background_weight
    if net_output.device.type == "cuda":
        weights = weights.cuda(net_output.device.index)
    result = (- ((2 * intersect + smooth_in_nom) / (denom + smooth)) * weights).mean()
    return result


def soft_dice_per_batch_2(net_output, gt, smooth=1., smooth_in_nom=1., background_weight=1, rebalance_weights=None):
    if rebalance_weights is not None and len(rebalance_weights) != gt.shape[1]:
        rebalance_weights = rebalance_weights[1:] # this is the case when use_bg=False
    axes = tuple([0] + list(range(2, len(net_output.size()))))
    tp = sum_tensor(net_output * gt, axes, keepdim=False)
    fn = sum_tensor((1 - net_output) * gt, axes, keepdim=False)
    fp = sum_tensor(net_output * (1 - gt), axes, keepdim=False)
    weights = torch.ones(tp.shape)
    weights[0] = background_weight
    if net_output.device.type == "cuda":
        weights = weights.cuda(net_output.device.index)
        
    if rebalance_weights is not None:
        rebalance_weights = torch.from_numpy(rebalance_weights).float()
        if net_output.device.type == "cuda":
            rebalance_weights = rebalance_weights.cuda(net_output.device.index)
        tp = tp * rebalance_weights
        fn = fn * rebalance_weights
    result = (- ((2 * tp + smooth_in_nom) / (2 * tp + fp + fn + smooth)) * weights).mean()
    return result

def sum_tensor(inp, axes, keepdim=False):
    axes = np.unique(axes).astype(int)
    if keepdim:
        for ax in axes:
            inp = inp.sum(int(ax), keepdim=True)
    else:
        for ax in sorted(axes, reverse=True):
            inp = inp.sum(int(ax))
    return inp

def soft_dice(net_output, gt, smooth=1., smooth_in_nom=1.):
    axes = tuple(range(2, len(net_output.size())))
    intersect = sum_tensor(net_output * gt, axes, keepdim=False)
    denom = sum_tensor(net_output + gt, axes, keepdim=False)
    result = (- ((2 * intersect + smooth_in_nom) / (denom + smooth))).mean()
    return result



class UNetExperiment3D(PytorchExperiment):    
    def setup(self):
        from UNet3D import UNet3D
        data_dir='/home/wwh/wyt/master/BraTS2020_TrainingData/'
        with open(os.path.join(data_dir, "splits.pkl"), 'rb') as f:
          splits = pickle.load(f)
        tr_keys = splits[0]['train']
        val_keys = splits[0]['val']
        test_keys = splits[0]['test']
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.train_data_loader = NumpyDataSet(base_dir=data_dir, target_size=64, batch_size=self.config.batch_size,keys=tr_keys)
        self.val_data_loader = NumpyDataSet(base_dir=data_dir, target_size=64, batch_size=self.config.batch_size,
                                            keys=val_keys, mode="val", do_reshuffle=False)
        self.model = UNet3D(num_classes=self.config.num_classes, in_channels=self.config.in_channels)
        self.model = torch.nn.DataParallel(self.model) # 多卡并行
        self.model.to(self.device)
        self.loss = DC_and_CE_loss({'batch_dice': True, 'smooth': 1e-5, 'smooth_in_nom': True,
                                    'do_bg': False, 'rebalance_weights': None, 'background_weight': 1}, OrderedDict())
        #print("loss ok")
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.config.learning_rate)
        # self.scheduler = ReduceLROnPlateau(self.optimizer, 'min')
        self.scheduler=StepLR(self.optimizer, step_size=5, gamma=0.1)

        # If directory for checkpoint is provided, we load it.
        if self.config.do_load_checkpoint:
            if self.config.checkpoint_dir == '':
                print('checkpoint_dir is empty, please provide directory to load checkpoint.')
            else:
                self.load_checkpoint(name=self.config.checkpoint_dir, save_types=("model",))

        self.save_checkpoint(name="checkpoint_start")
        
        self.elog.print('Experiment set up.')
        print("set up ok")
        
    def train(self, epoch):
        self.elog.print('=====TRAIN=====')
        self.model.train()

        batch_counter = 0
        count=0
        
        for data_batch in self.train_data_loader:
            '''
            问题:batch_size等于8进不来,与dataloader的StopIteration有关,测试发现,batch_size设为n,遍历的时候会迭代n次,每次取n个,8 * 8 > 48,会一次性取8组8个样例
                应该batch_size=n时,取一次,这一次取n个样例。
            解决方案一:将MultiThreadedDataLoader中Dataloader中batch_size调为1,外层封装的类batch_size设为8,然后data和target的获取方式修改,已实现
            解决方案二:重写一个简单的DataLoader
            '''
            # print('count: ', count)#进不去
            self.optimizer.zero_grad()

            # Shape of data_batch = [1, b, c, w, h]
            # Desired shape = [b, c, w, h]
            # Move data and target to the GPU
            # data = data_batch['data'][0].float().to(self.device)
            # target = data_batch['seg'][0].long().to(self.device)
            #-------------
            '''
            获取方式修改,将batch_size个值拼接
            validata处也要修改
            '''
            data = data_batch['data'][0]
            target = data_batch['seg'][0]
            # print('********')
            # print(data_batch['data'][0].shape)
            # print('********')
            for i in range(1, len(data_batch['data'])):
                data = torch.cat((data, data_batch['data'][i]), dim=0)
                target = torch.cat((target, data_batch['seg'][i]), dim=0)
            data = data.float().to(self.device)
            target = target.long().to(self.device)
            #-------------

            pred = self.model(data)
            target[target == -1] = 0 # -1改成0
            target[target == 4] = 3 # 4改成3
            loss = self.loss(pred, target.squeeze())
            # loss = self.loss(pred, target)
            print("middle")
            # loss = self.ce_loss(pred, target.squeeze())
            loss.backward()
            print("loss calculated")
            self.optimizer.step()
            
            # Some logging and plotting
            if (batch_counter % self.config.plot_freq) == 0:
                print("enter")
                print(type(loss.item()))

                # self.elog.print(str(self._epoch_idx)+':'+str(loss))
                self.elog.print('Epoch: %d Loss: %.4f' % (self._epoch_idx, loss.item()))#modify

                self.add_result(value=loss.item(), name='Train_Loss', tag='Loss', counter=epoch + (batch_counter / self.train_data_loader.data_loader.num_batches))
            # self.clog.show_image_grid(data[:,:,30].float(), name="data", normalize=True, scale_each=True, n_iter=epoch)
            # self.clog.show_image_grid(target[:,:,30].float(), name="mask", title="Mask", n_iter=epoch)
            # self.clog.show_image_grid(torch.argmax(pred.cpu(), dim=1, keepdim=True)[:,:,30], name="unt_argmax", title="Unet", n_iter=epoch)

            batch_counter += 1
            count=count+1
        if epoch>=10:
            self.save_checkpoint(name="checkpoint_epoch_"+str(epoch))

    def validate(self, epoch):
        
        print("----validate------")
        # if epoch % 5 != 0:
        #     return
        self.elog.print('VALIDATE')
        self.model.eval()

        data = None
        loss_list = []
        
        with torch.no_grad():
            
            count=0
            for data_batch in self.val_data_loader:
                # data = data_batch['data'][0].float().to(self.device)
                # target = data_batch['seg'][0].long().to(self.device)
                #-------------
                '''
                获取方式修改,将batch_size个值拼接
                '''
                data = data_batch['data'][0]
                target = data_batch['seg'][0]
                for i in range(1, len(data_batch['data'])):
                    data = torch.cat((data, data_batch['data'][i]), dim=0)
                    target = torch.cat((target, data_batch['seg'][i]), dim=0)
                data = data.float().to(self.device)
                target = target.long().to(self.device)
                #-------------
                print('count', count)
                pred = self.model(data)
                target[target == -1] = 0 # -1改成0
                target[target == 4] = 3 # 4改成3
                # from torch.autograd import Variable
                # # 将这些数据转换成Variable类型
                # inputs, labels,targets = Variable(data), Variable(pred), Variable(target)
                # # 接下来就是跑模型的环节了，我们这里使用print来代替
                # print("epoch：", epoch, "的第" , data_batch['idxs'][0], "个inputs", inputs.data.size())
                # print("labels", np.unique(labels.data.cpu()))
                # print("seg",np.unique(targets.data.cpu()))
                loss = self.loss(pred, target.squeeze())
                loss_list.append(loss.item())
                count=count+1

        assert data is not None, 'data is None. Please check if your dataloader works properly'
        self.scheduler.step(np.mean(loss_list))
        self.elog.print('Epoch: %d Loss: %.4f' % (self._epoch_idx, float(np.mean(loss_list))))

        self.add_result(value=np.mean(loss_list), name='Val_Loss', tag='Loss', counter=epoch+1)

        # self.clog.show_image_grid(data[:,:,30].float(), name="data_val", normalize=True, scale_each=True, n_iter=epoch)
        # self.clog.show_image_grid(target[:,:,30].float(), name="mask_val", title="Mask", n_iter=epoch)
        # self.clog.show_image_grid(torch.argmax(pred.data.cpu()[:,:,30], dim=1, keepdim=True), name="unt_argmax_val", title="Unet", n_iter=epoch)

    def test(self):
        pass

if __name__ == "__main__":
    sys.path.append("/home/wwh/wyt/master/networks/")
    
    c = get_config()
    #print(globals().get("__file__"))
    exp = UNetExperiment3D(config=c, name=c.name, n_epochs=c.n_epochs,
                             seed=42, append_rnd_to_name=c.append_rnd_string, globs=globals()#,
                             # visdomlogger_kwargs={"auto_start": c.start_visdom},
                             #loggers={
                            #     "visdom": ("visdom", {"auto_start": c.start_visdom})
                             #}
                             )

    exp.run()
    print("ok")