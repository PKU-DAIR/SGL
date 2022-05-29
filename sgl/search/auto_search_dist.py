import os
import time
import torch
import torch.nn as nn
from torch.optim import Adam
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.utils.data import Dataset, DataLoader

from sgl.search.base_search import BaseSearch
from sgl.search.utils import accuracy, set_seed

class TrainDataset(Dataset):
    def __init__(self, data, label):
        self.data = data
        self.label = label

    def __len__(self):
        return len(self.data)

    def __getitem__(self, ind):
        x = self.data[ind]
        y = self.label[ind]
        return x, y

class SearchManagerDist(BaseSearch):
    def __init__(self, dataset, model, seed=42):
        super(SearchManagerDist, self).__init__()

        self.__dataset = dataset
        self.__model = model
        self.__seed = seed
    
    def _execute(self, args):
        t_pre_start = time.time()
        self.__model.preprocess(self.__dataset.adj, self.__dataset.x)
        t_pre_end = time.time()
        time_preprocess = t_pre_end - t_pre_start

        args.world_size = args.gpus * args.nodes
        os.environ['MASTER_ADDR'] = '127.0.0.1'
        os.environ['MASTER_PORT'] = '1788'
        
        t_total = time.time()
        best_val = 0.
        best_test = 0.
        for i in range(10):
            mp.spawn(self._train, nprocs=args.gpus, args=(args, self.__dataset, self.__model))
            acc_val, acc_test = self._evaluate()
            print('Turn: {:03d}'.format(i + 1),
                    'acc_val: {:.4f}'.format(acc_val),
                    'acc_test: {:.4f}'.format(acc_test))
            if acc_val > best_val:
                best_val = acc_val
                best_test = acc_test
                torch.save(self.__model, './best.pt')

        acc_val, acc_test, time_forward = self._postprocess()
        print(f'Post val: {acc_val:.4f}, Post test: {acc_test:.4f}')
        if acc_val > best_val:
            best_val = acc_val
            best_test = acc_test
        total_time = time_preprocess + time_forward

        print("Optimization Finished!")
        print("Total time elapsed: {:.4f}s".format(time.time() - t_total))
        print(f'Best val: {best_val:.4f}, best test: {best_test:.4f}')
        return best_test, total_time

    def _train(self, gpu, args, dataset, model):
        rank = args.nr * args.gpus + gpu
        dist.init_process_group(backend='nccl', init_method='env://', world_size=args.world_size, rank=rank)
        set_seed(self.__seed)

        torch.cuda.set_device(gpu)
        model.cuda(gpu)
        criterion = nn.CrossEntropyLoss().cuda(gpu)
        optimizer = Adam(model.parameters(), lr=args.lr, weight_decay=args.wd)

        _processed_feat_list = model._processed_feat_list
        model = nn.parallel.DistributedDataParallel(model, device_ids=[gpu])

        train_dataset = TrainDataset(self.__dataset.train_idx, self.__dataset.y[self.__dataset.train_idx])
        train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset,
                                                                        num_replicas=args.world_size,
                                                                        rank=rank)
        train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                                batch_size=args.batch,
                                                shuffle=False,
                                                num_workers=0,
                                                pin_memory=True,
                                                sampler=train_sampler)
        
        for epoch in range(args.epochs):
            for i, (idx, labels) in enumerate(train_loader):
                transferred_feat_list = [feat[idx].cuda(
                    non_blocking=True) for feat in _processed_feat_list]
                labels = labels.cuda(
                    non_blocking=True)
                # Forward pass
                outputs = model(transferred_feat_list)
                loss = criterion(outputs, labels)

                # Backward and optimize
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

    def _postprocess(self):
        device = torch.device(f"cuda:{0}" if torch.cuda.is_available() else "cpu")
        model = torch.load('./best.pt')
        model = model.to(device)
        model.eval()

        t_forward_start = time.time()
        output = model.model_forward(range(self.__dataset.num_node), device)
        final_output = model.postprocess(self.__dataset.adj, output)
        t_forward_end = time.time()
        time_forward = t_forward_end - t_forward_start

        acc_val = accuracy(final_output[self.__dataset.val_idx], self.__dataset.y[self.__dataset.val_idx])
        acc_test = accuracy(final_output[self.__dataset.test_idx], self.__dataset.y[self.__dataset.test_idx])
        return acc_val, acc_test, time_forward

    def _evaluate(self):
        device = torch.device(f"cuda:{0}" if torch.cuda.is_available() else "cpu")
        self.__model = self.__model.to(device)
        self.__model.eval()

        val_output = self.__model.model_forward(self.__dataset.val_idx, device)
        test_output = self.__model.model_forward(self.__dataset.test_idx, device)

        acc_val = accuracy(val_output, self.__dataset.y[self.__dataset.val_idx])
        acc_test = accuracy(test_output, self.__dataset.y[self.__dataset.test_idx])
        return acc_val, acc_test
