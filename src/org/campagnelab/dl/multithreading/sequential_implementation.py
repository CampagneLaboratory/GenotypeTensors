from queue import Queue
from threading import Thread

import time
from torch.autograd import Variable


class DataProvider:
    def __init__(self, iterator, batch_names, is_cuda=False, volatile={}, requires_grad={}):
        self.iterator = iterator
        self.batch_names = batch_names
        self.batch_index = 0
        self.is_cuda = is_cuda
        self.volatile=volatile
        self.requires_grad=requires_grad

    def __iter__(self):
        return self

    def __next__(self):
        """
        This method returns the next batch of data, prepared for pytorch, on GPU when is_cuda is true, as variables.
        Use variable_kargs to pass arguments to the Variable calls.
        :return: Dictionary with named inputs and outputs.
        """
        dict = {}
        try:
            batch =next(self.iterator)
        except StopIteration:
            raise StopIteration
        for loader_index, batch in enumerate(batch):
            loader_name=self.batch_names[loader_index]
            dict[loader_name] = {}
            if len(self.requires_grad)==0:
                self.requires_grad[loader_name]=[]
            if len(self.volatile)==0:
                self.volatile[loader_name]=[]

            for var_name in batch.keys():

                var_batch = Variable(batch[var_name], requires_grad=(var_name in self.requires_grad[loader_name]),
                             volatile=(var_name in self.volatile[loader_name]))
                if self.is_cuda:
                    var_batch = var_batch.cuda(async=True)
                dict[loader_name][var_name]=var_batch


        return dict



class CpuGpuDataProvider(DataProvider):
    def __init__(self, iterator, batch_names, is_cuda=False, volatile={}, requires_grad={}, preload_n=3, preload_cuda_n=3):
        super(CpuGpuDataProvider, self).__init__(iterator=iterator,batch_names=batch_names, is_cuda=False, # do not put on GPU in super
                                                 volatile=volatile, requires_grad=requires_grad)
        self.is_cuda = is_cuda
        self.cpu_batches_queue = Queue(maxsize=preload_n)
        self.batch_index = 0
        if is_cuda:
            self.gpu_batches_queue = Queue(maxsize=preload_cuda_n)

    def __next__(self):
        """
        This method returns the next batch of data, prepared for pytorch, on GPU when is_cuda is true.
        :return: Dictionary with named inputs and outputs.
        """
        while not self.gpu_batches_queue.full():
            self.populate_cpu_queue({})
            self.populate_gpu_queue()

        self.batch_index += 1
        if self.is_cuda:
            return self.gpu_batches_queue.get(block=True)
        else:
            return self.cpu_batches_queue.get(block=True)

    def populate_cpu_queue(self, recode_functions):

            try:
                next_item=super().__next__()
                for batch_name in self.batch_names:
                    batch = next_item[batch_name]
                    for var_name in recode_functions.keys():
                        if var_name in batch.keys():
                            batch[var_name] = recode_functions[var_name](batch[var_name])

                self.cpu_batches_queue.put(next_item, block=True)
            except StopIteration:
                raise StopIteration

    def populate_gpu_queue(self):

            batches = self.cpu_batches_queue.get(block=True)
            for batch_name in self.batch_names:
                batch=batches[batch_name]
                for var_name in batch.keys():
                    batch[var_name]=batch[var_name].cuda(async =True)
            self.gpu_batches_queue.put(batches, block=True)


class MultiThreadedCpuGpuDataProvider(CpuGpuDataProvider):
    def __init__(self, iterator, batch_names, is_cuda=False, volatile={}, requires_grad={}, preload_n=20,
                 preload_cuda_n=20, recode_functions={}):
        super().__init__(iterator, batch_names, is_cuda=is_cuda,
                         volatile=volatile, requires_grad=requires_grad, preload_n=preload_n,
                         preload_cuda_n=preload_cuda_n)
        self.stop_iteration = False
        self.kill_threads=False
        def add_to_cpu_queue():
            while not self.kill_threads:
                try:
                    if not self.cpu_batches_queue.full():
                        self.populate_cpu_queue(recode_functions=recode_functions)
                    else:
                        time.sleep(10 / 1000.0)
                except StopIteration:
                    self.stop_iteration = True
                    break

        self.t1 = Thread(target=add_to_cpu_queue, name="BatchesToCPU")
        self.t1.start()

        while self.cpu_batches_queue.empty():
            time.sleep(10 / 1000.0)

        if is_cuda:
            def add_to_gpu_queue():
                while not self.kill_threads:

                    if not self.gpu_batches_queue.full():
                        self.populate_gpu_queue()
                    else:
                        time.sleep(10 / 1000.0)

            self.t2 = Thread(target=add_to_gpu_queue, name="BatchesToGPU")
            self.t2.start()

            time.sleep(10 / 1000.0)
    def __enter__(self):
        pass

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()

    def __next__(self):
        """
        This method returns the next batch of data, prepared for pytorch, on GPU when is_cuda is true.
        :return: Dictionary with named inputs and outputs.
        """
        #print("cpu queue size: {}".format(self.cpu_batches_queue.qsize()))
        #print("gpu queue size: {}".format(self.gpu_batches_queue.qsize()))
        while self.queues_are_empty():
            if self.stop_iteration and self.queues_are_empty():
                raise StopIteration
            time.sleep(10/1000)

        if self.is_cuda:

            return self.gpu_batches_queue.get(block=True,timeout=3)
        else:

            return self.cpu_batches_queue.get(block=True, timeout=3)

    def close(self):

        self.kill_threads=True
        time.sleep(1)

    def queues_are_empty(self):
        return self.cpu_batches_queue.empty() or self.is_cuda and self.gpu_batches_queue.empty() and self.cpu_batches_queue.empty()


