from queue import Queue, Empty
from threading import Thread

import time

import torch
from torch import Tensor
from torch.autograd import Variable


class DataProvider:
    def __init__(self, iterator, batch_names, device=None, requires_grad=None, recode_functions=None):
        self.iterator = iterator
        self.batch_names = batch_names
        self.batch_index = 0
        self.requires_grad = {} if requires_grad is None else requires_grad
        self.recode_functions = {} if recode_functions is None else recode_functions
        self.all_columns_to_keep=[]
        for batch_name in batch_names:
            if requires_grad is not None:
                for key in requires_grad[batch_name]:
                    self.all_columns_to_keep += [key]
            if recode_functions is not None:
                for key in recode_functions:
                    self.all_columns_to_keep += [key]
        self.all_columns_to_keep = set(self.all_columns_to_keep)
        self.device = device
        print(self.all_columns_to_keep)

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()

    def __iter__(self):
        return self

    def __next__(self):
        return self.__next_tuple__(device=self.device)

    def __next_tuple__(self, device):
        """
        This method returns the next batch of data, prepared for pytorch, on GPU when is_cuda is true, as variables.
        Use variable_kargs to pass arguments to the Variable calls.
        :return: Dictionary with named inputs and outputs.
        """
        data_dict = {}
        indices_dict = {}
        try:
            batch = next(self.iterator)
        except StopIteration:
            raise StopIteration

        for loader_index, (batch_indices, batch_data) in enumerate(batch):
            loader_name = self.batch_names[loader_index]
            data_dict[loader_name] = {}
            if len(self.requires_grad) == 0:
                self.requires_grad[loader_name] = []
            for var_name in batch_data.keys():
                if var_name in self.all_columns_to_keep:
                    if var_name in self.recode_functions.keys():
                        batch_data[var_name] = self.recode_functions[var_name](batch_data[var_name])
                    try:
                        # Attempt to create tensor of data
                        requires_grad = (var_name in self.requires_grad[loader_name])
                        data_dict[loader_name][var_name] = Tensor(batch_data[var_name],
                                                                  requires_grad=requires_grad).to(device)
                    except (TypeError, RuntimeError):
                        # just pass-through any other type of object:
                        data_dict[loader_name][var_name] = batch_data[var_name]
            indices_dict[loader_name] = batch_indices
        return indices_dict, data_dict

    def close(self):
        try:
            self.iterator.close()
        except:
           pass
        del self.iterator


class ThreadedDataProvider(DataProvider):
    def __init__(self, iterator, batch_names, device=None, requires_grad=None, preload_n=6, recode_functions=None):
        # do not put on GPU in super
        super(ThreadedDataProvider, self).__init__(iterator=iterator, batch_names=batch_names, device=device,
                                                   requires_grad=requires_grad, recode_functions=recode_functions)
        self.batches_queue = Queue(maxsize=preload_n)
        self.batch_index = 0

    def __next__(self):
        """
        This method returns the next batch of data, prepared for pytorch, on GPU when is_cuda is true.
        :return: Dictionary with named inputs and outputs.
        """
        while not self.batches_queue.full():
            self.populate_queue()

        self.batch_index += 1
        try:
            return self.batches_queue.get(block=True, timeout=3)
        except Empty:
            raise StopIteration

    def populate_queue(self):
        try:
            next_indices, next_data = self.__next_tuple__(False)
            self.batches_queue.put((next_indices, next_data), block=True)
        except StopIteration:
            raise StopIteration


class MultiThreadedDataProvider(ThreadedDataProvider):
    def __init__(self, iterator, batch_names, device=None, requires_grad=None, preload_n=40, recode_functions=None,
                 num_threads=3):
        super().__init__(iterator, batch_names, device=device, requires_grad=requires_grad, preload_n=preload_n,
                         recode_functions=recode_functions)
        self.stop_iteration = False
        self.kill_threads = False
        self.num_threads = num_threads
        self.threads = []

        def add_to_queue():
            while not self.kill_threads:
                try:
                    if not self.batches_queue.full():
                        self.populate_queue()
                    else:
                        time.sleep(10 / 1000.0)
                except StopIteration:
                    self.stop_iteration = True
                    break

        for _ in range(num_threads):
            thread = Thread(target=add_to_queue, name="BatchesToDevice")
            thread.start()
            self.threads.append(thread)

        while self.batches_queue.empty():
            time.sleep(10 / 1000.0)

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()

    def __next__(self):
        """
        This method returns the next batch of data, prepared for pytorch, on GPU when is_cuda is true.
        :return: Dictionary with named inputs and outputs.
        """
        try:
            return self.batches_queue.get(block=True, timeout=3)
        except Empty:
            raise StopIteration

    def close(self):
        super().close()
        self.kill_threads = True
        time.sleep(1)

    def queues_are_empty(self):
        return self.batches_queue.empty()
