from typing import Any, Iterable, Iterator, List, Optional, Union, Sequence, Tuple, cast

import torch
from torch import Tensor, nn
import torch.autograd
import torch.cuda
from .worker import Task, create_workers
from .partition import _split_module

# ASSIGNMENT 4.2
def _clock_cycles(num_batches: int, num_partitions: int) -> Iterable[List[Tuple[int, int]]]:
    '''Generate schedules for each clock cycle.

    An example of the generated schedule for m=3 and n=3 is as follows:
    
    k (i,j) (i,j) (i,j)
    - ----- ----- -----
    0 (0,0)
    1 (1,0) (0,1)
    2 (2,0) (1,1) (0,2)
    3       (2,1) (1,2)
    4             (2,2)

    where k is the clock number, i is the index of micro-batch, and j is the index of partition.

    Each schedule is a list of tuples. Each tuple contains the index of micro-batch and the index of partition.
    This function should yield schedules for each clock cycle.
    '''
    '''Generate schedules for each clock cycle.'''
    total_cycles = num_batches + num_partitions - 1

    for clock in range(total_cycles):
        tasks = []
        for microbatch in range(num_batches):
            partition = clock - microbatch
            if 0 <= partition < num_partitions:
                tasks.append((microbatch, partition))
        yield tasks

class Pipe(nn.Module):
    def __init__(
        self,
        module: nn.ModuleList,
        split_size: int = 1,
    ) -> None:
        super().__init__()

        self.split_size = int(split_size)
        self.partitions, self.devices = _split_module(module)
        (self.in_queues, self.out_queues) = create_workers(self.devices)

    # ASSIGNMENT 4.2
    def forward(self, x):
        ''' Forward the input x through the pipeline. The return value should be put in the last device.

        Hint:
        1. Divide the input mini-batch into micro-batches.
        2. Generate the clock schedule.
        3. Call self.compute to compute the micro-batches in parallel.
        4. Concatenate the micro-batches to form the mini-batch and return it.
        
        Please note that you should put the result on the last device. Putting the result on the same device as input x will lead to pipeline parallel training failing.
        '''
        microbatches = list(torch.chunk(x, self.split_size, dim=0))
        batches = microbatches

        schedule = list(_clock_cycles(num_batches=self.split_size, num_partitions=len(self.partitions)))

        for clock in schedule:
            self.compute(batches, clock)

        return torch.cat(batches, dim=0).to(self.devices[-1])


    # ASSIGNMENT 4.2
    def compute(self, batches, schedule: List[Tuple[int, int]]) -> None:
        '''Compute the micro-batches in parallel.

        Hint:
        1. Retrieve the partition and microbatch from the schedule.
        2. Use Task to send the computation to a worker. 
        3. Use the in_queues and out_queues to send and receive tasks.
        4. Store the result back to the batches.
        '''
        partitions = self.partitions
        devices = self.devices
        
        results = {}  # Store outputs of (microbatch_idx, partition_idx)

        # Submit tasks for each (microbatch, partition) in schedule
        for microbatch_idx, partition_idx in schedule:
            partition = self.partitions[partition_idx]
            device = self.devices[partition_idx]

            if partition_idx == 0:
                # First partition uses original input batch
                input_batch = batches[microbatch_idx].to(device)
            else:
                # Later partitions use output from previous partition
                input_batch = results[(microbatch_idx, partition_idx - 1)].to(device)

            task = Task(lambda module=partition, x=input_batch: module(x))
            self.in_queues[partition_idx].put(task)

        # Collect results from each partition
        for microbatch_idx, partition_idx in schedule:
            success, result = self.out_queues[partition_idx].get()
            if success:
                task, output = result
                results[(microbatch_idx, partition_idx)] = output
            else:
                exc_info = result
                raise exc_info[1].with_traceback(exc_info[2])

        # Update batches with the final output (after last partition)
        last_partition = len(self.partitions) - 1
        for i in range(self.split_size):
            final_key = (i, last_partition)
            if final_key in results:
                batches[i] = results[final_key]
            else:
                max_partition = max(k[1] for k in results if k[0] == i)
                batches[i] = results[(i, max_partition)]

