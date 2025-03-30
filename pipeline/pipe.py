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
        # microbatches = list(torch.chunk(x, self.split_size, dim=0))

        # schedule = list(_clock_cycles(num_batches=self.split_size, num_partitions=len(self.partitions)))
        # print(f"[DEBUG] Pipeline Schedule (split_size={self.split_size}, num_partitions={len(self.partitions)}) \t Len Batch is {len(microbatches)}:")
        # for cycle_idx, clock in enumerate(schedule):
        #     print(f"  Clock {cycle_idx}: {clock}")

        # for clock in schedule:
        #     self.compute(microbatches, clock)

        # return torch.cat(microbatches, dim=0).to(self.devices[-1])

        micro_x = list(x.split(self.split_size, dim=0))
        schedule = list(_clock_cycles(len(micro_x), len(self.partitions)))

        for sch_t in schedule:
            self.compute(micro_x, sch_t)
        return torch.cat(micro_x, dim=0).to(self.devices[-1])


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
        
        for microbatch_idx, partition_idx in schedule:
            partition = self.partitions[partition_idx]
            device = self.devices[partition_idx]

            if microbatch_idx >=len(batches):
                print(f"[WARNING] Skipping compute task: microbatch_idx={microbatch_idx} out of range (len(batches)={len(batches)}), partition_idx={partition_idx}")
                #continue  # Skip this task gracefully

            # Ensure input batch is moved to correct device before sending
            input_batch = batches[microbatch_idx].to(device)

            # Wrap the computation in a device-correct lambda
            task = Task(lambda module=partition, x=input_batch: module(x))
            self.in_queues[partition_idx].put(task)

        for microbatch_idx, partition_idx in schedule:
            success, result = self.out_queues[partition_idx].get()
            if success:
                task, output = result
                batches[microbatch_idx] = output
            else:
                exc_info = result  # <-- result is (exc_type, exc_val, tb)
                raise exc_info[1].with_traceback(exc_info[2])

