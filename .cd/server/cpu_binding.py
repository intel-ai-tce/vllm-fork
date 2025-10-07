# SPDX-License-Identifier: Apache-2.0
import os
from importlib import util
from typing import Optional

class CPU_Binding():

    def __init__(self,
                 world_size: int,
                 rank: int,
                 num_allocated_cpu: int):
        self.world_size = world_size
        self.rank = rank
        self.num_allocated_cpu = num_allocated_cpu
    def get_cpus_id_binding_based_on_numa_nodes(self) -> str:
        """Return CPUs id binding based on NUMA nodes.
        """
        rank_to_cpus = '' #self.local_omp_cpuid
        # Setup OpenMP thread affinity based on NUMA nodes automatically
        world_size = self.world_size #self.vllm_config.parallel_config.world_size
        libnuma_found = util.find_spec("numa") is not None
        psutil_found = util.find_spec("psutil") is not None
        if libnuma_found and psutil_found:
            import psutil
            from numa import info
            cpu_count = psutil.cpu_count(logical=False)
            cpus_allow_list = psutil.Process().cpu_affinity()
            numa_size = info.get_num_configured_nodes()
            cpu_count_per_numa = cpu_count // numa_size
            allocated_cpu_per_numa = self.num_allocated_cpu // numa_size 
            #num_of_reserved_cpu = min(num_reserved_cpu,
            #                          cpu_count_per_numa // 2)

            # check allow node_to_cpus list
            node_to_cpus = []
            for i in range(numa_size):
                node_intersect = set(
                    info.node_to_cpus(i)).intersection(cpus_allow_list)
                if bool(node_intersect):
                    node_to_cpus.append(list(node_intersect))

            if world_size > len(node_to_cpus):
                print(
                    "Auto thread-binding failed due to "
                    "world size: %d is larger than "
                    "allowed NUMA nodes number: %d."
                    "Please try to bind threads manually.", world_size,
                    len(node_to_cpus))
            else:
                start = cpu_count_per_numa - allocated_cpu_per_numa
                rank_to_cpus_list = node_to_cpus[self.rank][start:cpu_count_per_numa]
                rank_to_cpus = ','.join(str(x) for x in rank_to_cpus_list)
                print("rank %d auto thread-binding list: %s", self.rank, rank_to_cpus)
        else:
            print(
                "Auto thread-binding is not supported due to "
                "the lack of package numa and psutil,"
                "fallback to no thread-binding. To get better performance,"
                "please try to manually bind threads.")
        return rank_to_cpus

if __name__=="__main__":
    num_allocated_cpu = 18
    world_size = 2
    for i in range(world_size):
        cpu_binder = CPU_Binding(world_size, i, num_allocated_cpu)
        rank_to_cpus = cpu_binder.get_cpus_id_binding_based_on_numa_nodes()
        print(rank_to_cpus)
