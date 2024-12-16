#!/bin/sh

# Current file is only a shortcut

. /work/projects/ulhpc-tutorials/PS10-Horovod/soft/miniconda/scripts/env.sh

conda activate dsenv

module unload system/CUDA/11.1.1-iccifort-2020.4.304
module unload system/CUDAcore/11.1.1

