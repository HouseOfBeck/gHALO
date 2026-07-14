# Validation Baselines

## borg-validation-v1

System:
- Borg
- AMD Instinct MI250X
- ROCm 6.4.2
- Cray MPICH 8.1.31

Validated:
- MPI-HIP 1 node
- MPI-HIP 2 nodes
- RCCL conservative 1 node
- RCCL conservative 2 nodes
- RCCL stream ordered 1 node
- RCCL stream ordered 2 nodes

Environment:
- MPICH_GPU_SUPPORT_ENABLED=1
- MPICH_SMP_SINGLE_COPY_MODE=NONE
