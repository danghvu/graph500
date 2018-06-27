This repository implements three versions of graph500 (bfs-only) based on the
latest reference code, using the MPI+OMP model with `MPI_THREAD_MULTIPLE` where
each OpenMP thread communitates simultenously:

- `src/bfs_mpi_omp.c`, `src/bfs_mpi_omp_disp.c`: mainly uses `MPI_Send/_Recv`,
  half of the threads are dedicated for sending, and half are dedicated for
  receiving. The difference between the two is that `_disp` version will
  distribute the vertex when it arrives, while the other will keep them in
  private queue and distribute afterward.

- `src/bfs_mpi_omp_nb.c`: is non-blocking version, using mainly `MPI_Isend/_Irecv/_Test`.

This repository can be used to evaluate MPI implementations that support
MPI_THREAD_MULTIPLE. Please also set the number of threads, and affinity rule
`OMP_NUM_THREADS=64 OMP_PLACES=cores OMP_PROC_BIND=close`.

If you use these custom implementations, please cite
["Advanced Thread Synchronization for Multithreaded MPI Implementations"](https://ieeexplore.ieee.org/abstract/document/7973717/)

```
@inproceedings{dang2017advanced,
  title={Advanced Thread Synchronization for Multithreaded {MPI} Implementations},
  author={Dang, Hoang-Vu and Seo, Sangmin and Amer, Abdelhalim and Balaji, Pavan},
  booktitle={Cluster, Cloud and Grid Computing (CCGRID), 2017 17th IEEE/ACM International Symposium on},
  pages={314--324},
  year={2017},
  organization={IEEE}
}
```
