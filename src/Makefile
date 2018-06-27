CFLAGS = -Drestrict=__restrict__ -O3 -DGRAPH_GENERATOR_MPI -DREUSE_CSR_FOR_VALIDATION -I../aml -D_GNU_SOURCES -g3
LDFLAGS = -lpthread
MPICC = mpicc

# MPICC = $(HOME)/openmpi/bin/mpicc
# MPICC = $(HOME)/mpich/bin/mpicc -D_NO_PERSIST

all: graph500_reference_bfs graph500_mpi_omp_bfs graph500_mpi_omp_bfs_disp graph500_mpi_omp_bfs_nb

GENERATOR_SOURCES = ../generator/graph_generator.c ../generator/make_graph.c ../generator/splittable_mrg.c ../generator/utils.c
SOURCES = main.c utils.c validate.c ../aml/aml.c
HEADERS = common.h csr_reference.h bitmap_reference.h

graph500_reference_bfs_sssp: bfs_reference.c $(SOURCES) $(HEADERS) $(GENERATOR_SOURCES) csr_reference.c sssp_reference.c
	$(MPICC) $(CFLAGS) -DSSSP $(LDFLAGS) -o graph500_reference_bfs_sssp bfs_reference.c sssp_reference.c csr_reference.c $(SOURCES) $(GENERATOR_SOURCES) -lm

graph500_reference_bfs: bfs_reference.c $(SOURCES) $(HEADERS) $(GENERATOR_SOURCES) csr_reference.c
	$(MPICC) $(CFLAGS) $(LDFLAGS) -o graph500_reference_bfs bfs_reference.c csr_reference.c $(SOURCES) $(GENERATOR_SOURCES) -lm

graph500_custom_bfs: bfs_custom.c $(SOURCES) $(HEADERS) $(GENERATOR_SOURCES) csr_reference.c
	$(MPICC) $(CFLAGS) $(LDFLAGS) -o graph500_custom_bfs bfs_custom.c csr_reference.c $(SOURCES) $(GENERATOR_SOURCES) -lm

graph500_mpi_omp_bfs: bfs_mpi_omp.c $(SOURCES) $(HEADERS) $(GENERATOR_SOURCES) csr_reference.c
	$(MPICC) -Wall -g3 $(CFLAGS) $(LDFLAGS) -o graph500_mpi_omp_bfs bfs_mpi_omp.c csr_reference.c $(SOURCES) $(GENERATOR_SOURCES) -lm -fopenmp -DUSE_OMP

graph500_mpi_omp_bfs_disp: bfs_mpi_omp_disp.c $(SOURCES) $(HEADERS) $(GENERATOR_SOURCES) csr_reference.c
	$(MPICC) -Wall -g3 $(CFLAGS) $(LDFLAGS) -o graph500_mpi_omp_bfs_disp bfs_mpi_omp_disp.c csr_reference.c $(SOURCES) $(GENERATOR_SOURCES) -lm -fopenmp -DUSE_OMP

graph500_mpi_omp_bfs_nb: bfs_mpi_omp_nb.c $(SOURCES) $(HEADERS) $(GENERATOR_SOURCES) csr_reference.c
	$(MPICC) -Wall -g3 $(CFLAGS) $(LDFLAGS) -o graph500_mpi_omp_bfs_nb bfs_mpi_omp_nb.c csr_reference.c $(SOURCES) $(GENERATOR_SOURCES) -lm -fopenmp -DUSE_OMP

graph500_custom_bfs_sssp: bfs_custom.c sssp_custom.c $(SOURCES) $(HEADERS) $(GENERATOR_SOURCES) csr_reference.c
	$(MPICC) $(CFLAGS) $(LDFLAGS) -o graph500_custom_bfs_sssp bfs_custom.c sssp_custom.c csr_reference.c $(SOURCES) $(GENERATOR_SOURCES) -lm

clean:
	-rm -f graph500_*
