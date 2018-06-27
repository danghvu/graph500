/* Copyright (C) 2010-2011 The Trustees of Indiana University.             */
/*                                                                         */
/* Use, modification and distribution is subject to the Boost Software     */
/* License, Version 1.0. (See accompanying file LICENSE_1_0.txt or copy at */
/* http://www.boost.org/LICENSE_1_0.txt)                                   */
/*                                                                         */
/*  Authors: Jeremiah Willcock                                             */
/*           Andrew Lumsdaine                                              */

/* For this custom version                        */
/* Copyright (C) 2016-2018 University of Illinois */
/* Author: Hoang-Vu Dang (danghvu@gmail.com)      */

/* This non-blocking version is adapted from an implementation of                         */
/* Abdelhalim Amer (aamer@anl.gov) using graph500 version 2.2, with algorithm from        */
/* "Characterizing MPI and hybrid MPI+Threads applications at scale: case study with BFS" */

#include "common.h"
#include "aml.h"
#include "csr_reference.h"
#include "bitmap_reference.h"

#include <mpi.h>
#include <omp.h>
#include <stdint.h>
#include <inttypes.h>
#include <stdlib.h>
#include <stddef.h>
#include <string.h>
#include <limits.h>
#include <assert.h>
#include <math.h>

#define BATCH_SIZE (256*1024)
#define BATCH_OFF (tid * BATCH_SIZE) /* Batch offset without destination process id */
#define PROC_OFF (tid * size) /* Offset in data with destination process id */
#define BATCH_PROC_OFF (BATCH_OFF * size) /* Batch offset with destination process id */

unsigned long *visited;
int64_t visited_size;
int64_t *pred_glob, *column;
unsigned int *rowstarts;
oned_csr_graph g;

static int* oldq;
static int* newq;
static size_t oldq_count;
static size_t newq_count;
static int* threadq;
static size_t* threadq_count;
static int* sendbufs;
static size_t* sendbufs_counts /* 2x actual count */;
static MPI_Request* send_reqs;
static MPI_Request* recv_reqs;

static int* send_reqs_active;
static int* recvbuf;
static unsigned num_threads;

void make_graph_data_structure(const tuple_graph* const tg) {
    convert_graph_to_oned_csr(tg, &g);
    column = g.column;
    rowstarts = g.rowstarts;

    num_threads = omp_get_max_threads();
    omp_set_num_threads(num_threads);
    printf("[OMP] Using %d\n", num_threads);

    const size_t nlocalverts = g.nlocalverts;
    oldq = (int*)xmalloc(nlocalverts * sizeof(int));
    newq = (int*)xmalloc(nlocalverts * sizeof(int));

    visited_size = (nlocalverts + ulong_bits - 1) / ulong_bits;
    visited = (unsigned long*)xmalloc(visited_size * sizeof(unsigned long));

    sendbufs = (int*)xmalloc(num_threads * size * BATCH_SIZE * sizeof(int));
    sendbufs_counts = (size_t*)xmalloc(num_threads * size * sizeof(size_t)) /* 2x actual count */;

    send_reqs = (MPI_Request*)xmalloc(num_threads * size * sizeof(MPI_Request));
    recv_reqs = (MPI_Request*)xmalloc(num_threads * sizeof(MPI_Request));

    send_reqs_active = (int*)xmalloc(num_threads * size * sizeof(int));
    recvbuf = (int*)xmalloc(num_threads * BATCH_SIZE * sizeof(int));

    /* Set up per thread queues */
    threadq = (int*)xmalloc(num_threads * nlocalverts * sizeof(int));
    threadq_count = (size_t*)xmalloc(num_threads * sizeof(size_t));
}

void free_graph_data_structure(void) {
    free(oldq);
    free(newq);
    free(threadq);
    free(threadq_count);
    free(visited);
    free(sendbufs);
    free(sendbufs_counts);
    free(send_reqs);
    free(send_reqs_active);
    free(recvbuf);
    free_oned_csr_graph(&g);
}

static inline void insert_new_node(int tgt, int64_t src, int* tempq, size_t *v_count, int64_t* pred)
{
    if (!TEST_VISITEDLOC(tgt)) {
        SET_VISITEDLOC(tgt);
        if (pred[tgt] == -1) {
            pred[tgt] = src;
            tempq[(*v_count)] = tgt;
            (*v_count)++;
        }
    }
}

static inline int try_receive(unsigned tid, int64_t *pred, int *num_ranks_done_, size_t* v_count)
{
    int *tempq = (int*) &threadq[tid * g.nlocalverts];
    MPI_Status st;
    int count;
    int flag = 0;
    MPI_Test(&recv_reqs[tid], &flag, &st);
    if (!flag) return 1;

    MPI_Get_count(&st, MPI_INT, &count);
    if (count == 0) {
        (*num_ranks_done_)++;
    } else {
        int j;
        for (j = 0; j < count; j += 2) {
            int tgt = recvbuf[BATCH_OFF + j];
            int src = recvbuf[BATCH_OFF + j + 1];
            insert_new_node(tgt, VERTEX_TO_GLOBAL(st.MPI_SOURCE, src), tempq, v_count, pred);
        }
    }

    if (*num_ranks_done_ == size) {
        return 0;
    } else {
        MPI_Irecv(&recvbuf[BATCH_OFF], BATCH_SIZE, MPI_INT, MPI_ANY_SOURCE, tid, MPI_COMM_WORLD, &recv_reqs[tid]);
        return 1;
    }
}

static inline void send_threads(unsigned tid, int64_t *pred, int *num_ranks_done, size_t *v_count) {
    int *tempq = (int*) &threadq[tid * g.nlocalverts];
    int receiving = 0;

    if (size > 1) {
        receiving = 1;
        MPI_Irecv(&recvbuf[BATCH_OFF], BATCH_SIZE, MPI_INT, MPI_ANY_SOURCE, tid, MPI_COMM_WORLD, &recv_reqs[tid]);
    }

    for (size_t i = tid; i < oldq_count; i+= num_threads) {
        int src = oldq[i];
        /* Iterate through its incident edges. */
        unsigned j, j_end = g.rowstarts[oldq[i] + 1];
        for (j = g.rowstarts[oldq[i]]; j < j_end; ++j) {
            int64_t tgt = COLUMN(j);
            assert(tgt >= 0);
            int owner = VERTEX_OWNER(tgt);
            /* If the other endpoint is mine, update the visited map, predecessor
             * map, and next-level queue locally; otherwise, send the target and
             * the current vertex (its possible predecessor) to the target's owner.
             * */
            if (owner == rank) {
                insert_new_node((int) VERTEX_LOCAL(tgt), VERTEX_TO_GLOBAL(owner, src), tempq, v_count, pred);
            } else {
                /* Wait for buffer to be available */
                if (send_reqs_active[PROC_OFF + owner]) {
                    int flag = 0;
                    while (!flag) {
                        if (receiving)
                            receiving = try_receive(tid, pred, num_ranks_done, v_count);
                        MPI_Test(&send_reqs[PROC_OFF + owner], &flag, MPI_STATUS_IGNORE);
                    }
                    send_reqs_active[PROC_OFF + owner] = 0;
                }

                size_t c = sendbufs_counts[PROC_OFF + owner];
                sendbufs[BATCH_PROC_OFF + owner * BATCH_SIZE + c] = VERTEX_LOCAL(tgt);
                sendbufs[BATCH_PROC_OFF + owner * BATCH_SIZE + c + 1] = src;
                sendbufs_counts[PROC_OFF + owner] += 2;
                if (sendbufs_counts[PROC_OFF + owner] == BATCH_SIZE) {
                    MPI_Isend(&sendbufs[BATCH_PROC_OFF + owner * BATCH_SIZE], BATCH_SIZE,
                              MPI_INT, owner, tid, MPI_COMM_WORLD, &send_reqs[PROC_OFF + owner]);
                    send_reqs_active[PROC_OFF + owner] = 1;
                    sendbufs_counts[PROC_OFF + owner] = 0;
                }
            }
        }
    }

    /* Flush any coalescing buffers that still have messages. */
    int offset;
    for (offset = 1; offset < size; ++offset) {
        int dest = MOD_SIZE(rank + offset);
        if(sendbufs_counts[PROC_OFF + dest] != 0) {
            if (send_reqs_active[PROC_OFF + dest]) {
                int flag = 0;
                while (!flag) {
                    if (receiving)
                        receiving = try_receive(tid, pred, num_ranks_done, v_count);
                    MPI_Test(&send_reqs[PROC_OFF + dest], &flag, MPI_STATUS_IGNORE);
                }
                send_reqs_active[PROC_OFF + dest] = 0;
            }
            MPI_Isend(&sendbufs[BATCH_PROC_OFF + dest * BATCH_SIZE],
                      sendbufs_counts[PROC_OFF + dest], MPI_INT, dest, tid,
                      MPI_COMM_WORLD, &send_reqs[PROC_OFF + dest]); 
            send_reqs_active[PROC_OFF + dest] = 1;
            sendbufs_counts[PROC_OFF + dest] = 0;
        } else {
            send_reqs_active[PROC_OFF + dest] = 0;
        }
    }

    #pragma omp barrier 
    {
        for (int offset = 1; offset < size; ++offset) {
            int dest = MOD_SIZE(rank + offset);
            if (send_reqs_active[PROC_OFF + dest]) {
                int flag = 0;
                while (!flag) {
                    if (receiving)
                        receiving = try_receive(tid, pred, num_ranks_done, v_count);
                    MPI_Test(&send_reqs[PROC_OFF + dest], &flag, MPI_STATUS_IGNORE);
                }
                send_reqs_active[PROC_OFF + dest] = 0;
            }

            MPI_Send(0, 0, MPI_INT, dest, tid, MPI_COMM_WORLD); /* Signal no more sends */
        }
    }

    while (receiving) {
        receiving = try_receive(tid, pred, num_ranks_done, v_count);
    }
}

void run_bfs(int64_t root, int64_t* pred) {
    pred_glob=pred;

    const size_t nlocalverts = g.nlocalverts;
    oldq_count = 0;
    newq_count = 0;

    memset(visited, 0, visited_size * sizeof(unsigned long));
    memset(send_reqs_active, 0, num_threads * size * sizeof(int));

    /* Mark the root and put it into the queue. */
    if (VERTEX_OWNER(root) == rank) {
        SET_VISITED(root);
        pred[VERTEX_LOCAL(root)] = root;
        oldq[oldq_count++] = VERTEX_LOCAL(root);
    }

    while (1) {
        #pragma omp parallel
        {
            unsigned tid = omp_get_thread_num();
            int num_ranks_done = 1;

            memset(&sendbufs_counts[tid * size], 0, size * sizeof(size_t));
            size_t v_count = 0;

            send_threads(tid, pred, &num_ranks_done, &v_count);

            threadq_count[tid] = v_count;
        } /* OpenMP parallel */

        /* Sum the individual thread queues to newq */
        unsigned i,j;
        for(i=0; i < num_threads; i++) {
            for(j=0; j < threadq_count[i]; j++) {
                newq[newq_count] = threadq[i*nlocalverts + j];
                newq_count++;
            }
        }

        /* Test globally if all queues are empty. */
        int64_t global_newq_count;
        MPI_Allreduce(&newq_count, &global_newq_count, 1, MPI_INT64_T, MPI_SUM, MPI_COMM_WORLD);

        /* Quit if they all are empty. */
        if (global_newq_count == 0) break;

        /* Swap old and new queues; clear new queue for next level. */
        {int* temp = oldq; oldq = newq; newq = temp;}
        oldq_count = newq_count;
        newq_count = 0;
    }

    MPI_Barrier(MPI_COMM_WORLD);
}

size_t get_nlocalverts_for_pred(void) {
    return g.nlocalverts;
}

//user provided function to initialize predecessor array to whatevere value user needs
void clean_pred(int64_t* pred) {
    int i;
    for(i=0;i<g.nlocalverts;i++) pred[i]=-1;
}

//we need edge count to calculate teps. Validation will check if this count is correct
//user should change this function if another format (not standart CRS) used
void get_edge_count_for_teps(int64_t* edge_visit_count) {
    long i,j;
    long edge_count=0;
    for(i=0;i<g.nlocalverts;i++)
        if(pred_glob[i]!=-1) {
            for(j=g.rowstarts[i];j<g.rowstarts[i+1];j++)
                if(COLUMN(j)<=VERTEX_TO_GLOBAL(my_pe(),i))
                    edge_count++;
        }
    aml_long_allsum(&edge_count);
    *edge_visit_count=edge_count;
}

