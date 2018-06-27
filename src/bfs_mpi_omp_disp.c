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
/* Author: Hoang-Vu Dang (danghvu@gmail.com)     */

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

typedef struct {
    int* oldq;
    int* newq;
    size_t v_count;
    size_t last_count;
} thread_data_t;

static thread_data_t* tdata;

static int* sendbufs;
static size_t* sendbufs_counts /* 2x actual count */;
static MPI_Request* send_reqs;
static int* send_reqs_active;
static int* recvbuf;
static unsigned num_threads;
static int NSENDER;

void make_graph_data_structure(const tuple_graph* const tg) {
    convert_graph_to_oned_csr(tg, &g);
    column = g.column;
    rowstarts = g.rowstarts;

    num_threads = omp_get_max_threads();
    omp_set_num_threads(num_threads);
    if (getenv("NSENDER"))
        NSENDER = atoi(getenv("NSENDER"));
    else 
        NSENDER = num_threads / 2;
    if (size == 1)
        NSENDER = num_threads;
    printf("[OMP] Using %d, %d sender\n", num_threads, NSENDER);

    const size_t nlocalverts = g.nlocalverts;

    tdata = (thread_data_t*) xmalloc(sizeof(thread_data_t) * NSENDER);
    for (int i = 0 ; i < NSENDER; i++) {
        tdata[i].oldq = (int*)xmalloc(nlocalverts * sizeof(int));
        tdata[i].newq = (int*)xmalloc(nlocalverts * sizeof(int));
        tdata[i].v_count = 0;
        tdata[i].last_count = 0;
    }

    visited_size = (nlocalverts + ulong_bits - 1) / ulong_bits;
    visited = (unsigned long*)xmalloc(visited_size * sizeof(unsigned long));

    sendbufs = (int*)xmalloc(num_threads * size * BATCH_SIZE * sizeof(int));
    sendbufs_counts = (size_t*)xmalloc(num_threads * size * sizeof(size_t)) /* 2x actual count */;

    send_reqs = (MPI_Request*)xmalloc(num_threads * size * sizeof(MPI_Request));
    send_reqs_active = (int*)xmalloc(num_threads * size * sizeof(int));
    recvbuf = (int*)xmalloc(num_threads * BATCH_SIZE * sizeof(int));
}

void free_graph_data_structure(void) {
    free(visited);
    free(sendbufs);
    free(sendbufs_counts);
    free(send_reqs);
    free(send_reqs_active);
    free(recvbuf);
    free_oned_csr_graph(&g);
}

static inline void insert_new_node(int tgt, int64_t src, int64_t* pred)
{
    if (!TEST_VISITEDLOC(tgt)) {
        SET_VISITEDLOC(tgt);
        if (pred[tgt] == -1) {
            if (__sync_bool_compare_and_swap(&pred[tgt], -1, src)) {
                pred[tgt] = src;
                int tid = (tgt % NSENDER);
                size_t count = __sync_fetch_and_add(&tdata[tid].v_count, 1);
                tdata[tid].newq[count] = tgt;
            }
        }
    }
}

static inline void recv_threads(unsigned tid, unsigned nth, int64_t *pred, int *num_ranks_done_)
{
    MPI_Status st;

    /* Check all MPI requests and handle any that have completed. */  
    /* Test for incoming vertices to put onto the queue. */
    while (1) {
        int count;
        MPI_Recv(&recvbuf[BATCH_OFF], BATCH_SIZE, MPI_INT, MPI_ANY_SOURCE, MPI_ANY_TAG, MPI_COMM_WORLD, &st);
        MPI_Get_count(&st, MPI_INT, &count);
        /* count == 0 is a signal from a rank that it is done sending to me
         * (using MPI's non-overtaking rules to keep that signal after all "real" messages. */
        if (count == 0) {
            (*num_ranks_done_)++;
        } else {
            int j;
            for (j = 0; j < count; j += 2) {
                int tgt = recvbuf[BATCH_OFF + j];
                int src = recvbuf[BATCH_OFF + j + 1];
                insert_new_node(tgt, VERTEX_TO_GLOBAL(st.MPI_SOURCE, src), pred);
            }
        }
        if (*num_ranks_done_ == size) {
            break;
        }
    }
}

int sum = 0;

static inline void send_threads(unsigned tid, unsigned n_threads, int64_t *pred, int *num_ranks_done) {
    thread_data_t * data = &tdata[tid];
    int* oldq = data->oldq;
    int* newq = data->newq;

    for (size_t i = 0; i < data->last_count; i++) {
        int src = oldq[i];
        unsigned j, j_end = g.rowstarts[oldq[i] + 1];
        for (j = g.rowstarts[oldq[i]]; j < j_end; ++j) {
            int64_t tgt = COLUMN(j);
            int owner = VERTEX_OWNER(tgt);
            if (owner == rank) {
                insert_new_node((int) VERTEX_LOCAL(tgt), VERTEX_TO_GLOBAL(owner, src), pred);
            } else {
                /* Wait for buffer to be available */
                if (send_reqs_active[PROC_OFF + owner]) {
                    MPI_Wait(&send_reqs[PROC_OFF + owner], MPI_STATUS_IGNORE);
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
                MPI_Wait(&send_reqs[PROC_OFF + dest], MPI_STATUS_IGNORE);
                send_reqs_active[PROC_OFF + dest] = 0;
            }
            MPI_Send(&sendbufs[BATCH_PROC_OFF + dest * BATCH_SIZE], sendbufs_counts[PROC_OFF + dest], MPI_INT, dest, tid, MPI_COMM_WORLD); 
            send_reqs_active[PROC_OFF + dest] = 0;
            sendbufs_counts[PROC_OFF + dest] = 0;
        }
    }

    if (num_threads != n_threads) {
        int lsum = __sync_fetch_and_add(&sum, 1) + 1;
        if (lsum == n_threads) {
            for (int j = 0; j < num_threads - n_threads; j++) {
                for (int offset = 1; offset < size; ++offset) {
                    int dest = MOD_SIZE(rank + offset);
                    MPI_Send(0, 0, MPI_INT, dest, tid, MPI_COMM_WORLD); /* Signal no more sends */
                }
            }
            sum = 0;
        }
    }
}

void run_bfs(int64_t root, int64_t* pred) {
    pred_glob=pred;

    const size_t nlocalverts = g.nlocalverts;
    size_t newq_count = 0;

    memset(visited, 0, visited_size * sizeof(unsigned long));
    memset(send_reqs_active, 0, num_threads * size * sizeof(int));

    /* Mark the root and put it into the queue. */
    if (VERTEX_OWNER(root) == rank) {
        SET_VISITED(root);
        pred[VERTEX_LOCAL(root)] = root;
        tdata[0].oldq[tdata[0].last_count++] = VERTEX_LOCAL(root);
    }

    while (1) {
        #pragma omp parallel
        {
            unsigned tid = omp_get_thread_num();
            int num_ranks_done = 1;
            memset(&sendbufs_counts[tid * size], 0, size * sizeof(size_t));

            #pragma omp barrier
            if (tid >= NSENDER)
                recv_threads(tid, num_threads- NSENDER, pred, &num_ranks_done);
            else
                send_threads(tid, NSENDER, pred, &num_ranks_done);

        } /* OpenMP parallel */

        /* Sum the individual thread queues to newq */
        unsigned i,j;
        for(i=0; i < NSENDER; i++) {
            newq_count += tdata[i].v_count;
            tdata[i].last_count = tdata[i].v_count;
            tdata[i].v_count = 0;
            {int * tmp = tdata[i].newq; tdata[i].newq = tdata[i].oldq; tdata[i].oldq = tmp; }
        }

        /* Test globally if all queues are empty. */
        int64_t global_newq_count;
        MPI_Allreduce(&newq_count, &global_newq_count, 1, MPI_INT64_T, MPI_SUM, MPI_COMM_WORLD);

        /* Quit if they all are empty. */
        if (global_newq_count == 0) break;
        newq_count = 0;
    }
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

