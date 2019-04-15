/*
 * Copyright (c) 2009-2017 The University of Tennessee and The University
 *                         of Tennessee Research Foundation.  All rights
 *                         reserved.
 */
#ifndef __VECTOR_TWO_DIM_CYCLIC_H__
#define __VECTOR_TWO_DIM_CYCLIC_H__

#include "parsec/data_dist/matrix/matrix.h"
#include "parsec/data_dist/matrix/grid_2Dcyclic.h"

BEGIN_C_DECLS

/*******************************************************************
 * distributed data vector and basic functionalities
 *******************************************************************/
typedef enum vector_distrib {
    PlasmaVectorRow,
    PlasmaVectorCol,
    PlasmaVectorDiag
} vector_distrib_t;

/*
 * Vector structure inheriting from parsec_tiled_matrix_dc_t
 * Follows the same distribution than the diagonal tiles of the
 * two_dim_block_cyclic_t structure.
 */
typedef struct vector_two_dim_cyclic_s {
    parsec_tiled_matrix_dc_t super;
    grid_2Dcyclic_t     grid;
    vector_distrib_t    distrib; /**< Distribution used for the vector: Row, Column or diagonal */
    int   lcm;                   /**< number of processors present on diagonal */
    void *mat;                   /**< pointer to the beginning of the matrix   */
} vector_two_dim_cyclic_t;

/**
 * Initialize the description of a 2-D block cyclic distributed vector.
 *
 * @param dc matrix description structure, already allocated, that will be initialize
 * @param mtype type of data used for this matrix
 * @param nodes number of nodes
 * @param myrank rank of the local node (as of mpi rank)
 * @param mb number of elements in a segment
 * @param lm number of elements in the full vector
 * @param i starting element index for the computation on a subvector
 * @param m number of elements of the entire subvector
 * @param process_GridRows number of row of processes of in the process grid (has to divide nodes)
 */
void vector_two_dim_cyclic_init(vector_two_dim_cyclic_t * vdesc,
                                enum matrix_type    mtype,
                                enum vector_distrib distrib,
                                int nodes, int myrank,
                                int mb, int lm, int i, int m,
                                int process_GridRows );

void vector_two_dim_cyclic_supertiled_view( vector_two_dim_cyclic_t* target,
                                            vector_two_dim_cyclic_t* origin,
                                            int rst );

END_C_DECLS

#endif /* __VECTOR_TWO_DIM_CYCLIC_H__*/
