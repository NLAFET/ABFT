/*
 * Copyright (c) 2011-2018 The University of Tennessee and The University
 *                         of Tennessee Research Foundation.  All rights
 *                         reserved.
 * Copyright (c) 2013      Inria. All rights reserved.
 * $COPYRIGHT
 *
 * @precisions normal z -> s d c
 *
 */

#include "dplasma.h"
#include "dplasmatypes.h"
#include "dplasma/lib/dplasmaaux.h"
#include "parsec/private_mempool.h"

#include "zungqr_param.h"

/**
 *******************************************************************************
 *
 * @ingroup dplasma_complex64
 *
 *  dplasma_zungqr_param_New - Generates the parsec taskpool that computes the generation
 *  of an M-by-N matrix Q with orthonormal columns, which is defined as the
 *  first N columns of a product of K elementary reflectors of order M
 *
 *     Q  =  H(1) H(2) . . . H(k)
 *
 * as returned by dplasma_zgeqrf_param_New().
 *
 * WARNING: The computations are not done by this call.
 *
 *******************************************************************************
 *
 * @param[in] qrtree
 *          The structure that describes the trees used to perform the
 *          hierarchical QR factorization.
 *          See dplasma_hqr_init() or dplasma_systolic_init().
 *
 * @param[in] A
 *          Descriptor of the matrix A of size M-by-K factorized with the
 *          dplasma_zgeqrf_param_New() routine.
 *          On entry, the i-th column must contain the vector which
 *          defines the elementary reflector H(i), for i = 1,2,...,k, as
 *          returned by dplasma_zgeqrf_param_New() in the first k columns of its array
 *          argument A. N >= K >= 0.
 *
 * @param[in] TS
 *          Descriptor of the matrix TS distributed exactly as the A
 *          matrix. TS.mb defines the IB parameter of tile QR algorithm. This
 *          matrix must be of size A.mt * TS.mb - by - A.nt * TS.nb, with TS.nb
 *          == A.nb.  This matrix is initialized during the call to
 *          dplasma_zgeqrf_param_New().
 *
 * @param[in] TT
 *          Descriptor of the matrix TT distributed exactly as the A
 *          matrix. TT.mb defines the IB parameter of tile QR algorithm. This
 *          matrix must be of size A.mt * TT.mb - by - A.nt * TT.nb, with TT.nb
 *          == A.nb.  This matrix is initialized during the call to
 *          dplasma_zgeqrf_param_New().
 *
 * @param[in,out] Q
 *          Descriptor of the M-by-N matrix Q with orthonormal columns.
 *          On entry, the Id matrix.
 *          On exit, the orthonormal matrix Q.
 *          M >= N >= 0.
 *
 *******************************************************************************
 *
 * @return
 *          \retval The parsec taskpool which describes the operation to perform
 *                  NULL if one of the parameter is incorrect
 *
 *******************************************************************************
 *
 * @sa dplasma_zungqr_param_Destruct
 * @sa dplasma_zungqr_param
 * @sa dplasma_cungqr_param_New
 * @sa dplasma_dorgqr_param_New
 * @sa dplasma_sorgqr_param_New
 * @sa dplasma_zgeqrf_param_New
 *
 ******************************************************************************/
parsec_taskpool_t*
dplasma_zungqr_param_New( dplasma_qrtree_t *qrtree,
                          parsec_tiled_matrix_dc_t *A,
                          parsec_tiled_matrix_dc_t *TS,
                          parsec_tiled_matrix_dc_t *TT,
                          parsec_tiled_matrix_dc_t *Q )
{
    parsec_zungqr_param_taskpool_t* tp;
    int ib = TS->mb;

    if ( Q->n > Q->m ) {
        dplasma_error("dplasma_zungqr_param_New", "illegal size of Q (N should be smaller or equal to M)");
        return NULL;
    }
    if ( A->n > Q->n ) {
        dplasma_error("dplasma_zungqr_param_New", "illegal size of A (K should be smaller or equal to N)");
        return NULL;
    }
    if ( (TS->nt < A->nt) || (TS->mt < A->mt) ) {
        dplasma_error("dplasma_zungqr_param_New", "illegal size of TS (TS should have as many tiles as A)");
        return NULL;
    }
    if ( (TT->nt < A->nt) || (TT->mt < A->mt) ) {
        dplasma_error("dplasma_zungqr_param_New", "illegal size of TT (TT should have as many tiles as A)");
        return NULL;
    }

    tp = parsec_zungqr_param_new( A,
                                  TS,
                                  TT,
                                  Q,
                                  *qrtree,
                                  NULL );

    tp->_g_p_work = (parsec_memory_pool_t*)malloc(sizeof(parsec_memory_pool_t));
    parsec_private_memory_init( tp->_g_p_work, ib * TS->nb * sizeof(parsec_complex64_t) );

    /* Default type */
    dplasma_add2arena_tile( tp->arenas[PARSEC_zungqr_param_DEFAULT_ARENA],
                            A->mb*A->nb*sizeof(parsec_complex64_t),
                            PARSEC_ARENA_ALIGNMENT_SSE,
                            parsec_datatype_double_complex_t, A->mb );

    /* Upper triangular part of tile with diagonal */
    dplasma_add2arena_upper( tp->arenas[PARSEC_zungqr_param_UPPER_TILE_ARENA],
                             A->mb*A->nb*sizeof(parsec_complex64_t),
                             PARSEC_ARENA_ALIGNMENT_SSE,
                             parsec_datatype_double_complex_t, A->mb, 1 );

    /* Lower triangular part of tile without diagonal */
    dplasma_add2arena_lower( tp->arenas[PARSEC_zungqr_param_LOWER_TILE_ARENA],
                             A->mb*A->nb*sizeof(parsec_complex64_t),
                             PARSEC_ARENA_ALIGNMENT_SSE,
                             parsec_datatype_double_complex_t, A->mb, 0 );

    /* Little T */
    dplasma_add2arena_rectangle( tp->arenas[PARSEC_zungqr_param_LITTLE_T_ARENA],
                                 TS->mb*TS->nb*sizeof(parsec_complex64_t),
                                 PARSEC_ARENA_ALIGNMENT_SSE,
                                 parsec_datatype_double_complex_t, TS->mb, TS->nb, -1);

    return (parsec_taskpool_t*)tp;
}

/**
 *******************************************************************************
 *
 * @ingroup dplasma_complex64
 *
 *  dplasma_zungqr_param_Destruct - Free the data structure associated to an taskpool
 *  created with dplasma_zungqr_param_New().
 *
 *******************************************************************************
 *
 * @param[in,out] taskpool
 *          On entry, the taskpool to destroy.
 *          On exit, the taskpool cannot be used anymore.
 *
 *******************************************************************************
 *
 * @sa dplasma_zungqr_param_New
 * @sa dplasma_zungqr_param
 *
 ******************************************************************************/
void
dplasma_zungqr_param_Destruct( parsec_taskpool_t *tp )
{
    parsec_zungqr_param_taskpool_t *parsec_zungqr = (parsec_zungqr_param_taskpool_t *)tp;

    parsec_matrix_del2arena( parsec_zungqr->arenas[PARSEC_zungqr_param_DEFAULT_ARENA   ] );
    parsec_matrix_del2arena( parsec_zungqr->arenas[PARSEC_zungqr_param_UPPER_TILE_ARENA] );
    parsec_matrix_del2arena( parsec_zungqr->arenas[PARSEC_zungqr_param_LOWER_TILE_ARENA] );
    parsec_matrix_del2arena( parsec_zungqr->arenas[PARSEC_zungqr_param_LITTLE_T_ARENA  ] );

    parsec_private_memory_fini( parsec_zungqr->_g_p_work );
    free( parsec_zungqr->_g_p_work );

    parsec_taskpool_free(tp);
}

/**
 *******************************************************************************
 *
 * @ingroup dplasma_complex64
 *
 *  dplasma_zungqr_param - Generates an M-by-N matrix Q with
 *  orthonormal columns, which is defined as the first N columns of a product of
 *  K elementary reflectors of order M
 *
 *     Q  =  H(1) H(2) . . . H(k)
 *
 * as returned by dplasma_zgeqrf_param_New().
 *
 * WARNING: The computations are not done by this call.
 *
 *******************************************************************************
 *
 * @param[in,out] parsec
 *          The parsec context of the application that will run the operation.
 *
 * @param[in] qrtree
 *          The structure that describes the trees used to perform the
 *          hierarchical QR factorization.
 *          See dplasma_hqr_init() or dplasma_systolic_init().
 *
 * @param[in] A
 *          Descriptor of the matrix A of size M-by-K factorized with the
 *          dplasma_zgeqrf_param_New() routine.
 *          On entry, the i-th column must contain the vector which
 *          defines the elementary reflector H(i), for i = 1,2,...,k, as
 *          returned by dplasma_zgeqrf_param_New() in the first k columns of its array
 *          argument A. N >= K >= 0.
 *
 * @param[in] TS
 *          Descriptor of the matrix TS distributed exactly as the A
 *          matrix. TS.mb defines the IB parameter of tile QR algorithm. This
 *          matrix must be of size A.mt * TS.mb - by - A.nt * TS.nb, with TS.nb
 *          == A.nb.  This matrix is initialized during the call to
 *          dplasma_zgeqrf_param_New().
 *
 * @param[in] TT
 *          Descriptor of the matrix TT distributed exactly as the A
 *          matrix. TT.mb defines the IB parameter of tile QR algorithm. This
 *          matrix must be of size A.mt * TT.mb - by - A.nt * TT.nb, with TT.nb
 *          == A.nb.  This matrix is initialized during the call to
 *          dplasma_zgeqrf_param_New().
 *
 * @param[in,out] Q
 *          Descriptor of the M-by-N matrix Q with orthonormal columns.
 *          On entry, the Id matrix.
 *          On exit, the orthonormal matrix Q.
 *          M >= N >= 0.
 *
 *******************************************************************************
 *
 * @return
 *          \retval -i if the ith parameters is incorrect.
 *          \retval 0 on success.
 *
 *******************************************************************************
 *
 * @sa dplasma_zungqr_param_New
 * @sa dplasma_zungqr_param_Destruct
 * @sa dplasma_cungqr_param
 * @sa dplasma_dorgqr_param
 * @sa dplasma_sorgqr_param
 * @sa dplasma_zgeqrf_param
 *
 ******************************************************************************/
int
dplasma_zungqr_param( parsec_context_t *parsec,
                      dplasma_qrtree_t *qrtree,
                      parsec_tiled_matrix_dc_t *A,
                      parsec_tiled_matrix_dc_t *TS,
                      parsec_tiled_matrix_dc_t *TT,
                      parsec_tiled_matrix_dc_t *Q )
{
    parsec_taskpool_t *parsec_zungqr;

    if (parsec == NULL) {
        dplasma_error("dplasma_zungqr_param", "dplasma not initialized");
        return -1;
    }
    if ( Q->n > Q->m) {
        dplasma_error("dplasma_zungqr_param", "illegal number of columns in Q (N)");
        return -2;
    }
    if ( A->n > Q->n) {
        dplasma_error("dplasma_zungqr_param", "illegal number of columns in A (K)");
        return -3;
    }
    if ( A->m != Q->m ) {
        dplasma_error("dplasma_zungqr_param", "illegal number of rows in A");
        return -5;
    }

    if (dplasma_imin(Q->m, dplasma_imin(Q->n, A->n)) == 0)
        return 0;

    dplasma_qrtree_check( A, qrtree );

    parsec_zungqr = dplasma_zungqr_param_New(qrtree, A, TS, TT, Q);

    if ( parsec_zungqr != NULL ){
        parsec_context_add_taskpool(parsec, parsec_zungqr);
        dplasma_wait_until_completion(parsec);
        dplasma_zungqr_param_Destruct( parsec_zungqr );
    }
    return 0;
}
