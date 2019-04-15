/*
 * Copyright (c) 2010-2013 The University of Tennessee and The University
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

#include "zpotrf_L_abft2.h"

/**
 *******************************************************************************
 *
 * @ingroup dplasma_complex64
 *
 * dplasma_zpotrf_New - Generates the object that Computes the Cholesky
 * factorization of a symmetric positive definite (or Hermitian positive
 * definite in the complex case) matrix A.
 * The factorization has the form
 *
 *    \f[ A = \{_{L\times L^H, if uplo = PlasmaLower}^{U^H\times U, if uplo = PlasmaUpper} \f]
 *
 *  where U is an upper triangular matrix and L is a lower triangular matrix.
 *
 * WARNING: The computations are not done by this call.
 *
 *******************************************************************************
 *
 * @param[in] uplo
 *          = PlasmaUpper: Upper triangle of A is referenced;
 *          = PlasmaLower: Lower triangle of A is referenced.
 *
 * @param[in,out] A
 *          Descriptor of the distributed matrix A.
 *          On exit, the uplo part of A is overwritten with the factorized
 *          matrix.
 *
 * @param[out] info
 *          Address where to store the output information of the factorization,
 *          this is not synchronized between the nodes, and might not be set
 *          when function exists.
 *          On DAG completion:
 *              - info = 0 on all nodes if successful.
 *              - info > 0 if the leading minor of order i of A is not positive
 *                definite, so the factorization could not be completed, and the
 *                solution has not been computed. Info will be equal to i on the
 *                node that owns the diagonal element (i,i), and 0 on all other
 *                nodes.
 *
 *******************************************************************************
 *
 * @return
 *          \retval NULL if incorrect parameters are given.
 *          \retval The PaRSEC object describing the operation that can be
 *          enqueued in the runtime. It, then, needs to be
 *          destroy with the corresponding _destruct function.
 *
 *******************************************************************************
 *
 * @sa dplasma_zpotrf
 * @sa dplasma_zpotrf_Destruct
 * @sa dplasma_cpotrf_New
 * @sa dplasma_dpotrf_New
 * @sa dplasma_spotrf_New
 *
 ******************************************************************************/
parsec_taskpool_t*
dplasma_zpotrf_abft2_New( PLASMA_enum uplo, 
                          parsec_tiled_matrix_dc_t *A,
                          int *info )
{
    parsec_zpotrf_L_abft2_taskpool_t *tp;

    /* Check input arguments */
    if ((uplo != PlasmaUpper) && (uplo != PlasmaLower)) {
        dplasma_error("dplasma_zpotrf_New", "illegal value of uplo");
        return NULL /*-1*/;
    }

    *info = 0;
    if ( uplo == PlasmaUpper ) {
        dplasma_error("dplasma_zpotrf_New", "illegal value of uplo");
        return NULL /*-1*/;
    }
    tp = parsec_zpotrf_L_abft2_new( uplo, A, info, NULL, NULL, NULL, NULL);
    tp->_g_pv =  (parsec_memory_pool_t*)malloc(sizeof(parsec_memory_pool_t));
    parsec_private_memory_init( tp->_g_pv, ((A->nb)-2) * sizeof(parsec_complex64_t) );
    tp->_g_pint_v =  (parsec_memory_pool_t*)malloc(sizeof(parsec_memory_pool_t));
    parsec_private_memory_init( tp->_g_pint_v, ((A->nb)-2) * sizeof(parsec_complex64_t) );
    tp->_g_py =  (parsec_memory_pool_t*)malloc(sizeof(parsec_memory_pool_t));
    parsec_private_memory_init( tp->_g_py, ((A->nb)-2) * sizeof(parsec_complex64_t) );
    tp->_g_pint_y =  (parsec_memory_pool_t*)malloc(sizeof(parsec_memory_pool_t));
    parsec_private_memory_init( tp->_g_pint_y, ((A->nb)-2) * sizeof(parsec_complex64_t) );

    tp->_g_PRI_CHANGE = dplasma_aux_get_priority_limit( "POTRF", A );

    dplasma_add2arena_tile( tp->arenas[PARSEC_zpotrf_L_abft2_DEFAULT_ARENA],
                            A->mb*A->nb*sizeof(parsec_complex64_t),
                            PARSEC_ARENA_ALIGNMENT_SSE,
                            parsec_datatype_double_complex_t, A->mb );
    return (parsec_taskpool_t*)tp;
}

/**
 *******************************************************************************
 *
 * @ingroup 
 *
 *  dplasma_zpotrf_Destruct - Free the data structure associated to an object
 *  created with dplasma_zpotrf_New().
 *
 *******************************************************************************
 *
 * @param[in,out] o
 *          On entry, the object to destroy.
 *          On exit, the object cannot be used anymore.
 *
 *******************************************************************************
 *
 * @sa dplasma_zpotrf_New
 * @sa dplasma_zpotrf
 *
 ******************************************************************************/
void
dplasma_zpotrf_abft2_Destruct( parsec_taskpool_t *tp )
{
    parsec_zpotrf_L_abft2_taskpool_t *parsec_zpotrf_abft2 = (parsec_zpotrf_L_abft2_taskpool_t *)tp;

    parsec_matrix_del2arena( parsec_zpotrf_abft2->arenas[PARSEC_zpotrf_L_abft2_DEFAULT_ARENA] );
    parsec_private_memory_fini( parsec_zpotrf_abft2->_g_pv );
    parsec_private_memory_fini( parsec_zpotrf_abft2->_g_py );
    parsec_private_memory_fini( parsec_zpotrf_abft2->_g_pint_v  );
    parsec_private_memory_fini( parsec_zpotrf_abft2->_g_pint_y  );
    free( parsec_zpotrf_abft2->_g_pv );
    free( parsec_zpotrf_abft2->_g_py );
    free( parsec_zpotrf_abft2->_g_pint_v  );
    free( parsec_zpotrf_abft2->_g_pint_y  );
    parsec_taskpool_free(tp);
}

/**
 *******************************************************************************
 *
 * @ingroup dplasma_complex64
 *
 * dplasma_zpotrf - Computes the Cholesky factorization of a symmetric positive
 * definite (or Hermitian positive definite in the complex case) matrix A.
 * The factorization has the form
 *
 *    \f[ A = \{_{L\times L^H, if uplo = PlasmaLower}^{U^H\times U, if uplo = PlasmaUpper} \f]
 *
 *  where U is an upper triangular matrix and L is a lower triangular matrix.
 *
 *******************************************************************************
 *
 * @param[in,out] parsec
 *          The PaRSEC context of the application that will run the operation.
 *
 * @param[in] uplo
 *          = PlasmaUpper: Upper triangle of A is referenced;
 *          = PlasmaLower: Lower triangle of A is referenced.
 *
 * @param[in] A
 *          Descriptor of the distributed matrix A.
 *          On exit, the uplo part of A is overwritten with the factorized
 *          matrix.
 *
 *******************************************************************************
 *
 * @return
 *          \retval -i if the ith parameters is incorrect.
 *          \retval 0 on success.
 *          \retval > 0 if the leading minor of order i of A is not positive
 *          definite, so the factorization could not be completed, and the
 *          solution has not been computed. Info will be equal to i on the node
 *          that owns the diagonal element (i,i), and 0 on all other nodes.
 *
 *******************************************************************************
 *
 * @sa dplasma_zpotrf_New
 * @sa dplasma_zpotrf_Destruct
 * @sa dplasma_cpotrf
 * @sa dplasma_dpotrf
 * @sa dplasma_spotrf
 *
 ******************************************************************************/
int
dplasma_zpotrf_abft2( parsec_context_t *parsec,
                      PLASMA_enum uplo,
                      parsec_tiled_matrix_dc_t *A )
{
    parsec_taskpool_t *parsec_zpotrf_abft2 = NULL;
    int info = 0, ginfo = 0 ;

    parsec_zpotrf_abft2 = dplasma_zpotrf_abft2_New( uplo, A, &info );

    if ( parsec_zpotrf_abft2 != NULL )
    {
        parsec_context_add_taskpool( parsec, parsec_zpotrf_abft2);
        dplasma_wait_until_completion(parsec);
        dplasma_zpotrf_abft2_Destruct( parsec_zpotrf_abft2 );
    }

#if defined(HAVE_MPI)
    MPI_Allreduce( &info, &ginfo, 1, MPI_INT, MPI_MAX, MPI_COMM_WORLD);
#else
    ginfo = info;
#endif
    return ginfo;
}
