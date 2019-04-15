/*
 * Copyright (c) 2009-2012 The University of Tennessee and The University
 *                         of Tennessee Research Foundation.  All rights
 *                         reserved.
 *
 * @precisions normal z -> s d c
 *
 */

#include "common.h"
#include "parsec/data_dist/matrix/two_dim_rectangle_cyclic.h"

static int check_solution( parsec_context_t *parsec, int loud,
                           parsec_tiled_matrix_dc_t *dcA,
                           parsec_tiled_matrix_dc_t *dcB,
                           parsec_tiled_matrix_dc_t *dcX );

static int check_inverse( parsec_context_t *parsec, int loud,
                          parsec_tiled_matrix_dc_t *dcA,
                          parsec_tiled_matrix_dc_t *dcInvA,
                          parsec_tiled_matrix_dc_t *dcI );

int main(int argc, char ** argv)
{
    parsec_context_t* parsec;
    int iparam[IPARAM_SIZEOF];
    int *lu_tab;
    int info = 0;
    int i, ret = 0;
    dplasma_qrtree_t qrtree;
    extern double alpha;

    /* Set defaults for non argv iparams */
    iparam_default_facto(iparam);
    iparam_default_ibnbmb(iparam, 40, 200, 200);
    iparam[IPARAM_SMB] = 1;
    iparam[IPARAM_SNB] = 1;
    iparam[IPARAM_LDA] = -'m';
    iparam[IPARAM_LDB] = -'m';

    /* Initialize PaRSEC */
    parsec = setup_parsec(argc, argv, iparam);

    /* Make sure SMB and SNB are set to 1, since it conflicts with HQR */
    iparam[IPARAM_SMB] = 1;
    iparam[IPARAM_SNB] = 1;

    PASTE_CODE_IPARAM_LOCALS(iparam);
    PASTE_CODE_FLOPS(FLOPS_ZGETRF, ((DagDouble_t)M,(DagDouble_t)N));

    LDA = max(M, LDA);
    LDB = max(M, LDB);

    if ( M != N && check ) {
        fprintf(stderr, "Check is impossible if M != N\n");
        check = 0;
    }

    /* initializing matrix structure */
    PASTE_CODE_ALLOCATE_MATRIX(dcA, 1,
                               two_dim_block_cyclic, (&dcA, matrix_ComplexDouble, matrix_Tile,
                                                      nodes, rank, MB, NB, LDA, N, 0, 0,
                                                      M, N, SMB, SNB, P));
    PASTE_CODE_ALLOCATE_MATRIX(dcTS, 1,
                               two_dim_block_cyclic, (&dcTS, matrix_ComplexDouble, matrix_Tile,
                                                      nodes, rank, IB, NB, MT*IB, N, 0, 0,
                                                      MT*IB, N, SMB, SNB, P));
    PASTE_CODE_ALLOCATE_MATRIX(dcTT, 1,
                               two_dim_block_cyclic, (&dcTT, matrix_ComplexDouble, matrix_Tile,
                                                      nodes, rank, IB, NB, MT*IB, N, 0, 0,
                                                      MT*IB, N, SMB, SNB, P));
    PASTE_CODE_ALLOCATE_MATRIX(dcIPIV, 1,
                               two_dim_block_cyclic, (&dcIPIV, matrix_Integer, matrix_Tile,
                                                      nodes, rank, MB, 1, M, NT, 0, 0,
                                                      M, NT, SMB, SNB, P));
    PASTE_CODE_ALLOCATE_MATRIX(dcA0, check,
                               two_dim_block_cyclic, (&dcA0, matrix_ComplexDouble, matrix_Tile,
                                                      nodes, rank, MB, NB, LDA, N, 0, 0,
                                                      M, N, SMB, SNB, P));
    /* Random B check */
    PASTE_CODE_ALLOCATE_MATRIX(dcB, check,
                               two_dim_block_cyclic, (&dcB, matrix_ComplexDouble, matrix_Tile,
                                                      nodes, rank, MB, NB, LDB, NRHS, 0, 0,
                                                      M, NRHS, SMB, SNB, P));
    PASTE_CODE_ALLOCATE_MATRIX(dcX, check,
                               two_dim_block_cyclic, (&dcX, matrix_ComplexDouble, matrix_Tile,
                                                      nodes, rank, MB, NB, LDB, NRHS, 0, 0,
                                                      M, NRHS, SMB, SNB, P));
    /* Inverse check */
    PASTE_CODE_ALLOCATE_MATRIX(dcInvA, check_inv,
                               two_dim_block_cyclic, (&dcInvA, matrix_ComplexDouble, matrix_Tile,
                                                      nodes, rank, MB, NB, LDA, N, 0, 0,
                                                      M, N, SMB, SNB, P));
    PASTE_CODE_ALLOCATE_MATRIX(dcI, check_inv,
                               two_dim_block_cyclic, (&dcI, matrix_ComplexDouble, matrix_Tile,
                                                      nodes, rank, MB, NB, LDA, N, 0, 0,
                                                      M, N, SMB, SNB, P));

    lu_tab = (int *)malloc( dplasma_imin(MT, NT)*sizeof(int) );
    for(i=0; i< dplasma_imin(MT, NT); i++)
        lu_tab[i] = -1;

    /* matrix generation */
    if(loud > 2) printf("+++ Generate matrices ... ");
    dplasma_zpltmg( parsec, matrix_init, (parsec_tiled_matrix_dc_t *)&dcA, random_seed );
    dplasma_zlaset( parsec, PlasmaUpperLower, 0., 0., (parsec_tiled_matrix_dc_t *)&dcTS);
    dplasma_zlaset( parsec, PlasmaUpperLower, 0., 0., (parsec_tiled_matrix_dc_t *)&dcTT);
    dplasma_hqr_init( &qrtree,
                      PlasmaNoTrans, (parsec_tiled_matrix_dc_t *)&dcA,
                      iparam[IPARAM_LOWLVL_TREE],
                      iparam[IPARAM_HIGHLVL_TREE],
                      iparam[IPARAM_QR_TS_SZE],
                      P,/*iparam[IPARAM_QR_HLVL_SZE],*/
                      0 /*iparam[IPARAM_QR_DOMINO]*/,
                      0 /*iparam[IPARAM_QR_TSRR]  */);
    if ( check ) {
        dplasma_zlacpy( parsec, PlasmaUpperLower,
                        (parsec_tiled_matrix_dc_t *)&dcA,
                        (parsec_tiled_matrix_dc_t *)&dcA0 );
        dplasma_zplrnt( parsec, 0, (parsec_tiled_matrix_dc_t *)&dcB, random_seed+1 );
        dplasma_zlacpy( parsec, PlasmaUpperLower,
                        (parsec_tiled_matrix_dc_t *)&dcB,
                        (parsec_tiled_matrix_dc_t *)&dcX );
    }
    if ( check_inv ) {
        dplasma_zlaset( parsec, PlasmaUpperLower, 0., 1., (parsec_tiled_matrix_dc_t *)&dcI);
        dplasma_zlaset( parsec, PlasmaUpperLower, 0., 1., (parsec_tiled_matrix_dc_t *)&dcInvA);
    }
    if(loud > 2) printf("Done\n");

    /* Create PaRSEC */
    if(loud > 2) printf("+++ Computing getrf_qrf ... ");
    PASTE_CODE_ENQUEUE_KERNEL(parsec, zgetrf_qrf,
                              (&qrtree,
                               (parsec_tiled_matrix_dc_t*)&dcA,
                               (parsec_tiled_matrix_dc_t*)&dcIPIV,
                               (parsec_tiled_matrix_dc_t*)&dcTS,
                               (parsec_tiled_matrix_dc_t*)&dcTT,
                               iparam[IPARAM_QR_HLVL_SZE], alpha, lu_tab,
                               &info));
    /* lets rock! */
    PASTE_CODE_PROGRESS_KERNEL(parsec, zgetrf_qrf);
    dplasma_zgetrf_qrf_Destruct( PARSEC_zgetrf_qrf );
    if(loud > 2) printf("Done.\n");

    /* Compute percentage of LU/QR */
    {
#if defined(PARSEC_HAVE_MPI)
        {
            int *lu_tab2 = (int*)malloc( MT*sizeof(int) );
            MPI_Allreduce ( lu_tab, lu_tab2, MT, MPI_INT, MPI_MAX, MPI_COMM_WORLD);
            memcpy( lu_tab, lu_tab2, MT*sizeof(int) );
            free(lu_tab2);
        }
#endif

        int i, nblu, nbqr;
        nblu = 0;
        nbqr = dplasma_imin(MT, NT);
        for(i=0; i<nbqr; i++) {
            nblu += lu_tab[i];
        }
        nbqr -= nblu;

        // if (loud > 3 || (rank == 0 && loud)) {
        if (rank == 0 && loud) {
            printf("[%d] LU/QR repartition: %d(%.2f) LU / %d(%.2f) QR \n", rank,
                   nblu, 100. * (double)nblu / (double)(nblu+nbqr),
                   nbqr, 100. * (double)nbqr / (double)(nblu+nbqr));
            printf("[%d] lu_tab: ", rank);
            for(i=0; i<dplasma_imin(MT, NT); i++) {
                printf("%d ", lu_tab[i]);
            }
            printf("\n");
        }
    }

    if ( info != 0 ) {
        if( rank == 0 && loud ) printf("-- Factorization is suspicious (info = %d) ! \n", info );
        ret |= 1;
    }
    else if ( check ) {
        /*
         * First check with a right hand side
         */
        dplasma_ztrsmpl_qrf( parsec, &qrtree,
                             (parsec_tiled_matrix_dc_t *)&dcA,
                             (parsec_tiled_matrix_dc_t *)&dcIPIV,
                             (parsec_tiled_matrix_dc_t *)&dcX,
                             (parsec_tiled_matrix_dc_t *)&dcTS,
                             (parsec_tiled_matrix_dc_t *)&dcTT,
                             lu_tab);
        dplasma_ztrsm(parsec, PlasmaLeft, PlasmaUpper, PlasmaNoTrans, PlasmaNonUnit, 1.0,
                      (parsec_tiled_matrix_dc_t *)&dcA,
                      (parsec_tiled_matrix_dc_t *)&dcX);

        /* Check the solution */
        ret |= check_solution( parsec, (rank == 0) ? loud : 0,
                               (parsec_tiled_matrix_dc_t *)&dcA0,
                               (parsec_tiled_matrix_dc_t *)&dcB,
                               (parsec_tiled_matrix_dc_t *)&dcX);

        /*
         * Second check with inverse
         */
        if ( check_inv ) {
            dplasma_ztrsmpl_qrf( parsec, &qrtree,
                                 (parsec_tiled_matrix_dc_t *)&dcA,
                                 (parsec_tiled_matrix_dc_t *)&dcIPIV,
                                 (parsec_tiled_matrix_dc_t *)&dcInvA,
                                 (parsec_tiled_matrix_dc_t *)&dcTS,
                                 (parsec_tiled_matrix_dc_t *)&dcTT,
                                 lu_tab);
            dplasma_ztrsm(parsec, PlasmaLeft, PlasmaUpper, PlasmaNoTrans, PlasmaNonUnit, 1.0,
                          (parsec_tiled_matrix_dc_t *)&dcA,
                          (parsec_tiled_matrix_dc_t *)&dcInvA);

            /* Check the solution */
            ret |= check_inverse(parsec, (rank == 0) ? loud : 0,
                                 (parsec_tiled_matrix_dc_t *)&dcA0,
                                 (parsec_tiled_matrix_dc_t *)&dcInvA,
                                 (parsec_tiled_matrix_dc_t *)&dcI);
        }
    }

    if ( check ) {
        parsec_data_free(dcA0.mat);
        parsec_data_collection_destroy( (parsec_data_collection_t*)&dcA0);
        parsec_data_free(dcB.mat);
        parsec_data_collection_destroy( (parsec_data_collection_t*)&dcB);
        parsec_data_free(dcX.mat);
        parsec_data_collection_destroy( (parsec_data_collection_t*)&dcX);
        if ( check_inv ) {
            parsec_data_free(dcInvA.mat);
            parsec_tiled_matrix_dc_destroy((parsec_tiled_matrix_dc_t*)&dcInvA);
            parsec_data_free(dcI.mat);
            parsec_tiled_matrix_dc_destroy((parsec_tiled_matrix_dc_t*)&dcI);
        }
    }

    dplasma_hqr_finalize( &qrtree );

    parsec_data_free(dcA.mat);
    parsec_tiled_matrix_dc_destroy((parsec_tiled_matrix_dc_t*)&dcA);
    parsec_data_free(dcTS.mat);
    parsec_tiled_matrix_dc_destroy((parsec_tiled_matrix_dc_t*)&dcTS);
    parsec_data_free(dcTT.mat);
    parsec_tiled_matrix_dc_destroy((parsec_tiled_matrix_dc_t*)&dcTT);
    parsec_data_free(dcIPIV.mat);
    parsec_tiled_matrix_dc_destroy((parsec_tiled_matrix_dc_t*)&dcIPIV);
    free(lu_tab);

    cleanup_parsec(parsec, iparam);
    return ret;
}

static int check_solution( parsec_context_t *parsec, int loud,
                           parsec_tiled_matrix_dc_t *dcA,
                           parsec_tiled_matrix_dc_t *dcB,
                           parsec_tiled_matrix_dc_t *dcX )
{
    int info_solution;
    double Rnorm = 0.0;
    double Anorm = 0.0;
    double Bnorm = 0.0;
    double Xnorm, result;
    int m = dcB->m;
    double eps = LAPACKE_dlamch_work('e');

    Anorm = dplasma_zlange(parsec, PlasmaInfNorm, dcA);
    Bnorm = dplasma_zlange(parsec, PlasmaInfNorm, dcB);
    Xnorm = dplasma_zlange(parsec, PlasmaInfNorm, dcX);

    /* Compute b - A*x */
    dplasma_zgemm( parsec, PlasmaNoTrans, PlasmaNoTrans, -1.0, dcA, dcX, 1.0, dcB);

    Rnorm = dplasma_zlange(parsec, PlasmaInfNorm, dcB);

    result = Rnorm / ( ( Anorm * Xnorm + Bnorm ) * m * eps ) ;

    if ( loud > 2 ) {
        printf("============\n");
        printf("Checking the Residual of the solution \n");
        if ( loud > 3 )
            printf( "-- ||A||_oo = %e, ||X||_oo = %e, ||B||_oo= %e, ||A X - B||_oo = %e\n",
                    Anorm, Xnorm, Bnorm, Rnorm );
        printf("-- ||Ax-B||_oo/((||A||_oo||x||_oo+||B||_oo).N.eps) = %e \n", result);
    }

    if (  isnan(Xnorm) || isinf(Xnorm) || isnan(result) || isinf(result) || (result > 60.0) ) {
        if( loud ) printf("-- Solution with b is suspicious ! \n");
        info_solution = 1;
    }
    else{
        if( loud ) printf("-- Solution with b is CORRECT ! \n");
        info_solution = 0;
    }

    return info_solution;
}

static int check_inverse( parsec_context_t *parsec, int loud,
                          parsec_tiled_matrix_dc_t *dcA,
                          parsec_tiled_matrix_dc_t *dcInvA,
                          parsec_tiled_matrix_dc_t *dcI )
{
    int info_solution;
    double Anorm    = 0.0;
    double InvAnorm = 0.0;
    double Rnorm, result;
    int m = dcA->m;
    double eps = LAPACKE_dlamch_work('e');

    Anorm    = dplasma_zlange(parsec, PlasmaInfNorm, dcA   );
    InvAnorm = dplasma_zlange(parsec, PlasmaInfNorm, dcInvA);

    /* Compute I - A*A^{-1} */
    dplasma_zgemm( parsec, PlasmaNoTrans, PlasmaNoTrans, -1.0, dcA, dcInvA, 1.0, dcI);

    Rnorm = dplasma_zlange(parsec, PlasmaInfNorm, dcI);

    result = Rnorm / ( ( Anorm * InvAnorm ) * m * eps ) ;

    if ( loud > 2 ) {
        printf("============\n");
        printf("Checking the Residual of the solution \n");
        if ( loud > 3 )
            printf( "-- ||A||_oo = %e, ||A^{-1}||_oo = %e, ||A A^{-1} - I||_oo = %e\n",
                    Anorm, InvAnorm, Rnorm );
        printf("-- ||AA^{-1}-I||_oo/((||A||_oo||A^{-1}||_oo).N.eps) = %e \n", result);
    }

    if (  isnan(Rnorm) || isinf(Rnorm) || isnan(result) || isinf(result) || (result > 60.0) ) {
        if( loud ) printf("-- Solution with I is suspicious ! \n");
        info_solution = 1;
    }
    else{
        if( loud ) printf("-- Solution with I is CORRECT ! \n");
        info_solution = 0;
    }

    return info_solution;
}
