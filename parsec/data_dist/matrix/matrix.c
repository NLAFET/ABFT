/*
 * Copyright (c) 2010-2018 The University of Tennessee and The University
 *                         of Tennessee Research Foundation.  All rights
 *                         reserved.
 */
/************************************************************
 *distributed matrix generation
 ************************************************************/

#include "parsec/parsec_config.h"
#include "parsec/parsec_internal.h"
#include "parsec/utils/debug.h"
#include "parsec/data.h"
#include "parsec/data_distribution.h"
#include "parsec/data_dist/matrix/matrix.h"
#include "parsec/data_dist/matrix/two_dim_rectangle_cyclic.h"
#include "parsec/data_dist/matrix/sym_two_dim_rectangle_cyclic.h"
#include "parsec/data_dist/matrix/two_dim_tabular.h"

#if defined(PARSEC_HAVE_MPI)
#include <mpi.h>
#endif
#include <string.h>

static uint32_t tiled_matrix_data_key(struct parsec_data_collection_s *desc, ...);

#if defined(PARSEC_PROF_TRACE)
static int      tiled_matrix_key_to_string(struct parsec_data_collection_s * desc, uint32_t datakey, char * buffer, uint32_t buffer_size);
#endif

parsec_data_t*
parsec_matrix_create_data(parsec_tiled_matrix_dc_t* matrix,
                         void* ptr,
                         int pos,
                         parsec_data_key_t key)
{
    assert( pos <= matrix->nb_local_tiles );
    return parsec_data_create( matrix->data_map + pos,
                              &(matrix->super), key, ptr,
                              matrix->bsiz * parsec_datadist_getsizeoftype(matrix->mtype) );
}

void
parsec_matrix_destroy_data( parsec_tiled_matrix_dc_t* matrix )
{
    if ( matrix->data_map != NULL ) {
        parsec_data_t **data = matrix->data_map;

        for(int i = 0; i < matrix->nb_local_tiles; i++, data++) {
            if( NULL == *data ) continue;
            parsec_data_destroy( *data );
        }

        free( matrix->data_map );
        matrix->data_map = NULL;
    }
}

parsec_data_t*
fake_data_of(parsec_data_collection_t *mat, ...)
{
    return parsec_matrix_create_data( (parsec_tiled_matrix_dc_t*)mat, NULL,
                                     0, 0 );
}

/***************************************************************************//**
 *  Internal static descriptor initializer (PLASMA code)
 **/
void parsec_tiled_matrix_dc_init( parsec_tiled_matrix_dc_t *tdesc,
                             enum matrix_type    mtyp,
                             enum matrix_storage storage,
                             int dtype, int nodes, int myrank,
                             int mb, int nb,
                             int lm, int ln,
                             int i,  int j,
                             int m,  int n)
{
    parsec_data_collection_t *o = (parsec_data_collection_t*)tdesc;

    /* Super setup */
    parsec_data_collection_init( o, nodes, myrank );

    /* Change the common data_key */
    o->data_key = tiled_matrix_data_key;

    /**
     * Setup the tiled matrix properties
     */

    /* Matrix address */
    /* tdesc->mat = NULL; */
    /* tdesc->A21 = (lm - lm%mb)*(ln - ln%nb); */
    /* tdesc->A12 = (     lm%mb)*(ln - ln%nb) + tdesc->A21; */
    /* tdesc->A22 = (lm - lm%mb)*(     ln%nb) + tdesc->A12; */

    /* Matrix properties */
    tdesc->data_map = NULL;
    tdesc->mtype    = mtyp;
    tdesc->storage  = storage;
    tdesc->dtype    = parsec_tiled_matrix_dc_type | dtype;
    tdesc->tileld   = (storage == matrix_Tile) ? mb : lm;
    tdesc->mb       = mb;
    tdesc->nb       = nb;
    tdesc->bsiz     = mb * nb;

    /* Large matrix parameters */
    tdesc->lm = lm;
    tdesc->ln = ln;

    /* Large matrix derived parameters */
    /* tdesc->lm1 = (lm/mb); */
    /* tdesc->ln1 = (ln/nb); */
    tdesc->lmt = (lm%mb==0) ? (lm/mb) : (lm/mb+1);
    tdesc->lnt = (ln%nb==0) ? (ln/nb) : (ln/nb+1);

    /* Update lm and ln to include the padding */
    if ( storage != matrix_Lapack ) {
        tdesc->lm = tdesc->lmt * tdesc->mb;
        tdesc->ln = tdesc->lnt * tdesc->nb;
    }

    /* Locally stored matrix dimensions */
    tdesc->llm = tdesc->lm;
    tdesc->lln = tdesc->ln;

    /* Submatrix parameters */
    tdesc->i = i;
    tdesc->j = j;
    tdesc->m = m;
    tdesc->n = n;

    /* Submatrix derived parameters */
    tdesc->mt = (i+m-1)/mb - i/mb + 1;
    tdesc->nt = (j+n-1)/nb - j/nb + 1;

    /* finish to update the main object properties */
#if defined(PARSEC_PROF_TRACE)
    o->key_to_string = tiled_matrix_key_to_string;
    asprintf(&(o->key_dim), "(%d, %d)", tdesc->lmt, tdesc->lnt);
#endif
}

void
parsec_tiled_matrix_dc_destroy( parsec_tiled_matrix_dc_t *tdesc )
{
    parsec_matrix_destroy_data( tdesc );
    parsec_data_collection_destroy( (parsec_data_collection_t*)tdesc );
}


parsec_tiled_matrix_dc_t *
tiled_matrix_submatrix( parsec_tiled_matrix_dc_t *tdesc,
                        int i, int j, int m, int n)
{
    int mb, nb;
    parsec_tiled_matrix_dc_t *newdesc;

    mb = tdesc->mb;
    nb = tdesc->nb;

    if ( (i < 0) || ( (i%mb) != 0 ) ) {
        parsec_warning("Invalid value of i");
        return NULL;
    }
    if ( (j < 0) || ( (j%nb) != 0 ) ) {
        parsec_warning("Invalid value of j");
        return NULL;
    }
    if ( (m < 0) || ((m+i) > tdesc->lm) ) {
        parsec_warning("Invalid value of m");
        return NULL;
    }
    if ( (n < 0) || ((n+j) > tdesc->ln) ) {
        parsec_warning("Invalid value of n");
        return NULL;
    }

    if( tdesc->dtype & two_dim_block_cyclic_type ) {
        newdesc = (parsec_tiled_matrix_dc_t*) malloc ( sizeof(two_dim_block_cyclic_t) );
        memcpy( newdesc, tdesc, sizeof(two_dim_block_cyclic_t) );
    }
    else if( tdesc->dtype & sym_two_dim_block_cyclic_type ) {
        newdesc = (parsec_tiled_matrix_dc_t*) malloc ( sizeof(sym_two_dim_block_cyclic_t) );
        memcpy( newdesc, tdesc, sizeof(sym_two_dim_block_cyclic_t) );
    }
    else if( tdesc->dtype & two_dim_tabular_type ) {
        newdesc = (parsec_tiled_matrix_dc_t*) malloc ( sizeof(two_dim_tabular_t) );
        memcpy( newdesc, tdesc, sizeof(two_dim_tabular_t) );
    } else {
        parsec_warning("Type not completely defined");
        return NULL;
    }

    // Submatrix parameters
    newdesc->i = i;
    newdesc->j = j;
    newdesc->m = m;
    newdesc->n = n;
    // Submatrix derived parameters
    newdesc->mt = (i+m-1)/mb - i/mb + 1;
    newdesc->nt = (j+n-1)/nb - j/nb + 1;
    return newdesc;
}

/* return a unique key (unique only for the specified parsec_dc) associated to a data */
static uint32_t tiled_matrix_data_key(struct parsec_data_collection_s *desc, ...)
{
    parsec_tiled_matrix_dc_t * dc;
    unsigned int m, n;
    va_list ap;
    dc = (parsec_tiled_matrix_dc_t*)desc;

    /* Get coordinates */
    va_start(ap, desc);
    m = va_arg(ap, unsigned int);
    n = va_arg(ap, unsigned int);
    va_end(ap);

    /* Offset by (i,j) to translate (m,n) in the global matrix */
    m += dc->i / dc->mb;
    n += dc->j / dc->nb;

    return ((n * dc->lmt) + m);
}

#if defined(PARSEC_PROF_TRACE)
static int  tiled_matrix_key_to_string(struct parsec_data_collection_s *desc, uint32_t datakey, char * buffer, uint32_t buffer_size)
/* return a string meaningful for profiling about data */
{
    parsec_tiled_matrix_dc_t * dc;
    unsigned int m, n;
    int res;
    dc = (parsec_tiled_matrix_dc_t*)desc;
    m = datakey % dc->lmt;
    n = datakey / dc->lmt;
    res = snprintf(buffer, buffer_size, "(%u, %u)", m, n);
    if (res < 0)
    {
        parsec_warning("Wrong key_to_string for tile (%u, %u) key: %u", m, n, datakey);
    }
    return res;
}
#endif /* PARSEC_PROF_TRACE */

/*
 * Writes the data into the file filename
 * Sequential function per node
 */
int tiled_matrix_data_write(parsec_tiled_matrix_dc_t *tdesc, char *filename)
{
    parsec_data_collection_t *dc = &(tdesc->super);
    parsec_data_t* data;
    FILE *tmpf;
    char *buf;
    int i, j, k;
    uint32_t myrank = tdesc->super.myrank;
    int eltsize =  parsec_datadist_getsizeoftype( tdesc->mtype );

    tmpf = fopen(filename, "w");
    if(NULL == tmpf) {
        parsec_warning("The file %s cannot be open", filename);
        return -1;
    }

    if ( tdesc->storage == matrix_Tile ) {
        for (i = 0 ; i < tdesc->mt ; i++)
            for ( j = 0 ; j< tdesc->nt ; j++) {
                if ( dc->rank_of( dc, i, j ) == myrank ) {
                    data = dc->data_of( dc, i, j );
                    buf = parsec_data_get_ptr(data, 0);
                    fwrite(buf, eltsize, tdesc->bsiz, tmpf );
                }
            }
    } else {
        for (i = 0 ; i < tdesc->mt ; i++)
            for ( j = 0 ; j< tdesc->nt ; j++) {
                if ( dc->rank_of( dc, i, j ) == myrank ) {
                    data = dc->data_of( dc, i, j );
                    buf = parsec_data_get_ptr(data, 0);
                    for (k=0; k<tdesc->nb; k++) {
                        fwrite(buf, eltsize, tdesc->mb, tmpf );
                        buf += eltsize * tdesc->lm;
                    }
                }
            }
    }


    fclose(tmpf);
    return 0;
}

/*
 * Read the data from the file filename
 * Sequential function per node
 */
int tiled_matrix_data_read(parsec_tiled_matrix_dc_t *tdesc, char *filename)
{
    parsec_data_collection_t *dc = &(tdesc->super);
    parsec_data_t* data;
    FILE *tmpf;
    char *buf;
    int i, j, k, ret;
    uint32_t myrank = tdesc->super.myrank;
    int eltsize =  parsec_datadist_getsizeoftype( tdesc->mtype );

    tmpf = fopen(filename, "w");
    if(NULL == tmpf) {
        parsec_warning("The file %s cannot be open", filename);
        return -1;
    }

    if ( tdesc->storage == matrix_Tile ) {
        for (i = 0 ; i < tdesc->mt ; i++)
            for ( j = 0 ; j< tdesc->nt ; j++) {
                if ( dc->rank_of( dc, i, j ) == myrank ) {
                    data = dc->data_of( dc, i, j );
                    buf = parsec_data_get_ptr(data, 0);
                    ret = fread(buf, eltsize, tdesc->bsiz, tmpf );
                    if ( ret !=  tdesc->bsiz ) {
                        parsec_warning("The read on tile(%d, %d) read %d elements instead of %d",
                                i, j, ret, tdesc->bsiz);
                        fclose(tmpf);
                        return -1;
                    }
                }
            }
    } else {
        for (i = 0 ; i < tdesc->mt ; i++)
            for ( j = 0 ; j< tdesc->nt ; j++) {
                if ( dc->rank_of( dc, i, j ) == myrank ) {
                    data = dc->data_of( dc, i, j );
                    buf = parsec_data_get_ptr(data, 0);
                    for (k=0; k < tdesc->nb; k++) {
                        ret = fread(buf, eltsize, tdesc->mb, tmpf );
                        if ( ret !=  tdesc->mb ) {
                            parsec_warning("The read on tile(%d, %d) read %d elements instead of %d",
                                    i, j, ret, tdesc->mb);
                            fclose(tmpf);
                            return -1;
                        }
                        buf += eltsize * tdesc->lm;
                    }
                }
            }
    }

    fclose(tmpf);
    return 0;
}
