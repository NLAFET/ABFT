/*
 * Copyright (c) 2012-2018 The University of Tennessee and The University
 *                         of Tennessee Research Foundation. All rights
 *                         reserved.
 * $COPYRIGHT
 *
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <assert.h>

#include "dplasma/lib/butterfly_map.h"
#include "parsec/data_internal.h"

seg_info_t parsec_rbt_calculate_constants(const parsec_tiled_matrix_dc_t *A, int L, int ib, int jb){
    int am, an, bm, bn, cm, cn, dm, dn, em, en, fm, fn;
    int mb, width, height, block_count;
    int cstartn, cendn, cstartm, cendm;
    int N, nb;
    seg_info_t seg;

    memset(&seg, 0, sizeof(seg_info_t));

    N  = A->lm;
    nb = A->nb;
    /* The matrix has to be symmetric if we are applying the random butterfly transformation */
    mb = nb;
    block_count = 1<<L;

    /* Calculate starting, middle and ending point for this butterfly */
    seg.spm = ib*N/block_count;
    seg.spn = jb*N/block_count;

    seg.epm = (ib+1)*N/block_count-1;
    seg.epn = (jb+1)*N/block_count-1;

    seg.mpm = (seg.spm + seg.epm + 1)/2;
    seg.mpn = (seg.spn + seg.epn + 1)/2;


    /* Calculate the different sizes that might appear */
    am = seg.spm%mb;
    bm = mb-am;
    cm = seg.mpm%mb;
    dm = mb-cm;
    em = (dm>bm) ? dm-bm : mb + (dm-bm);
    fm = mb-em;

    an = seg.spn%nb;
    bn = nb-an;
    cn = seg.mpn%nb;
    dn = nb-cn;
    en = (dn>bn) ? dn-bn : nb + (dn-bn);
    fn = nb-en;

    cstartn = seg.spn;
    if( bn != nb ){
      cstartn += bn;
    }
    cendn = seg.mpn;
    if( (cn != nb) && (seg.mpn > cstartn+nb) ){
        /* if there is a right type and I can fit more than a whole center
         * before the end, then I got to subtract the right type from the
         * middle, to find the end of the center type
         */
        cendn -= cn;
    }

    cstartm = seg.spm;
    if( bm != mb ){
      cstartm += bm;
    }
    cendm = seg.mpm;
    if( (cm != mb) && (seg.mpm > cstartm+mb) ){
        /* if there is a bottom type and I can fit more than a whole center
         * before the end, then I got to subtract the bottom type from the
         * middle, to find the end of the center type
         */
        cendm -= cm;
    }

    /* top edge types */
    if( bm < fm+em ) {
        if( fm < bm ) {
            height = bm-fm;
            seg.t_cnt.m = 2;
            seg.t_sz.m1 = height;
            seg.t_sz.m2 = fm;
        }else{
            height = bm;
            seg.t_cnt.m = 1;
            seg.t_sz.m1 = height;
            seg.t_sz.m2 = 0;
        }
    }

    /* left edge types */
    if( bn < fn+en ) {
        if( fn < bn ) {
            width = bn-fn;
            seg.l_cnt.n = 2;
            seg.l_sz.n1 = width;
            seg.l_sz.n2 = fn;
        }else{
            width = bn;
            seg.l_cnt.n = 1;
            seg.l_sz.n1 = width;
            seg.l_sz.n2 = 0;
        }
    }

    /* right edge types */
    if( cn < en+fn && cn && (0 < seg.mpn-(cstartn+nb)) ) {
        if( en < cn ){
            width = cn-en;
            seg.r_cnt.n = 2;
            seg.r_sz.n1 = en;
            seg.r_sz.n2 = width;
        }else{
            width = cn;
            seg.r_cnt.n = 1;
            seg.r_sz.n1 = width;
            seg.r_sz.n2 = 0;
        }
    }

    /* bottom edge types */
    if( cm < em+fm && cm && (0 < seg.mpm-(cstartm+mb)) ) {
    
        if( em < cm ){
            height = cm-em;
            seg.b_cnt.m = 2;
            seg.b_sz.m1 = em;
            seg.b_sz.m2 = height;
        }else{
            height = cm;
            seg.b_cnt.m = 1;
            seg.b_sz.m1 = height;
            seg.b_sz.m2 = 0;
        }
    }

    /* center types */
    do{
        if( (0 < fn) && (0 < seg.mpn-(cstartn+en)) ){
            seg.c_cnt.n = 2;
            seg.c_sz.n2 = fn;
            seg.c_sz.n1 = en;
        }else if( en <= seg.mpn-cstartn ){
            seg.c_cnt.n = 1;
            seg.c_sz.n1 = en;
        }
    

        if( (0 < fm) && (0 < seg.mpm-(cstartm+em)) ){
            seg.c_cnt.m = 2;
            seg.c_sz.m2 = fm;
            seg.c_sz.m1 = em;
        }else if( em <= seg.mpm-cstartm ){
            seg.c_cnt.m = 1;
            seg.c_sz.m1 = em;
        }

        if( (1 == seg.c_cnt.n) && (cendn-cstartn < nb) ){
            seg.c_seg_cnt_n = 1;
        }else{
            seg.c_seg_cnt_n = seg.c_cnt.n*(cendn-cstartn)/nb;
        }

        if( (1 == seg.c_cnt.m) && (cendm-cstartm < mb) ){
            seg.c_seg_cnt_m = 1;
        }else{
            seg.c_seg_cnt_m = seg.c_cnt.m*(cendm-cstartm)/mb;
        }
    }while(0); // just to give me a scope without looking ugly.

    seg.tot_seg_cnt_n = 2*(seg.l_cnt.n + seg.c_seg_cnt_n + seg.r_cnt.n);
    seg.tot_seg_cnt_m = 2*(seg.t_cnt.m + seg.c_seg_cnt_m + seg.b_cnt.m);

    return seg;
}

void segment_to_tile(const parsec_seg_dc_t *seg_dc, int m, int n, int *m_tile, int *n_tile, uintptr_t *offset){
    seg_info_t seg;
    int mb, nb;
    int abs_m=0, abs_n=0;
    int right=0, bottom=0;

    seg = seg_dc->seg_info;
    mb = seg_dc->A_org->mb;
    nb = seg_dc->A_org->nb;

    if( n >= seg.tot_seg_cnt_n || m >= seg.tot_seg_cnt_m ){
        fprintf(stderr,"invalid segment coordinates\n");
        return;
    }

    if( n >= seg.tot_seg_cnt_n/2 ){
        n -= seg.tot_seg_cnt_n/2;
        right = 1;
    }
    if( m >= seg.tot_seg_cnt_m/2 ){
        m -= seg.tot_seg_cnt_m/2;
        bottom = 1;
    }

    /* Horizontal */
    if( n < seg.l_cnt.n ){ /* left edge */
        abs_n = seg.spn;
        if( 1 == n ){
            abs_n += seg.l_sz.n1;
        }
    }else if( n < (seg.l_cnt.n+seg.c_seg_cnt_n) ){ /* center */
        abs_n = seg.spn + seg.l_sz.n1 + seg.l_sz.n2;
        abs_n += ((n-seg.l_cnt.n)/seg.c_cnt.n)*nb;
        if( (n-seg.l_cnt.n) % seg.c_cnt.n ){
            abs_n += seg.c_sz.n1;
        }
    }else{ /* right edge */
        abs_n = seg.mpn - (seg.r_sz.n1 + seg.r_sz.n2);
        if( n - (seg.l_cnt.n+seg.c_seg_cnt_n) ){
            abs_n += seg.r_sz.n1;
        }
    }

    /* Vertical */
    if( m < seg.t_cnt.m ){ /* top edge */
        abs_m = seg.spm;
        if( 1 == m ){
            abs_m += seg.t_sz.m1;
        }
    }else if( m < (seg.t_cnt.m+seg.c_seg_cnt_m) ){ /* center */
        abs_m = seg.spm + seg.t_sz.m1 + seg.t_sz.m2;
        abs_m += ((m-seg.t_cnt.m)/seg.c_cnt.m)*nb;
        if( (m-seg.t_cnt.m) % seg.c_cnt.m ){
            abs_m += seg.c_sz.m1;
        }
    }else{ /* bottom edge */
        abs_m = seg.mpm - (seg.b_sz.m1 + seg.b_sz.m2);
        if( m - (seg.t_cnt.m+seg.c_seg_cnt_m) ){
            abs_m += seg.b_sz.m1;
        }
    }

    if( right ){
        abs_n += seg.mpn-seg.spn;
    }
    if( bottom ){
        abs_m += seg.mpm-seg.spm;
    }

    *m_tile = abs_m/mb;
    *n_tile = abs_n/nb;
    *offset = (abs_n%nb)*mb+(abs_m%mb);

    return;
}

int type_index_to_sizes(const seg_info_t *seg, int type_index, unsigned *m_sz, unsigned *n_sz){
    unsigned width = 0;
    unsigned height = 0;
    /* int abs_m, abs_n; */
    unsigned type_index_n, type_index_m;
    int success = 1;

    if( type_index < 0 ){
        return 0;
    }

    type_index_n = type_index%6;
    type_index_m = type_index/6;

    switch(type_index_n){
        /**** left edge ****/
        case 0:
            /* abs_n = seg->spn; */
            width = seg->l_sz.n1;
            break;
        case 1:
            /* abs_n = seg->spn; */
            /* width = seg->l_sz.n1; */
            /* abs_n += seg->l_sz.n1; */
            width = seg->l_sz.n2;
            break;
        /**** center ****/
        case 2:
            /* abs_n = seg->spn + seg->l_sz.n1 + seg->l_sz.n2; */
            width = seg->c_sz.n1;
            break;
        case 3:
            /*
            abs_n = seg->spn + seg->l_sz.n1 + seg->l_sz.n2;
            abs_n += seg->c_sz.n1;
            */
            width = seg->c_sz.n2;
            break;
        /**** right edge ****/
        case 4:
            /* abs_n = seg->mpn - (seg->r_sz.n1 + seg->r_sz.n2); */
            width = seg->r_sz.n1;
            break;
        case 5:
            /*
            abs_n = seg->mpn - (seg->r_sz.n1 + seg->r_sz.n2);
            abs_n += seg->r_sz.n1;
            */
            width = seg->r_sz.n2;
            break;
        default: assert(0);
    }

    switch(type_index_m){
        /**** top edge ****/
        case 0:
            /* abs_m = seg->spm; */
            height = seg->t_sz.m1;
            break;
        case 1:
            /*
            abs_m = seg->spm;
            abs_m += seg->t_sz.m1;
            */
            height = seg->t_sz.m2;
            break;
        /**** center ****/
        case 2:
            /* abs_m = seg->spm + seg->t_sz.m1 + seg->t_sz.m2; */
            height = seg->c_sz.m1;
            break;
        case 3:
            /*
            abs_m = seg->spm + seg->t_sz.m1 + seg->t_sz.m2;
            abs_m += seg->c_sz.m1;
            */
            height = seg->c_sz.m2;
            break;
        /**** bottom edge ****/
        case 4:
            /* abs_m = seg->mpm - (seg->b_sz.m1 + seg->b_sz.m2); */
            height = seg->b_sz.m1;
            break;
        case 5:
            /*
            abs_m = seg->mpm - (seg->b_sz.m1 + seg->b_sz.m2);
            abs_m += seg->b_sz.m1;
            */
            height = seg->b_sz.m2;
            break;
        default: assert(0);
    }

    if( !height || !width ){
        success = 0;
    }

    /* *m_off = abs_m%mb; */
    /* *n_off = abs_n%nb; */
    *m_sz = height;
    *n_sz = width;

    return success;
}

int segment_to_arena_index(const parsec_seg_dc_t* but_dc, int m, int n){
    /* if using named types in the JDF or the default type, then you need to
     * offset the following value by the number of named+default types used
     */
    return segment_to_type_index(&but_dc->seg_info, m, n);
}

int segment_to_type_index(const seg_info_t *seg, int m, int n){
    int type_index_n, type_index_m, type_index;

    if( n >= seg->tot_seg_cnt_n || m >= seg->tot_seg_cnt_m ){
        fprintf(stderr,"invalid segment coordinates\n");
        return -1;
    }

    if( n >= seg->tot_seg_cnt_n/2 ){
        n -= seg->tot_seg_cnt_n/2;
    }
    if( m >= seg->tot_seg_cnt_n/2 ){
        m -= seg->tot_seg_cnt_n/2;
    }

    /* Horizontal */
    if( n < seg->l_cnt.n ){ /* left edge */
        type_index_n = 0;
        if( 1 == n ){
            type_index_n = 1;
        }
    }else if( n < (seg->l_cnt.n+seg->c_seg_cnt_n) ){ /* center */
        type_index_n = 2;
        if( (n-seg->l_cnt.n) % seg->c_cnt.n ){
            type_index_n = 3;
        }
    }else{ /* right edge */
        type_index_n = 4;
        if( n - (seg->l_cnt.n+seg->c_seg_cnt_n) ){
            type_index_n = 5;
        }
    }

    /* Vertical */
    if( m < seg->t_cnt.m ){ /* top edge */
        type_index_m = 0;
        if( 1 == m ){
            type_index_m = 1;
        }
    }else if( m < (seg->t_cnt.m+seg->c_seg_cnt_m) ){ /* center */
        type_index_m = 2;
        if( (m-seg->t_cnt.m) % seg->c_cnt.m ){
            type_index_m = 3;
        }
    }else{ /* bottom edge */
        type_index_m = 4;
        if( m - (seg->t_cnt.m+seg->c_seg_cnt_m) ){
            type_index_m = 5;
        }
    }

    type_index = type_index_m*6+type_index_n;

    return type_index;
}

