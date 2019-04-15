/*
 * Copyright (c) 2012-2015 The University of Tennessee and The University
 *                         of Tennessee Research Foundation. All rights
 *                         reserved.
 * $COPYRIGHT
 *
 */

#ifndef _RBT_MAPPING_H_
#define _RBT_MAPPING_H_
#include "parsec/data_distribution.h"
#include "parsec/data_dist/matrix/matrix.h"

typedef struct{
  int m;
  int n;
} seg_count_t;

typedef struct{
  int m1;
  int m2;
  int n1;
  int n2;
} seg_size_t;

typedef struct{
  seg_count_t t_cnt, b_cnt, l_cnt, r_cnt, c_cnt;
  seg_size_t  t_sz,  b_sz,  l_sz,  r_sz,  c_sz;
  int c_seg_cnt_m;
  int c_seg_cnt_n;
  int tot_seg_cnt_m;
  int tot_seg_cnt_n;
  int spm, mpm, epm;
  int spn, mpn, epn;
} seg_info_t;

typedef struct parsec_seg_dc{
 parsec_tiled_matrix_dc_t super;
 parsec_tiled_matrix_dc_t *A_org;
 seg_info_t seg_info;
 int level;
}parsec_seg_dc_t;

/* forward declarations */
seg_info_t parsec_rbt_calculate_constants(const parsec_tiled_matrix_dc_t *A, int L, int ib, int jb);
void segment_to_tile(const parsec_seg_dc_t *seg_dc, int m, int n, int *m_tile, int *n_tile, uintptr_t *offset);
int type_index_to_sizes(const seg_info_t *seg, int type_index, unsigned *m_sz, unsigned *n_sz);
int segment_to_arena_index(const parsec_seg_dc_t* but_dc, int m, int n);
int segment_to_type_index(const seg_info_t *seg, int m, int n);

#endif
