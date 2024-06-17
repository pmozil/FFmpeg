/*
 * Copyright (C) 2007 Marco Gerards <marco@gnu.org>
 * Copyright (C) 2009 David Conrad
 * Copyright (C) 2011 Jordi Ortiz
 *
 * This file is part of FFmpeg.
 *
 * FFmpeg is free software; you can redistribute it and/or
 * modify it under the terms of the GNU Lesser General Public
 * License as published by the Free Software Foundation; either
 * version 2.1 of the License, or (at your option) any later version.
 *
 * FFmpeg is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
 * Lesser General Public License for more details.
 *
 * You should have received a copy of the GNU Lesser General Public
 * License along with FFmpeg; if not, write to the Free Software
 * Foundation, Inc., 51 Franklin Street, Fifth Floor, Boston, MA 02110-1301 USA
 */

/**
 * @file
 * Dirac Decoder
 * @author Marco Gerards <marco@gnu.org>, David Conrad, Jordi Ortiz <nenjordi@gmail.com>
 */


#ifndef DIRACDEC_H
#define DIRACDEC_H

#include "libavutil/mem.h"
#include "libavutil/mem_internal.h"
#include "libavutil/pixdesc.h"
#include "libavutil/thread.h"
#include "avcodec.h"
#include "get_bits.h"
#include "codec_internal.h"
#include "decode.h"
#include "golomb.h"
#include "dirac_arith.h"
#include "dirac_vlc.h"
#include "mpegvideoencdsp.h"
#include "dirac_dwt.h"
#include "dirac.h"
#include "diractab.h"
#include "diracdsp.h"
#include "videodsp.h"
#include "hwaccel_internal.h"

#define EDGE_WIDTH 16

/**
 * The spec limits this to 3 for frame coding, but in practice can be as high as 6
 */
#define MAX_REFERENCE_FRAMES 8
#define MAX_DELAY 5         /* limit for main profile for frame coding (TODO: field coding) */
#define MAX_FRAMES (MAX_REFERENCE_FRAMES + MAX_DELAY + 1)
#define MAX_QUANT 255        /* max quant for VC-2 */
#define MAX_BLOCKSIZE 32    /* maximum xblen/yblen we support */

/**
 * DiracBlock->ref flags, if set then the block does MC from the given ref
 */
#define DIRAC_REF_MASK_REF1   1
#define DIRAC_REF_MASK_REF2   2
#define DIRAC_REF_MASK_GLOBAL 4

/**
 * Value of Picture.reference when Picture is not a reference picture, but
 * is held for delayed output.
 */
#define DELAYED_PIC_REF 4

#define CALC_PADDING(size, depth)                       \
    (((size + (1 << depth) - 1) >> depth) << depth)

#define DIVRNDUP(a, b) (((a) + (b) - 1) / (b))

typedef struct {
    AVFrame *avframe;
    int interpolated[3];    /* 1 if hpel[] is valid */
    uint8_t *hpel[3][4];
    uint8_t *hpel_base[3][4];
    int reference;
    unsigned picture_number;
} DiracFrame;

typedef struct {
    union {
        int16_t mv[2][2];
        int16_t dc[3];
    } u; /* anonymous unions aren't in C99 :( */
    uint8_t ref;
} DiracBlock;

typedef struct SubBand {
    int level;
    int orientation;
    int stride; /* in bytes */
    int width;
    int height;
    int pshift;
    int quant;
    uint8_t *ibuf;
    struct SubBand *parent;

    /* for low delay */
    unsigned length;
    const uint8_t *coeff_data;
} SubBand;

typedef struct Plane {
    DWTPlane idwt;

    int width;
    int height;
    ptrdiff_t stride;

    /* block length */
    uint8_t xblen;
    uint8_t yblen;
    /* block separation (block n+1 starts after this many pixels in block n) */
    uint8_t xbsep;
    uint8_t ybsep;
    /* amount of overspill on each edge (half of the overlap between blocks) */
    uint8_t xoffset;
    uint8_t yoffset;

    SubBand band[MAX_DWT_LEVELS][4];
} Plane;

/* Used by Low Delay and High Quality profiles */
typedef struct DiracSlice {
    GetBitContext gb;
    int slice_x;
    int slice_y;
    int bytes;
} DiracSlice;

typedef struct DiracContext {
    AVCodecContext *avctx;
    MpegvideoEncDSPContext mpvencdsp;
    VideoDSPContext vdsp;
    DiracDSPContext diracdsp;
    DiracVersionInfo version;
    GetBitContext gb;
    AVDiracSeqHeader seq;
    enum AVPixelFormat sof_pix_fmt;
    void *hwaccel_picture_private;
    int seen_sequence_header;
    int64_t frame_number;       /* number of the next frame to display       */
    Plane plane[3];
    int chroma_x_shift;
    int chroma_y_shift;

    int bit_depth;              /* bit depth                                 */
    int pshift;                 /* pixel shift = bit_depth > 8               */

    int zero_res;               /* zero residue flag                         */
    int is_arith;               /* whether coeffs use arith or golomb coding */
    int core_syntax;            /* use core syntax only                      */
    int low_delay;              /* use the low delay syntax                  */
    int hq_picture;             /* high quality picture, enables low_delay   */
    int ld_picture;             /* use low delay picture, turns on low_delay */
    int dc_prediction;          /* has dc prediction                         */
    int globalmc_flag;          /* use global motion compensation            */
    int num_refs;               /* number of reference pictures              */

    /* wavelet decoding */
    unsigned wavelet_depth;     /* depth of the IDWT                         */
    unsigned wavelet_idx;

    /**
     * schroedinger older than 1.0.8 doesn't store
     * quant delta if only one codebook exists in a band
     */
    unsigned old_delta_quant;
    unsigned codeblock_mode;

    unsigned num_x;              /* number of horizontal slices               */
    unsigned num_y;              /* number of vertical slices                 */

    uint8_t *thread_buf;         /* Per-thread buffer for coefficient storage */
    int threads_num_buf;         /* Current # of buffers allocated            */
    int thread_buf_size;         /* Each thread has a buffer this size        */

    DiracSlice *slice_params_buf;
    int slice_params_num_buf;

    struct {
        unsigned width;
        unsigned height;
    } codeblock[MAX_DWT_LEVELS+1];

    struct {
        AVRational bytes;       /* average bytes per slice                   */
        uint8_t quant[MAX_DWT_LEVELS][4]; /* [DIRAC_STD] E.1 */
    } lowdelay;

    struct {
        unsigned prefix_bytes;
        uint64_t size_scaler;
    } highquality;

    struct {
        int pan_tilt[2];        /* pan/tilt vector                           */
        int zrs[2][2];          /* zoom/rotate/shear matrix                  */
        int perspective[2];     /* perspective vector                        */
        unsigned zrs_exp;
        unsigned perspective_exp;
    } globalmc[2];

    /* motion compensation */
    uint8_t mv_precision;       /* [DIRAC_STD] REFS_WT_PRECISION             */
    int16_t weight[2];          /* [DIRAC_STD] REF1_WT and REF2_WT           */
    unsigned weight_log2denom;  /* [DIRAC_STD] REFS_WT_PRECISION             */

    int blwidth;                /* number of blocks (horizontally)           */
    int blheight;               /* number of blocks (vertically)             */
    int sbwidth;                /* number of superblocks (horizontally)      */
    int sbheight;               /* number of superblocks (vertically)        */

    uint8_t *sbsplit;
    DiracBlock *blmotion;

    uint8_t *edge_emu_buffer[4];
    uint8_t *edge_emu_buffer_base;

    uint16_t *mctmp;            /* buffer holding the MC data multiplied by OBMC weights */
    uint8_t *mcscratch;
    int buffer_stride;

    DECLARE_ALIGNED(16, uint8_t, obmc_weight)[3][MAX_BLOCKSIZE*MAX_BLOCKSIZE];

    void (*put_pixels_tab[4])(uint8_t *dst, const uint8_t *src[5], int stride, int h);
    void (*avg_pixels_tab[4])(uint8_t *dst, const uint8_t *src[5], int stride, int h);
    void (*add_obmc)(uint16_t *dst, const uint8_t *src, int stride, const uint8_t *obmc_weight, int yblen);
    dirac_weight_func weight_func;
    dirac_biweight_func biweight_func;

    DiracFrame *current_picture;
    DiracFrame *ref_pics[2];

    DiracFrame *ref_frames[MAX_REFERENCE_FRAMES+1];
    DiracFrame *delay_frames[MAX_DELAY+1];
    DiracFrame all_frames[MAX_FRAMES];
} DiracContext;

enum dirac_subband {
    subband_ll = 0,
    subband_hl = 1,
    subband_lh = 2,
    subband_hh = 3,
    subband_nb,
};

typedef struct SliceCoeffs {
    int32_t left;
    int32_t top;
    int32_t tot_h;
    int32_t tot_v;
    int32_t tot;
} SliceCoeffs;

typedef struct SliceCoeffsPushConst {
    int32_t wavelet_depth;
    int32_t slices_num;
} SliceCoeffsPushConst;

typedef struct DiracSliceVkBuf {
    int32_t offs[16];
    SliceCoeffs slices[16];
} DiracSliceVkBuf;

#endif
