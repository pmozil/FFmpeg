/*
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

#include "dirac_dwt.h"
#include "diracdec.h"
#include "libavcodec/dirac_vlc.h"
#include "libavcodec/pthread_internal.h"
#include "libavutil/vulkan_spirv.h"
#include "libavutil/vulkan_loader.h"
#include "libavutil/vulkan.h"
#include "vulkan_decode.h"

typedef struct SubbandOffset {
    int base_off;
    int stride;
    int pad0;
    int pad1;
} SubbandOffset;

typedef struct SliceCoeffVk {
    int left;
    int top;
    int tot_h;
    int tot_v;
    int tot;
    int offs;
    int pad0;
    int pad1;
} SliceCoeffVk;

typedef struct WaveletPushConst {
    int real_plane_dims[6];
    int plane_offs[3];
    int plane_strides[3];
    int dw[3];
    int wavelet_depth;
} WaveletPushConst;

typedef struct DiracVulkanDecodeContext {
    FFVulkanContext vkctx;
    VkSamplerYcbcrConversion yuv_sampler;
    VkSampler sampler;

    FFVulkanShader vert_wavelet[9];
    FFVulkanShader horiz_wavelet[9];
    FFVulkanShader cpy_to_image[3];
    FFVulkanShader quant;

    AVVulkanDeviceQueueFamily *qf;
    FFVkExecPool exec_pool;

    int quant_val_buf_size;
    int thread_buf_size;
    int32_t *quant_val_buf_vk_ptr;
    FFVkBuffer *quant_val_buf;
    AVBufferRef *av_quant_val_buf;
    size_t quant_val_buf_offs;

    int n_slice_bufs;
    int slice_buf_size;
    SliceCoeffVk *slice_buf_vk_ptr;
    FFVkBuffer *quant_buf;
    AVBufferRef *av_quant_buf;
    size_t quant_buf_offs;

    int32_t *quant_buf_vk_ptr;
    int quant_buf_size;
    FFVkBuffer *slice_buf;
    AVBufferRef *av_slice_buf;
    size_t slice_buf_offs;

    FFVkBuffer tmp_buf;
    FFVkBuffer tmp_interleave_buf;

    FFVkBuffer subband_info;
    SubbandOffset *subband_info_ptr;

    int slice_vals_size;

    WaveletPushConst pConst;
} DiracVulkanDecodeContext;

typedef struct DiracVulkanDecodePicture {
    DiracFrame *frame;
} DiracVulkanDecodePicture;

static void free_common(AVCodecContext *avctx) {
    DiracVulkanDecodeContext *dec = avctx->internal->hwaccel_priv_data;
    DiracContext *ctx = avctx->priv_data;
    FFVulkanContext *s = &dec->vkctx;
    FFVulkanFunctions *vk = &dec->vkctx.vkfn;

    if (ctx->hwaccel_picture_private) {
        av_free(ctx->hwaccel_picture_private);
    }

    /* Wait on and free execution pool */
    if (dec->exec_pool.cmd_bufs) {
        ff_vk_exec_pool_free(s, &dec->exec_pool);
    }

    ff_vk_shader_free(s, &dec->quant);

    for (int i = 0; i < 3; i++) {
        ff_vk_shader_free(s, &dec->cpy_to_image[i]);
    }

    for (int i = 0; i < 9; i++) {
        ff_vk_shader_free(s, &dec->vert_wavelet[i]);

        ff_vk_shader_free(s, &dec->horiz_wavelet[i]);
    }
    if (dec->sampler)
        vk->DestroySampler(s->hwctx->act_dev, dec->sampler, s->hwctx->alloc);

    av_buffer_unref(&dec->av_quant_val_buf);
    av_buffer_unref(&dec->av_quant_buf);
    av_buffer_unref(&dec->av_slice_buf);
    av_buffer_unref(&dec->av_slice_buf);

    ff_vk_free_buf(&dec->vkctx, &dec->subband_info);

    ff_vk_free_buf(&dec->vkctx, &dec->tmp_buf);
    ff_vk_free_buf(&dec->vkctx, &dec->tmp_interleave_buf);

    ff_vk_uninit(s);
}

static av_always_inline inline void bar_read(VkBufferMemoryBarrier2 *buf_bar,
                                             int *nb_buf_bar, FFVkBuffer *buf) {
    buf_bar[(*nb_buf_bar)++] = (VkBufferMemoryBarrier2){
        .sType = VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER_2,
        .srcStageMask = VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT,
        .dstStageMask = VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT,
        .srcAccessMask = VK_ACCESS_2_MEMORY_READ_BIT,
        .dstAccessMask = VK_ACCESS_2_MEMORY_READ_BIT,
        .srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
        .dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
        .buffer = buf->buf,
        .size = buf->size,
        .offset = 0,
    };
}

static av_always_inline inline void
bar_write(VkBufferMemoryBarrier2 *buf_bar, int *nb_buf_bar, FFVkBuffer *buf) {
    buf_bar[(*nb_buf_bar)++] = (VkBufferMemoryBarrier2){
        .sType = VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER_2,
        .srcStageMask = VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT,
        .dstStageMask = VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT,
        .srcAccessMask = VK_ACCESS_2_MEMORY_READ_BIT,
        .dstAccessMask = VK_ACCESS_2_MEMORY_WRITE_BIT,
        .srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
        .dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
        .buffer = buf->buf,
        .size = buf->size,
        .offset = 0,
    };
}

static inline int alloc_tmp_bufs(DiracContext *ctx,
                                 DiracVulkanDecodeContext *dec) {
    int err, plane_size;

    plane_size = sizeof(int32_t) *
                 (ctx->plane[0].idwt.width * ctx->plane[0].idwt.height +
                  ctx->plane[1].idwt.width * ctx->plane[1].idwt.height +
                  ctx->plane[2].idwt.width * ctx->plane[2].idwt.height);

    if (dec->tmp_buf.buf != NULL) {
        ff_vk_free_buf(&dec->vkctx, &dec->tmp_buf);
        ff_vk_free_buf(&dec->vkctx, &dec->tmp_interleave_buf);
    }

    err = ff_vk_create_buf(&dec->vkctx, &dec->tmp_buf, plane_size, NULL, NULL,
                           VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT |
                               VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
                           VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
    if (err < 0)
        return err;

    err = ff_vk_create_buf(&dec->vkctx, &dec->tmp_interleave_buf, plane_size,
                           NULL, NULL,
                           VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT |
                               VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
                           VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
    if (err < 0)
        return err;

    return 0;
}

static inline int alloc_host_mapped_buf(DiracVulkanDecodeContext *dec,
                                        size_t req_size, void **mem,
                                        AVBufferRef **avbuf, FFVkBuffer **buf) {
    int err;

    err = ff_vk_create_avbuf(&dec->vkctx, avbuf, req_size, NULL, NULL,
                             VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT |
                                 VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
                             VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT);
    if (err < 0)
        return err;

    *buf = (FFVkBuffer *)(*avbuf)->data;
    err = ff_vk_map_buffer(&dec->vkctx, *buf, (uint8_t **)mem, 0);
    if (err < 0)
        return err;

    return 0;
}

static int alloc_slices_buf(DiracContext *ctx, DiracVulkanDecodeContext *dec) {
    int err, length = ctx->num_y * ctx->num_x;

    dec->n_slice_bufs = length;

    if (dec->slice_buf_vk_ptr) {
        av_buffer_unref(&dec->av_slice_buf);
    }

    dec->slice_buf_size = sizeof(SliceCoeffVk) * length * 3 * MAX_DWT_LEVELS;
    err = alloc_host_mapped_buf(dec, dec->slice_buf_size,
                                (void **)&dec->slice_buf_vk_ptr,
                                &dec->av_slice_buf, &dec->slice_buf);
    if (err < 0)
        return err;

    return 0;
}

static int alloc_dequant_buf(DiracContext *ctx, DiracVulkanDecodeContext *dec) {
    int err, length = ctx->num_y * ctx->num_x;

    if (dec->quant_buf_vk_ptr) {
        av_buffer_unref(&dec->av_quant_buf);
    }

    dec->n_slice_bufs = length;

    dec->quant_buf_size = sizeof(int32_t) * MAX_DWT_LEVELS * 8 * length;
    err = alloc_host_mapped_buf(dec, dec->quant_buf_size,
                                (void **)&dec->quant_buf_vk_ptr,
                                &dec->av_quant_buf, &dec->quant_buf);
    if (err < 0)
        return err;

    return 0;
}

static int subband_coeffs(const DiracContext *s, int x, int y, int p, int off,
                          SliceCoeffVk *c) {
    int level, coef = 0;
    for (level = 0; level <= s->wavelet_depth; level++) {
        SliceCoeffVk *o = &c[level];
        const SubBand *b =
            &s->plane[p].band[level][3]; /* orientation doens't matter */
        o->top = b->height * y / s->num_y;
        o->left = b->width * x / s->num_x;
        o->tot_h = ((b->width * (x + 1)) / s->num_x) - o->left;
        o->tot_v = ((b->height * (y + 1)) / s->num_y) - o->top;
        o->tot = o->tot_h * o->tot_v;
        o->offs = off + coef;
        coef += o->tot * (4 - !!level);
    }
    return coef;
}

static int alloc_quant_buf(DiracContext *ctx, DiracVulkanDecodeContext *dec) {
    int err, length = ctx->num_y * ctx->num_x, coef_buf_size;
    SliceCoeffVk tmp[MAX_DWT_LEVELS];
    coef_buf_size =
        subband_coeffs(ctx, ctx->num_x - 1, ctx->num_y - 1, 0, 0, tmp) + 8;
    coef_buf_size = coef_buf_size + 512 * sizeof(int32_t);
    dec->slice_vals_size = coef_buf_size / sizeof(int32_t);

    if (dec->quant_val_buf_vk_ptr) {
        av_buffer_unref(&dec->av_quant_val_buf);
    }

    dec->thread_buf_size = coef_buf_size;

    dec->quant_val_buf_size = dec->thread_buf_size * 3 * length;
    err = alloc_host_mapped_buf(dec, dec->quant_val_buf_size,
                                (void **)&dec->quant_val_buf_vk_ptr,
                                &dec->av_quant_val_buf, &dec->quant_val_buf);
    if (err < 0)
        return err;

    return 0;
}

extern const char *ff_source_vulkan_dirac_structs_comp;

static int compile_shader(DiracVulkanDecodeContext *s, FFVkSPIRVCompiler *spv,
                            FFVulkanShader *shd, FFVulkanDescriptorSetBinding *desc,
                            const int desc_size, const char *ext[], const int n_ext,
                            const char *shd_name, const char *shader,
                            int dims[3], int push_c_size) {
    int err = 0;
    size_t spv_len = 0;
    uint8_t *spv_data = NULL;
    void *spv_opaque = NULL;
    FFVulkanContext *vkctx = &s->vkctx;

    RET(ff_vk_shader_init(vkctx, shd, shd_name,
                          VK_SHADER_STAGE_COMPUTE_BIT, ext, n_ext,
                          dims[0], dims[1], dims[2],
                          0));

    /* Common codec header */
    GLSLD(ff_source_vulkan_dirac_structs_comp);

    if (push_c_size > 0) {
        RET(ff_vk_shader_add_descriptor_set(vkctx, shd, desc, desc_size, 1, 0));
    }

    ff_vk_shader_add_push_const(shd, 0, push_c_size, VK_SHADER_STAGE_COMPUTE_BIT);

    GLSLD(shader);

    RET(spv->compile_shader(vkctx, spv, shd, &spv_data, &spv_len, "main",
                            &spv_opaque));
    RET(ff_vk_shader_link(vkctx, shd, spv_data, spv_len, "main"));

    RET(ff_vk_shader_register_exec(vkctx, &s->exec_pool, shd));

fail:
    if (spv_opaque)
        spv->free_shader(spv, &spv_opaque);

    return err;
}

/* ----- Copy Shader init and pipeline pass ----- */

extern const char *ff_source_vulkan_dirac_cpy_to_image_8bit_comp;
extern const char *ff_source_vulkan_dirac_cpy_to_image_10bit_comp;
extern const char *ff_source_vulkan_dirac_cpy_to_image_12bit_comp;

static int init_cpy_shd(DiracVulkanDecodeContext *s, FFVkSPIRVCompiler *spv,
                        int idx) {
    int err = 0;
    const int planes = av_pix_fmt_count_planes(s->vkctx.output_format);

    static const char *ext[] = {
        "GL_EXT_debug_printf",
        "GL_EXT_scalar_block_layout",
        "GL_EXT_shader_explicit_arithmetic_types",
    };

    FFVulkanDescriptorSetBinding *desc = (FFVulkanDescriptorSetBinding[]){
        {
            .name = "in_buf",
            .stages = VK_SHADER_STAGE_COMPUTE_BIT,
            .type = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
            .buf_content = "int32_t inBuf[];",
            .mem_quali = "readonly",
            .dimensions = 1,
        },
        {
            .name = "out_img",
            .stages = VK_SHADER_STAGE_COMPUTE_BIT,
            .type = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE,
            .mem_quali = "writeonly",
            .mem_layout = "rgba32f",
            .dimensions = 2,
            .elems = planes,
        },
    };

    int dims[3] = {8, 8, 1};

    const char *shd_txt = NULL;
    if (idx == 0) {
        shd_txt = ff_source_vulkan_dirac_cpy_to_image_8bit_comp;
    } else if (idx == 1) {
        shd_txt = ff_source_vulkan_dirac_cpy_to_image_10bit_comp;
    } else if (idx == 2) {
        shd_txt = ff_source_vulkan_dirac_cpy_to_image_12bit_comp;
    } else {
        return AVERROR_INVALIDDATA;
    }

    err = compile_shader(s, spv, &s->cpy_to_image[idx],
                        desc, 2, ext, 3,
                        "cpy_to_image", (const char *)shd_txt,
                        dims, sizeof(WaveletPushConst));

    return err;
}

static av_always_inline int inline cpy_to_image_pass(
    DiracVulkanDecodeContext *dec, DiracContext *ctx, FFVkExecContext *exec,
    VkImageView *views, VkBufferMemoryBarrier2 *buf_bar, int *nb_buf_bar,
    VkImageMemoryBarrier2 *img_bar, int *nb_img_bar, int idx) {
    int err, prev_nb_bar = *nb_buf_bar, prev_nb_img_bar = *nb_img_bar;
    FFVulkanFunctions *vk = &dec->vkctx.vkfn;
    DiracVulkanDecodePicture *pic = ctx->hwaccel_picture_private;

    err = ff_vk_shader_update_desc_buffer(&dec->vkctx, exec, &dec->cpy_to_image[idx],
                                            0, 0, 0, &dec->tmp_buf, 0,
                                            dec->tmp_buf.size, VK_FORMAT_UNDEFINED);
    if (err < 0)
        return err;

    ff_vk_shader_update_img_array(&dec->vkctx, exec, &dec->cpy_to_image[idx],
                                      pic->frame->avframe, views, 0, 1,
                                      VK_IMAGE_LAYOUT_GENERAL, dec->sampler);

    dec->pConst.real_plane_dims[0] = ctx->plane[0].idwt.width;
    dec->pConst.real_plane_dims[1] = ctx->plane[0].idwt.height;
    dec->pConst.real_plane_dims[2] = ctx->plane[1].idwt.width;
    dec->pConst.real_plane_dims[3] = ctx->plane[1].idwt.height;
    dec->pConst.real_plane_dims[4] = ctx->plane[2].idwt.width;
    dec->pConst.real_plane_dims[5] = ctx->plane[2].idwt.height;

    dec->pConst.plane_strides[0] = ctx->plane[0].idwt.width;
    dec->pConst.plane_strides[1] = ctx->plane[1].idwt.width;
    dec->pConst.plane_strides[2] = ctx->plane[2].idwt.width;

    dec->pConst.plane_offs[0] = 0;
    dec->pConst.plane_offs[1] =
        ctx->plane[0].idwt.width * ctx->plane[0].idwt.height;
    dec->pConst.plane_offs[2] =
        dec->pConst.plane_offs[1] +
        ctx->plane[1].idwt.width * ctx->plane[1].idwt.height;

    ff_vk_shader_update_push_const(&dec->vkctx, exec, &dec->cpy_to_image[idx],
                                    VK_SHADER_STAGE_COMPUTE_BIT, 0,
                                    sizeof(WaveletPushConst), &dec->pConst);

    bar_read(buf_bar, nb_buf_bar, &dec->tmp_interleave_buf);
    bar_write(buf_bar, nb_buf_bar, &dec->tmp_interleave_buf);
    bar_read(buf_bar, nb_buf_bar, &dec->tmp_buf);
    bar_write(buf_bar, nb_buf_bar, &dec->tmp_buf);

    ff_vk_frame_barrier(&dec->vkctx, exec, pic->frame->avframe, img_bar,
                        nb_img_bar, VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT,
                        VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT,
                        VK_ACCESS_SHADER_WRITE_BIT, VK_IMAGE_LAYOUT_GENERAL,
                        VK_QUEUE_FAMILY_IGNORED);

    vk->CmdPipelineBarrier2(
        exec->buf, &(VkDependencyInfo){
                       .sType = VK_STRUCTURE_TYPE_DEPENDENCY_INFO,
                       .pBufferMemoryBarriers = buf_bar + prev_nb_bar,
                       .bufferMemoryBarrierCount = *nb_buf_bar - prev_nb_bar,
                       .pImageMemoryBarriers = img_bar + prev_nb_img_bar,
                       .imageMemoryBarrierCount = *nb_img_bar - prev_nb_img_bar,
                   });

    ff_vk_exec_bind_shader(&dec->vkctx, exec, &dec->cpy_to_image[idx]);

    vk->CmdDispatch(exec->buf, ctx->plane[0].idwt.width >> 3,
                    ctx->plane[0].idwt.height >> 3, 3);

    prev_nb_img_bar = *nb_img_bar;
    ff_vk_frame_barrier(&dec->vkctx, exec, pic->frame->avframe, img_bar,
                        nb_img_bar, VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT,
                        VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT,
                        VK_ACCESS_SHADER_READ_BIT, VK_IMAGE_LAYOUT_GENERAL,
                        VK_QUEUE_FAMILY_IGNORED);

    vk->CmdPipelineBarrier2(
        exec->buf, &(VkDependencyInfo){
                       .sType = VK_STRUCTURE_TYPE_DEPENDENCY_INFO,
                       .pImageMemoryBarriers = img_bar + prev_nb_img_bar,
                       .imageMemoryBarrierCount = *nb_img_bar - prev_nb_img_bar,
                   });

    return 0;
}

// /* ----- LeGall Wavelet init and pipeline pass ----- */

extern const char *ff_source_vulkan_dirac_legall_vert_comp;
extern const char *ff_source_vulkan_dirac_legall_horiz_comp;

static int init_wavelet_shd_legall_vert(DiracVulkanDecodeContext *s,
                                        FFVkSPIRVCompiler *spv) {
    int err = 0;
    int dims[3] = {8, 8, 1};
    static const char *ext[] = {
        "GL_EXT_debug_printf",
        "GL_EXT_scalar_block_layout",
        "GL_EXT_shader_explicit_arithmetic_types",
    };
    FFVulkanDescriptorSetBinding *desc = (FFVulkanDescriptorSetBinding[]){
        {
            .name = "in_buf",
            .stages = VK_SHADER_STAGE_COMPUTE_BIT,
            .type = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
            .buf_content = "int32_t inBuf[];",
            .mem_quali = "readonly",
            .dimensions = 1,
        },
        {
            .name = "out_buf",
            .stages = VK_SHADER_STAGE_COMPUTE_BIT,
            .type = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
            .buf_content = "int32_t outBuf[];",
            .mem_quali = "writeonly",
            .dimensions = 1,
        },
    };

    err = compile_shader(s, spv, &s->vert_wavelet[DWT_DIRAC_LEGALL5_3],
                        desc, 2, ext, 3,
                        "legall", (const char *)ff_source_vulkan_dirac_legall_vert_comp,
                        dims, sizeof(WaveletPushConst));

    return err;
}

static int init_wavelet_shd_legall_horiz(DiracVulkanDecodeContext *s,
                                         FFVkSPIRVCompiler *spv) {
    int err = 0;
    int dims[3] = {8, 8, 1};
    static const char *ext[] = {
        "GL_EXT_debug_printf",
        "GL_EXT_scalar_block_layout",
        "GL_EXT_shader_explicit_arithmetic_types",
    };
    FFVulkanDescriptorSetBinding *desc = (FFVulkanDescriptorSetBinding[]){
        {
            .name = "in_buf",
            .stages = VK_SHADER_STAGE_COMPUTE_BIT,
            .type = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
            .buf_content = "int32_t inBuf[];",
            .mem_quali = "readonly",
            .dimensions = 1,
        },
        {
            .name = "out_buf",
            .stages = VK_SHADER_STAGE_COMPUTE_BIT,
            .type = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
            .buf_content = "int32_t outBuf[];",
            .mem_quali = "writeonly",
            .dimensions = 1,
        },
    };

    err = compile_shader(s, spv, &s->horiz_wavelet[DWT_DIRAC_LEGALL5_3],
                        desc, 2, ext, 3,
                        "legall", (const char *)ff_source_vulkan_dirac_legall_horiz_comp,
                        dims, sizeof(WaveletPushConst));

    return err;
}

static av_always_inline int inline wavelet_legall_pass(
    DiracVulkanDecodeContext *dec, DiracContext *ctx, FFVkExecContext *exec,
    VkBufferMemoryBarrier2 *buf_bar, int *nb_buf_bar) {
    int err;
    int barrier_num = *nb_buf_bar;
    int wavelet_idx = DWT_DIRAC_LEGALL5_3;
    FFVulkanFunctions *vk = &dec->vkctx.vkfn;

    FFVulkanShader *hor = &dec->horiz_wavelet[wavelet_idx];
    FFVulkanShader *vert = &dec->vert_wavelet[wavelet_idx];

    err = ff_vk_shader_update_desc_buffer(&dec->vkctx, exec, vert, 0, 0, 0,
                                            &dec->tmp_buf, 0, dec->tmp_buf.size,
                                            VK_FORMAT_UNDEFINED);
    if (err < 0)
        goto fail;
    err = ff_vk_shader_update_desc_buffer(
        &dec->vkctx, exec, vert, 0, 1, 0, &dec->tmp_interleave_buf, 0,
        dec->tmp_interleave_buf.size, VK_FORMAT_UNDEFINED);
    if (err < 0)
        goto fail;

    err = ff_vk_shader_update_desc_buffer(
        &dec->vkctx, exec, hor, 0, 0, 0, &dec->tmp_interleave_buf, 0,
        dec->tmp_interleave_buf.size, VK_FORMAT_UNDEFINED);
    if (err < 0)
        goto fail;
    err = ff_vk_shader_update_desc_buffer(&dec->vkctx, exec, hor, 0, 1, 0,
                                      &dec->tmp_buf, 0, dec->tmp_buf.size,
                                      VK_FORMAT_UNDEFINED);
    if (err < 0)
        goto fail;

    for (int i = ctx->wavelet_depth - 1; i >= 0; i--) {
        dec->pConst.plane_strides[0] = ctx->plane[0].idwt.width << i;
        dec->pConst.plane_strides[1] = ctx->plane[1].idwt.width << i;
        dec->pConst.plane_strides[2] = ctx->plane[2].idwt.width << i;

        dec->pConst.plane_offs[0] = 0;
        dec->pConst.plane_offs[1] =
            ctx->plane[0].idwt.width * ctx->plane[0].idwt.height;
        dec->pConst.plane_offs[2] =
            dec->pConst.plane_offs[1] +
            ctx->plane[1].idwt.width * ctx->plane[1].idwt.height;

        dec->pConst.dw[0] = ctx->plane[0].idwt.width >> (i + 1);
        dec->pConst.dw[1] = ctx->plane[1].idwt.width >> (i + 1);
        dec->pConst.dw[2] = ctx->plane[2].idwt.width >> (i + 1);

        dec->pConst.real_plane_dims[0] = (ctx->plane[0].idwt.width) >> i;
        dec->pConst.real_plane_dims[1] = (ctx->plane[0].idwt.height) >> i;
        dec->pConst.real_plane_dims[2] = (ctx->plane[1].idwt.width) >> i;
        dec->pConst.real_plane_dims[3] = (ctx->plane[1].idwt.height) >> i;
        dec->pConst.real_plane_dims[4] = (ctx->plane[2].idwt.width) >> i;
        dec->pConst.real_plane_dims[5] = (ctx->plane[2].idwt.height) >> i;

        /* Vertical wavelet pass */
        ff_vk_shader_update_push_const(&dec->vkctx, exec, vert,
                                        VK_SHADER_STAGE_COMPUTE_BIT, 0,
                                        sizeof(WaveletPushConst), &dec->pConst);

        ff_vk_exec_bind_shader(&dec->vkctx, exec, vert);
        barrier_num = *nb_buf_bar;
        bar_read(buf_bar, nb_buf_bar, &dec->tmp_buf);
        bar_write(buf_bar, nb_buf_bar, &dec->tmp_buf);
        bar_read(buf_bar, nb_buf_bar, &dec->tmp_interleave_buf);
        bar_write(buf_bar, nb_buf_bar, &dec->tmp_interleave_buf);

        vk->CmdPipelineBarrier2(
            exec->buf,
            &(VkDependencyInfo){
                .sType = VK_STRUCTURE_TYPE_DEPENDENCY_INFO,
                .pBufferMemoryBarriers = buf_bar + barrier_num,
                .bufferMemoryBarrierCount = *nb_buf_bar - barrier_num,
            });

        vk->CmdDispatch(exec->buf, dec->pConst.real_plane_dims[0] >> 3,
                        dec->pConst.real_plane_dims[1] >> 4, 3);

        /* Horizontal wavelet pass */
        ff_vk_shader_update_push_const(&dec->vkctx, exec, hor,
                                        VK_SHADER_STAGE_COMPUTE_BIT, 0,
                                        sizeof(WaveletPushConst), &dec->pConst);

        ff_vk_exec_bind_shader(&dec->vkctx, exec, hor);
        barrier_num = *nb_buf_bar;
        bar_read(buf_bar, nb_buf_bar, &dec->tmp_buf);
        bar_write(buf_bar, nb_buf_bar, &dec->tmp_buf);
        bar_read(buf_bar, nb_buf_bar, &dec->tmp_interleave_buf);
        bar_write(buf_bar, nb_buf_bar, &dec->tmp_interleave_buf);

        vk->CmdPipelineBarrier2(
            exec->buf,
            &(VkDependencyInfo){
                .sType = VK_STRUCTURE_TYPE_DEPENDENCY_INFO,
                .pBufferMemoryBarriers = buf_bar + barrier_num,
                .bufferMemoryBarrierCount = *nb_buf_bar - barrier_num,
            });

        vk->CmdDispatch(exec->buf, dec->pConst.real_plane_dims[0] >> 4,
                        dec->pConst.real_plane_dims[1] >> 3, 3);
    }

    barrier_num = *nb_buf_bar;
    bar_read(buf_bar, nb_buf_bar, &dec->tmp_buf);
    bar_write(buf_bar, nb_buf_bar, &dec->tmp_buf);
    bar_read(buf_bar, nb_buf_bar, &dec->tmp_interleave_buf);
    bar_write(buf_bar, nb_buf_bar, &dec->tmp_interleave_buf);

    vk->CmdPipelineBarrier2(
        exec->buf, &(VkDependencyInfo){
                       .sType = VK_STRUCTURE_TYPE_DEPENDENCY_INFO,
                       .pBufferMemoryBarriers = buf_bar + barrier_num,
                       .bufferMemoryBarrierCount = *nb_buf_bar - barrier_num,
                   });

    return 0;
fail:
    ff_vk_exec_discard_deps(&dec->vkctx, exec);
    return err;
}

// /* ----- Fidelity init and pipeline pass ----- */
//
// static const char fidelity_low[] = {C(
//     0, int32_t fidelity_low(int32_t v0, int32_t v1, int32_t v2, int32_t v3,
//                             int32_t v4, int32_t v5, int32_t v6, int32_t v7) {)
//       C(1,      return  (-2 * v0 + 10 * v1 - 25 * v2 + 81 * v3 + 81 * v4 - 25 * v5 + 10 * v6 - 2 * v7 + 128) >> 8;)
//       C(0,
//     })};
//
// static const char fidelity_high[] = {C(
//     0, int32_t fidelity_high(int32_t v0, int32_t v1, int32_t v2, int32_t v3,
//                              int32_t v4, int32_t v5, int32_t v6, int32_t v7) {)
//       C(1,      return  (-8 * v0 + 21 * v1 - 46 * v2 + 161 * v3 + 161 * v4 - 46 * v5 + 21 * v6 - 8 * v7 + 128) >> 8;)
//       C(0,
//     })};
//
// static const char fidelity_low_y[] = {C(
//     0, int32_t fidelity_low_y(int plane, int x, int y) {                  )
//     C(1,    const int h = plane_sizes[plane].y;                             )
//     C(1,                                                                    )
//     C(1,    const int32_t v1 = inBuf[getIdx(plane, x, y + 1)];              )
//     C(1,                                                                    )
//     C(1,    const int y_6 = ((y - 6) > 0) ? (y - 6) : 0;                    )
//     C(1,    const int32_t v_6 = inBuf[getIdx(plane, x, y_6)];               )
//     C(1,                                                                    )
//     C(1,    const int y_4 = ((y - 4) > 0) ? (y - 4) : 0;                    )
//     C(1,    const int32_t v_4 = inBuf[getIdx(plane, x, y_4)];               )
//     C(1,                                                                    )
//     C(1,    const int y_2 = ((y - 2) > 0) ? (y - 2) : 0;                    )
//     C(1,    const int32_t v_2 = inBuf[getIdx(plane, x, y_2)];               )
//     C(1,                                                                    )
//     C(1,    const int32_t v0 = inBuf[getIdx(plane, x, y)];                  )
//     C(1,                                                                    )
//     C(1,    const int y2 = ((y + 2) < h) ? (y + 2) : (h - 2);               )
//     C(1,    const int32_t v2 = inBuf[getIdx(plane, x, y2)];                 )
//     C(1,                                                                    )
//     C(1,    const int y4 = ((y + 4) < h) ? (y + 4) : (h - 2);               )
//     C(1,    const int32_t v4 = inBuf[getIdx(plane, x, y4)];                 )
//     C(1,                                                                    )
//     C(1,    const int y6 = ((y + 6) < h) ? (y + 6) : (h - 2);               )
//     C(1,    const int32_t v6 = inBuf[getIdx(plane, x, y6)];                 )
//     C(1,                                                                    )
//     C(1,    const int y8 = ((y + 8) < h) ? (y + 8) : (h - 2);               )
//     C(1,    const int32_t v8 = inBuf[getIdx(plane, x, y8)];                 )
//     C(1,                                                                    )
//     C(1,    return v1 + fidelity_low(v_6, v_4, v_2, v0, v2, v4, v6, v8);    )
//     C(0,
//     })};
//
// static const char fidelity_vert[] = {C(
//     0, void idwt_vert(int plane, int x, int y) {                                              )
//     C(1,    const int h = plane_sizes[plane].y;                                                 )
//     C(1,                                                                                        )
//     C(1,    const int32_t v0 = inBuf[getIdx(plane, x, y)];                                      )
//     C(1,    const int32_t v1 = fidelity_low_y(plane, x, y);                                     )
//     C(1,    const int32_t v_7 = (y - 8 > 0) ? fidelity_low_y(plane, x, y - 8) : v1;             )
//     C(1,    const int32_t v_5 = (y - 6 > 0) ? fidelity_low_y(plane, x, y - 6) : v1;             )
//     C(1,    const int32_t v_3 = (y - 4 > 0) ? fidelity_low_y(plane, x, y - 4) : v1;             )
//     C(1,    const int32_t v_1 = (y - 2 > 0) ? fidelity_low_y(plane, x, y - 2) : v1;             )
//     C(1,    const int32_t v3  = (y + 2 < h) ? fidelity_low_y(plane, x, y + 2) :                 )
//     C(1,                                        fidelity_low_y(plane, x, h - 2);                )
//     C(1,    const int32_t v5  = (y + 4 < h) ? fidelity_low_y(plane, x, y + 4) :                 )
//     C(1,                                        fidelity_low_y(plane, x, h - 2);                )
//     C(1,    const int32_t v7  = (y + 6 < h) ? fidelity_low_y(plane, x, y + 6) :                 )
//     C(1,                                        fidelity_low_y(plane, x, h - 2);                )
//     C(1,    outBuf[getIdx(plane, x, y)] = v0 - fidelity_high(v_7, v_5, v_3, v_1, v1, v3, v5, v7);)
//     C(1,    outBuf[getIdx(plane, x, y + 1)] = v1;                                               )
//     C(0,
//     })};
//
// static const char fidelity_low_x[] = {C(
//     0, int32_t fidelity_low_x(int plane, int x, int y) {                  )
//     C(1,    const int w = plane_sizes[plane].x;                             )
//     C(1,    const int dw = w / 2 - 1;                                       )
//     C(1,                                                                    )
//     C(1,    const int x_3 = clamp(x - 3, 0, dw);                            )
//     C(1,    const int32_t v_3 = inBuf[getIdx(plane, x_3, y)];               )
//     C(1,                                                                    )
//     C(1,    const int x_2 = clamp(x - 2, 0, dw);                            )
//     C(1,    const int32_t v_2 = inBuf[getIdx(plane, x_2, y)];               )
//     C(1,                                                                    )
//     C(1,    const int x_1 = clamp(x - 1, 0, dw);                            )
//     C(1,    const int32_t v_1 = inBuf[getIdx(plane, x_1, y)];               )
//     C(1,                                                                    )
//     C(1,    const int32_t v0 = inBuf[getIdx(plane, x, y)];                  )
//     C(1,                                                                    )
//     C(1,    const int x_w = x + dw + 1;                                     )
//     C(1,    const int32_t v_w = inBuf[getIdx(plane, x_w, y)];               )
//     C(1,                                                                    )
//     C(1,    const int x1 = clamp(x + 1, 0, dw);                             )
//     C(1,    const int32_t v1 = inBuf[getIdx(plane, x1, y)];                 )
//     C(1,                                                                    )
//     C(1,    const int x2 = clamp(x + 2, 0, dw);                             )
//     C(1,    const int32_t v2 = inBuf[getIdx(plane, x2, y)];                 )
//     C(1,                                                                    )
//     C(1,    const int x3 = clamp(x + 3, 0, dw);                             )
//     C(1,    const int32_t v3 = inBuf[getIdx(plane, x3, y)];                 )
//     C(1,                                                                    )
//     C(1,    const int x4 = clamp(x + 4, 0, dw);                             )
//     C(1,    const int32_t v4 = inBuf[getIdx(plane, x4, y)];                 )
//     C(1,                                                                    )
//     C(1,    return v_w + fidelity_low(v_3, v_2, v_1, v0, v1, v2, v3, v4);   )
//     C(0,
//     })};
//
// static const char fidelity_horiz[] = {C(
//     0, void idwt_horiz(int plane, int x, int y) {                                             )
//     C(1,    const int w = plane_sizes[plane].x;                                                 )
//     C(1,    const int dw = w / 2 - 1;                                                           )
//     C(1,                                                                                        )
//     C(1,    const int32_t vo0 = inBuf[getIdx(plane, x, y)];                                     )
//     C(1,                                                                                        )
//     C(1,    const int x_4 = clamp(x - 4, 0, dw);                                                )
//     C(1,    const int32_t v_4 = fidelity_low_x(plane, x_4, y);                                  )
//     C(1,    const int x_3 = clamp(x - 3, 0, dw);                                                )
//     C(1,    const int32_t v_3 = fidelity_low_x(plane, x_3, y);                                  )
//     C(1,    const int x_2 = clamp(x - 2, 0, dw);                                                )
//     C(1,    const int32_t v_2 = fidelity_low_x(plane, x_2, y);                                  )
//     C(1,    const int x_1 = clamp(x - 1, 0, dw);                                                )
//     C(1,    const int32_t v_1 = fidelity_low_x(plane, x_1, y);                                  )
//     C(1,    const int x0 = clamp(x, 0, dw);                                                     )
//     C(1,    const int32_t v0 = fidelity_low_x(plane, x0, y);                                    )
//     C(1,    const int x1 = clamp(x + 1, 0, dw);                                                 )
//     C(1,    const int32_t v1 = fidelity_low_x(plane, x1, y);                                    )
//     C(1,    const int x2 = clamp(x + 2, 0, dw);                                                 )
//     C(1,    const int32_t v2 = fidelity_low_x(plane, x2, y);                                    )
//     C(1,    const int x3 = clamp(x + 3, 0, dw);                                                 )
//     C(1,    const int32_t v3 = fidelity_low_x(plane, x3, y);                                    )
//     C(1,                                                                                        )
//     C(1,    outBuf[getIdx(plane, 2 * x, y)] = vo0 - fidelity_high(v_4, v_3, v_2, v_1, v0, v1, v2, v3);)
//     C(1,    outBuf[getIdx(plane, 2 * x + 1, y)] = v0;                                           )
//     C(0,
//     })};
//
// static int init_wavelet_shd_fidelity_vert(DiracVulkanDecodeContext *s,
//                                           FFVkSPIRVCompiler *spv) {
//     int err = 0;
//     uint8_t *spv_data;
//     size_t spv_len;
//     void *spv_opaque = NULL;
//     int wavelet_idx = DWT_DIRAC_FIDELITY;
//     FFVulkanContext *vkctx = &s->vkctx;
//     FFVulkanDescriptorSetBinding *desc;
//     FFVkSPIRVShader *shd = &s->vert_wavelet_shd[wavelet_idx];
//     FFVulkanPipeline *pl = &s->vert_wavelet_pl[wavelet_idx];
//     FFVkExecPool *exec = &s->exec_pool;
//
//     RET(ff_vk_shader_init(pl, shd, "fidelity_vert", VK_SHADER_STAGE_COMPUTE_BIT,
//                           0));
//
//     shd = &s->vert_wavelet_shd[wavelet_idx];
//     ff_vk_shader_set_compute_sizes(shd, 8, 8, 1);
//
//     GLSLC(0, #extension GL_EXT_scalar_block_layout : require);
//     GLSLC(0, #extension GL_EXT_shader_explicit_arithmetic_types : require);
//
//     desc = (FFVulkanDescriptorSetBinding[]){
//         {
//             .name = "in_buf",
//             .stages = VK_SHADER_STAGE_COMPUTE_BIT,
//             .type = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
//             .buf_content = "int32_t inBuf[];",
//             .mem_quali = "readonly",
//             .dimensions = 1,
//         },
//         {
//             .name = "out_buf",
//             .stages = VK_SHADER_STAGE_COMPUTE_BIT,
//             .type = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
//             .buf_content = "int32_t outBuf[];",
//             .mem_quali = "writeonly",
//             .dimensions = 1,
//         },
//     };
//     RET(ff_vk_pipeline_descriptor_set_add(vkctx, pl, shd, desc, 2, 0, 0));
//
//     ff_vk_add_push_constant(pl, 0, sizeof(WaveletPushConst),
//                             VK_SHADER_STAGE_COMPUTE_BIT);
//
//     GLSLC(
//         0, layout(push_constant, std430) uniform pushConstants {  );
//             GLSLC(1, ivec2 plane_sizes[3];);
//             GLSLC(1, int plane_offs[3];);
//             GLSLC(1, int plane_strides[3];);
//             GLSLC(1, int dw[3];);
//             GLSLC(1, int wavelet_depth;);
//     GLSLC(0,
//         };);
//     GLSLC(0, );
//
//     GLSLD(get_idx);
//     GLSLD(fidelity_low);
//     GLSLD(fidelity_high);
//     GLSLD(fidelity_low_y);
//     GLSLD(fidelity_vert);
//
//     GLSLC(
//         0, void main() {                                                              );
//             GLSLC(1, int off_y = int(gl_WorkGroupSize.y * gl_NumWorkGroups.y););
//             GLSLC(1, int off_x = int(gl_WorkGroupSize.x * gl_NumWorkGroups.x););
//             GLSLC(1, int pic_z = int(gl_GlobalInvocationID.z););
//             GLSLC(1, );
//             GLSLC(1, uint h = int(plane_sizes[pic_z].y););
//             GLSLC(2, uint w = int(plane_sizes[pic_z].x););
//             GLSLC(1, );
//             GLSLC(1, int y = int(gl_GlobalInvocationID.y););
//             GLSLC(
//                 1, for (; 2 * y < h; y += off_y) {                                         );
//                     GLSLC(2, int x = int(gl_GlobalInvocationID.x););
//                     GLSLC(
//                         2, for (; x < w; x += off_x) {                                         );
//                             GLSLC(3, idwt_vert(pic_z, x, 2 * y););
//     GLSLC(2,
//                         });
//     GLSLC(1,
//                 });
//     GLSLC(0,
//         });
//
//     RET(spv->compile_shader(spv, vkctx, shd, &spv_data, &spv_len, "main",
//                             &spv_opaque));
//     RET(ff_vk_shader_create(vkctx, shd, spv_data, spv_len, "main"));
//     RET(ff_vk_init_compute_pipeline(vkctx, pl, shd));
//
//     RET(ff_vk_exec_pipeline_register(vkctx, exec, pl));
//
// fail:
//     if (spv_opaque)
//         spv->free_shader(spv, &spv_opaque);
//
//     return err;
// }
//
// static int init_wavelet_shd_fidelity_horiz(DiracVulkanDecodeContext *s,
//                                            FFVkSPIRVCompiler *spv) {
//     int err = 0;
//     uint8_t *spv_data;
//     size_t spv_len;
//     void *spv_opaque = NULL;
//     int wavelet_idx = DWT_DIRAC_FIDELITY;
//     FFVulkanContext *vkctx = &s->vkctx;
//     FFVulkanDescriptorSetBinding *desc;
//     FFVkSPIRVShader *shd = &s->horiz_wavelet_shd[wavelet_idx];
//     FFVulkanPipeline *pl = &s->horiz_wavelet_pl[wavelet_idx];
//     FFVkExecPool *exec = &s->exec_pool;
//
//     RET(ff_vk_shader_init(pl, shd, "fidelity_horiz",
//                           VK_SHADER_STAGE_COMPUTE_BIT, 0));
//
//     shd = &s->horiz_wavelet_shd[wavelet_idx];
//     ff_vk_shader_set_compute_sizes(shd, 8, 8, 1);
//
//     GLSLC(0, #extension GL_EXT_debug_printf : enable);
//     GLSLC(0, #extension GL_EXT_scalar_block_layout : require);
//     GLSLC(0, #extension GL_EXT_shader_explicit_arithmetic_types : require);
//
//     desc = (FFVulkanDescriptorSetBinding[]){
//         {
//             .name = "in_buf",
//             .stages = VK_SHADER_STAGE_COMPUTE_BIT,
//             .type = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
//             .buf_content = "int32_t inBuf[];",
//             .mem_quali = "readonly",
//             .dimensions = 1,
//         },
//         {
//             .name = "out_buf",
//             .stages = VK_SHADER_STAGE_COMPUTE_BIT,
//             .type = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
//             .buf_content = "int32_t outBuf[];",
//             .mem_quali = "writeonly",
//             .dimensions = 1,
//         },
//     };
//     RET(ff_vk_pipeline_descriptor_set_add(vkctx, pl, shd, desc, 2, 0, 0));
//
//     ff_vk_add_push_constant(pl, 0, sizeof(WaveletPushConst),
//                             VK_SHADER_STAGE_COMPUTE_BIT);
//
//     GLSLC(
//         0, layout(push_constant, std430) uniform pushConstants {  );
//             GLSLC(1, ivec2 plane_sizes[3];);
//             GLSLC(1, int plane_offs[3];);
//             GLSLC(1, int plane_strides[3];);
//             GLSLC(1, int dw[3];);
//             GLSLC(1, int wavelet_depth;);
//     GLSLC(0,
//         };);
//     GLSLC(0, );
//
//     GLSLD(get_idx);
//     GLSLD(fidelity_low);
//     GLSLD(fidelity_high);
//     GLSLD(fidelity_low_x);
//     GLSLD(fidelity_horiz);
//
//     GLSLC(
//         0, void main() {                                                              );
//             GLSLC(1, int off_y = int(gl_WorkGroupSize.y * gl_NumWorkGroups.y););
//             GLSLC(1, int off_x = int(gl_WorkGroupSize.x * gl_NumWorkGroups.x););
//             GLSLC(1, int pic_z = int(gl_GlobalInvocationID.z););
//             GLSLC(1, );
//             GLSLC(1, uint h = int(plane_sizes[pic_z].y););
//             GLSLC(2, uint w = int(plane_sizes[pic_z].x););
//             GLSLC(1, );
//             GLSLC(1, int y = int(gl_GlobalInvocationID.y););
//             GLSLC(
//                 1, for (; y < h; y += off_y) {                                             );
//                     GLSLC(2, int x = int(gl_GlobalInvocationID.x););
//                     GLSLC(
//                         2, for (; 2 * x < w; x += off_x) {                                     );
//                             GLSLC(3, idwt_horiz(pic_z, x, y););
//     GLSLC(2,
//                         });
//     GLSLC(1,
//                 });
//     GLSLC(0,
//         });
//
//     RET(spv->compile_shader(spv, vkctx, shd, &spv_data, &spv_len, "main",
//                             &spv_opaque));
//     RET(ff_vk_shader_create(vkctx, shd, spv_data, spv_len, "main"));
//     RET(ff_vk_init_compute_pipeline(vkctx, pl, shd));
//     RET(ff_vk_exec_pipeline_register(vkctx, exec, pl));
//
// fail:
//     if (spv_opaque)
//         spv->free_shader(spv, &spv_opaque);
//
//     return err;
// }
//
// static av_always_inline int inline wavelet_fidelity_pass(
//     DiracVulkanDecodeContext *dec, DiracContext *ctx, FFVkExecContext *exec,
//     VkBufferMemoryBarrier2 *buf_bar, int *nb_buf_bar) {
//     int err;
//     int barrier_num = *nb_buf_bar;
//     int wavelet_idx = DWT_DIRAC_FIDELITY;
//     FFVulkanFunctions *vk = &dec->vkctx.vkfn;
//
//     FFVulkanPipeline *pl_hor = &dec->horiz_wavelet_pl[wavelet_idx];
//     FFVulkanPipeline *pl_vert = &dec->vert_wavelet_pl[wavelet_idx];
//
//     err = ff_vk_set_descriptor_buffer(&dec->vkctx, pl_vert, exec, 0, 0, 0,
//                                       &dec->tmp_buf, 0, dec->tmp_buf.size,
//                                       VK_FORMAT_UNDEFINED);
//     if (err < 0)
//         goto fail;
//     err = ff_vk_set_descriptor_buffer(
//         &dec->vkctx, pl_vert, exec, 0, 1, 0, &dec->tmp_interleave_buf, 0,
//         dec->tmp_interleave_buf.size, VK_FORMAT_UNDEFINED);
//     if (err < 0)
//         goto fail;
//
//     err = ff_vk_set_descriptor_buffer(
//         &dec->vkctx, pl_hor, exec, 0, 0, 0, &dec->tmp_interleave_buf, 0,
//         dec->tmp_interleave_buf.size, VK_FORMAT_UNDEFINED);
//     if (err < 0)
//         goto fail;
//     err = ff_vk_set_descriptor_buffer(&dec->vkctx, pl_hor, exec, 0, 1, 0,
//                                       &dec->tmp_buf, 0, dec->tmp_buf.size,
//                                       VK_FORMAT_UNDEFINED);
//     if (err < 0)
//         goto fail;
//
//     for (int i = ctx->wavelet_depth - 1; i >= 0; i--) {
//         dec->pConst.plane_strides[0] = ctx->plane[0].idwt.width << i;
//         dec->pConst.plane_strides[1] = ctx->plane[1].idwt.width << i;
//         dec->pConst.plane_strides[2] = ctx->plane[2].idwt.width << i;
//
//         dec->pConst.plane_offs[0] = 0;
//         dec->pConst.plane_offs[1] =
//             ctx->plane[0].idwt.width * ctx->plane[0].idwt.height;
//         dec->pConst.plane_offs[2] =
//             dec->pConst.plane_offs[1] +
//             ctx->plane[1].idwt.width * ctx->plane[1].idwt.height;
//
//         dec->pConst.dw[0] = ctx->plane[0].idwt.width >> (i + 1);
//         dec->pConst.dw[1] = ctx->plane[1].idwt.width >> (i + 1);
//         dec->pConst.dw[2] = ctx->plane[2].idwt.width >> (i + 1);
//
//         dec->pConst.real_plane_dims[0] = (ctx->plane[0].idwt.width) >> i;
//         dec->pConst.real_plane_dims[1] = (ctx->plane[0].idwt.height) >> i;
//         dec->pConst.real_plane_dims[2] = (ctx->plane[1].idwt.width) >> i;
//         dec->pConst.real_plane_dims[3] = (ctx->plane[1].idwt.height) >> i;
//         dec->pConst.real_plane_dims[4] = (ctx->plane[2].idwt.width) >> i;
//         dec->pConst.real_plane_dims[5] = (ctx->plane[2].idwt.height) >> i;
//
//         /* Vertical wavelet pass */
//         ff_vk_update_push_exec(&dec->vkctx, exec, pl_vert,
//                                VK_SHADER_STAGE_COMPUTE_BIT, 0,
//                                sizeof(WaveletPushConst), &dec->pConst);
//
//         barrier_num = *nb_buf_bar;
//         bar_read(buf_bar, nb_buf_bar, &dec->tmp_buf);
//         bar_write(buf_bar, nb_buf_bar, &dec->tmp_buf);
//         bar_read(buf_bar, nb_buf_bar, &dec->tmp_interleave_buf);
//         bar_write(buf_bar, nb_buf_bar, &dec->tmp_interleave_buf);
//
//         vk->CmdPipelineBarrier2(
//             exec->buf,
//             &(VkDependencyInfo){
//                 .sType = VK_STRUCTURE_TYPE_DEPENDENCY_INFO,
//                 .pBufferMemoryBarriers = buf_bar + barrier_num,
//                 .bufferMemoryBarrierCount = *nb_buf_bar - barrier_num,
//             });
//
//         ff_vk_exec_bind_pipeline(&dec->vkctx, exec, pl_vert);
//         vk->CmdDispatch(exec->buf, dec->pConst.real_plane_dims[0] >> 3,
//                         dec->pConst.real_plane_dims[1] >> 4, 3);
//
//         /* Horizontal wavelet pass */
//         ff_vk_update_push_exec(&dec->vkctx, exec, pl_hor,
//                                VK_SHADER_STAGE_COMPUTE_BIT, 0,
//                                sizeof(WaveletPushConst), &dec->pConst);
//
//         barrier_num = *nb_buf_bar;
//         bar_read(buf_bar, nb_buf_bar, &dec->tmp_buf);
//         bar_write(buf_bar, nb_buf_bar, &dec->tmp_buf);
//         bar_read(buf_bar, nb_buf_bar, &dec->tmp_interleave_buf);
//         bar_write(buf_bar, nb_buf_bar, &dec->tmp_interleave_buf);
//
//         vk->CmdPipelineBarrier2(
//             exec->buf,
//             &(VkDependencyInfo){
//                 .sType = VK_STRUCTURE_TYPE_DEPENDENCY_INFO,
//                 .pBufferMemoryBarriers = buf_bar + barrier_num,
//                 .bufferMemoryBarrierCount = *nb_buf_bar - barrier_num,
//             });
//
//         ff_vk_exec_bind_pipeline(&dec->vkctx, exec, pl_hor);
//         vk->CmdDispatch(exec->buf, dec->pConst.real_plane_dims[0] >> 4,
//                         dec->pConst.real_plane_dims[1] >> 3, 3);
//     }
//
//     barrier_num = *nb_buf_bar;
//     bar_read(buf_bar, nb_buf_bar, &dec->tmp_buf);
//     bar_write(buf_bar, nb_buf_bar, &dec->tmp_buf);
//     bar_read(buf_bar, nb_buf_bar, &dec->tmp_interleave_buf);
//     bar_write(buf_bar, nb_buf_bar, &dec->tmp_interleave_buf);
//
//     vk->CmdPipelineBarrier2(
//         exec->buf, &(VkDependencyInfo){
//                        .sType = VK_STRUCTURE_TYPE_DEPENDENCY_INFO,
//                        .pBufferMemoryBarriers = buf_bar + barrier_num,
//                        .bufferMemoryBarrierCount = *nb_buf_bar - barrier_num,
//                    });
//
//     return 0;
// fail:
//     ff_vk_exec_discard_deps(&dec->vkctx, exec);
//     return err;
// }
//
// /* ----- Daubechies(9, 7) init and pipeline pass ----- */
//
// static const char daub97_low1[] = {C(
//     0, int32_t daub97_low1(int32_t v1, int32_t v2, int32_t v3) {    )
//       C(1,      return  v2 - ((1817 * (v1 + v2) + 2048) >> 12);         )
//       C(0,
//     })};
//
// static const char daub97_high1[] = {C(
//     0, int32_t daub97_high1(int32_t v1, int32_t v2, int32_t v3) {   )
//       C(1,      return  v2 - ((113 * (v1 + v2) + 64) >> 7);             )
//       C(0,
//     })};
//
// static const char daub97_low0[] = {C(
//     0, int32_t daub97_low0(int32_t v1, int32_t v2, int32_t v3) {    )
//       C(1,      return  v2 - ((217 * (v1 + v2) + 2048) >> 12);          )
//       C(0,
//     })};
//
// static const char daub97_high0[] = {C(
//     0, int32_t daub97_high0(int32_t v1, int32_t v2, int32_t v3) {   )
//       C(1,      return  v2 - ((6947 * (v1 + v2) + 2048) >> 12);         )
//       C(0,
//     })};
//
// static const char daub97_low_x0[] = {C(
//     0, int32_t daub97_low_x0(int plane, int x, int y) {         )
//       C(1,      int w = plane_sizes[plane].x;                       )
//       C(1,      int dw = plane_sizes[plane].x / 2;                  )
//       C(1,                                                          )
//       C(1,      int x0 = (x == 0) ? dw : x + dw;                    )
//       C(1,      int32_t v0 = inBuf[getIdx(plane, x0, y)];           )
//       C(1,                                                          )
//       C(1,      int32_t v1 = inBuf[getIdx(plane, x, y)];            )
//       C(1,                                                          )
//       C(1,      int x2 = x + dw;                                    )
//       C(1,      int32_t v2 = inBuf[getIdx(plane, x0, y)];           )
//       C(1,                                                          )
//       C(1,      return daub97_low1(v0, v1, v2);                     )
//       C(0,
//     })};
//
// static const char daub97_high_x0[] = {C(
//     0, int32_t daub97_high_x0(int plane, int x, int y) {        )
//       C(1,      int w = plane_sizes[plane].x;                       )
//       C(1,      int dw = plane_sizes[plane].x / 2;                  )
//       C(1,                                                          )
//       C(1,      int x0 = (x == dw - 1) ? (dw - 1) : (x - 1);        )
//       C(1,      int32_t v0 = daub97_low_x0(plane, x0, y);           )
//       C(1,                                                          )
//       C(1,      int32_t v1 = inBuf[getIdx(plane, x + dw - 1, y)];   )
//       C(1,                                                          )
//       C(1,      int32_t v2 = daub97_low_x0(plane, x, y);            )
//       C(1,                                                          )
//       C(1,      return daub97_high1(v0, v1, v2);                    )
//       C(0,
//     })};
//
// static const char daub97_low_x1[] = {C(
//     0, int32_t daub97_low_x1(int plane, int x, int y) {         )
//       C(1,      int w = plane_sizes[plane].x;                       )
//       C(1,      int dw = plane_sizes[plane].x / 2;                  )
//       C(1,                                                          )
//       C(1,      int32_t v0 = daub97_high_x0(plane, x, y);           )
//       C(1,                                                          )
//       C(1,      int32_t v1 = daub97_low_x0(plane, x, y);            )
//       C(1,                                                          )
//       C(1,      int32_t v2 = daub97_high_x0(plane, x + 1, y);       )
//       C(1,                                                          )
//       C(1,      return daub97_low0(v0, v1, v2);                     )
//       C(0,
//     })};
//
// static const char daub97_high_x1[] = {C(
//     0, int32_t daub97_high_x1(int plane, int x, int y) {        )
//       C(1,      int w = plane_sizes[plane].x;                       )
//       C(1,      int dw = plane_sizes[plane].x / 2;                  )
//       C(1,                                                          )
//       C(1,      int x0 = clamp(x - 1, 0, dw);                       )
//       C(1,      int32_t v0 = daub97_low_x1(plane, x0, y);           )
//       C(1,                                                          )
//       C(1,      int32_t v1 = daub97_high_x0(plane, x + 1, y);       )
//       C(1,                                                          )
//       C(1,      int x2 = clamp(x, 0, dw);                           )
//       C(1,      int32_t v2 = daub97_low_x1(plane, x2, y);           )
//       C(1,                                                          )
//       C(1,      return daub97_high0(v0, v1, v2);                    )
//       C(0,
//     })};
//
// static const char daub97_horiz[] = {C(
//     0, void idwt_horiz(int plane, int x, int y) {                   )
//       C(1,      int w = plane_sizes[plane].x;                           )
//       C(1,      int dw = plane_sizes[plane].x / 2;                      )
//       C(1,                                                              )
//       C(1,      int32_t v0 = daub97_low_x1(plane, x, y);                )
//       C(1,      int32_t v1 = daub97_high_x1(plane, x, y);               )
//       C(1,                                                              )
//       C(1,    outBuf[getIdx(plane, 2 * x, y)]     = ~((~v0) >> 1);      )
//       C(1,    outBuf[getIdx(plane, 2 * x + 1, y)] = ~((~v1) >> 1);      )
//       C(0,
//     })};
//
// static int init_wavelet_shd_daub97_vert(DiracVulkanDecodeContext *s,
//                                         FFVkSPIRVCompiler *spv) {
//     int err = 0;
//     uint8_t *spv_data;
//     size_t spv_len;
//     void *spv_opaque = NULL;
//     int wavelet_idx = DWT_DIRAC_DAUB9_7;
//     FFVulkanContext *vkctx = &s->vkctx;
//     FFVulkanDescriptorSetBinding *desc;
//     FFVkSPIRVShader *shd = &s->vert_wavelet_shd[wavelet_idx];
//     FFVulkanPipeline *pl = &s->vert_wavelet_pl[wavelet_idx];
//     FFVkExecPool *exec = &s->exec_pool;
//
//     RET(ff_vk_shader_init(pl, shd, "daub97_vert", VK_SHADER_STAGE_COMPUTE_BIT,
//                           0));
//
//     shd = &s->vert_wavelet_shd[wavelet_idx];
//     ff_vk_shader_set_compute_sizes(shd, 8, 1, 1);
//
//     GLSLC(0, #extension GL_EXT_debug_printf : enable);
//     GLSLC(0, #extension GL_EXT_scalar_block_layout : require);
//     GLSLC(0, #extension GL_EXT_shader_explicit_arithmetic_types : require);
//
//     desc = (FFVulkanDescriptorSetBinding[]){
//         {
//             .name = "in_buf",
//             .stages = VK_SHADER_STAGE_COMPUTE_BIT,
//             .type = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
//             .buf_content = "int32_t inBuf[];",
//             .dimensions = 1,
//         },
//         {
//             .name = "out_buf",
//             .stages = VK_SHADER_STAGE_COMPUTE_BIT,
//             .type = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
//             .buf_content = "int32_t outBuf[];",
//             .mem_quali = "writeonly",
//             .dimensions = 1,
//         },
//     };
//     RET(ff_vk_pipeline_descriptor_set_add(vkctx, pl, shd, desc, 2, 0, 0));
//
//     ff_vk_add_push_constant(pl, 0, sizeof(WaveletPushConst),
//                             VK_SHADER_STAGE_COMPUTE_BIT);
//
//     GLSLC(
//         0, layout(push_constant, std430) uniform pushConstants {  );
//             GLSLC(1, ivec2 plane_sizes[3];);
//             GLSLC(1, int plane_offs[3];);
//             GLSLC(1, int plane_strides[3];);
//             GLSLC(1, int dw[3];);
//             GLSLC(1, int wavelet_depth;);
//     GLSLC(0,
//         };);
//     GLSLC(0, );
//
//     GLSLD(get_idx);
//
//     GLSLC(
//         0, void main() {                                                                  );
//             GLSLC(1, int off_x = int(gl_WorkGroupSize.x * gl_NumWorkGroups.x););
//             GLSLC(1, int pic_z = int(gl_GlobalInvocationID.z););
//             GLSLC(1, );
//             GLSLC(1, uint h = int(plane_sizes[pic_z].y););
//             GLSLC(1, uint w = int(plane_sizes[pic_z].x););
//             GLSLC(1, );
//             GLSLC(2, int x = int(gl_GlobalInvocationID.x););
//             GLSLC(
//                 1, for (; x < w; x += off_x) {                                                 );
//                     GLSLC(
//                         2, for (int y = 0; y < h; y += 2) {                                        );
//                             GLSLC(3, int32_t v0 = inBuf[getIdx(
//                                          pic_z, x, int(clamp(y - 1, 0, h)))];);
//                             GLSLC(3,
//                                   int32_t v1 = inBuf[getIdx(pic_z, x, y + 1)];);
//                             GLSLC(3, inBuf[getIdx(pic_z, x, y)] -=
//                                      (1817 * (v0 + v1 + 2048)) >> 12;);
//     GLSLC(2,
//                         });
//                     GLSLC(
//                         2, for (int y = 0; y < h; y += 2) {                                        );
//                             GLSLC(3, int32_t v0 = inBuf[getIdx(pic_z, x, y)];);
//                             GLSLC(3,
//                                   int32_t v1 = inBuf[getIdx(
//                                       pic_z, x, int(clamp(y + 2, 0, h - 2)))];);
//                             GLSLC(3, inBuf[getIdx(pic_z, x, y + 1)] -=
//                                      (3616 * (v0 + v1 + 2048)) >> 12;);
//     GLSLC(2,
//                         });
//                     GLSLC(
//                         2, for (int y = 0; y < h; y += 2) {                                        );
//                             GLSLC(3, int32_t v0 = inBuf[getIdx(
//                                          pic_z, x, int(clamp(y - 1, 0, h)))];);
//                             GLSLC(3,
//                                   int32_t v1 = inBuf[getIdx(pic_z, x, y + 1)];);
//                             GLSLC(3, int32_t v2 = inBuf[getIdx(pic_z, x, y)];);
//                             GLSLC(3, outBuf[getIdx(pic_z, x, y)] =
//                                          v2 + (217 * (v0 + v1 + 2048)) >> 12;);
//     GLSLC(2,
//                         });
//                     GLSLC(
//                         2, for (int y = 0; y < h; y += 2) {                                        );
//                             GLSLC(3, int32_t v0 = inBuf[getIdx(pic_z, x, y)];);
//                             GLSLC(3,
//                                   int32_t v1 = inBuf[getIdx(
//                                       pic_z, x, int(clamp(y + 2, 0, h - 2)))];);
//                             GLSLC(3,
//                                   int32_t v2 = inBuf[getIdx(pic_z, x, y + 1)];);
//                             GLSLC(3, outBuf[getIdx(pic_z, x, y + 1)] =
//                                          v2 + (6497 * (v0 + v1 + 2048)) >> 12;);
//     GLSLC(2,
//                         });
//     GLSLC(1,
//                 });
//     GLSLC(0,
//         });
//
//     RET(spv->compile_shader(spv, vkctx, shd, &spv_data, &spv_len, "main",
//                             &spv_opaque));
//     RET(ff_vk_shader_create(vkctx, shd, spv_data, spv_len, "main"));
//     RET(ff_vk_init_compute_pipeline(vkctx, pl, shd));
//     RET(ff_vk_exec_pipeline_register(vkctx, exec, pl));
//
// fail:
//     if (spv_opaque)
//         spv->free_shader(spv, &spv_opaque);
//
//     return err;
// }
//
// static int init_wavelet_shd_daub97_horiz(DiracVulkanDecodeContext *s,
//                                          FFVkSPIRVCompiler *spv) {
//     int err = 0;
//     uint8_t *spv_data;
//     size_t spv_len;
//     void *spv_opaque = NULL;
//     int wavelet_idx = DWT_DIRAC_DAUB9_7;
//     FFVulkanContext *vkctx = &s->vkctx;
//     FFVulkanDescriptorSetBinding *desc;
//     FFVkSPIRVShader *shd = &s->horiz_wavelet_shd[wavelet_idx];
//     FFVulkanPipeline *pl = &s->horiz_wavelet_pl[wavelet_idx];
//     FFVkExecPool *exec = &s->exec_pool;
//
//     RET(ff_vk_shader_init(pl, shd, "daub97_horiz", VK_SHADER_STAGE_COMPUTE_BIT,
//                           0));
//
//     shd = &s->horiz_wavelet_shd[wavelet_idx];
//     ff_vk_shader_set_compute_sizes(shd, 8, 8, 1);
//
//     GLSLC(0, #extension GL_EXT_debug_printf : enable);
//     GLSLC(0, #extension GL_EXT_scalar_block_layout : require);
//     GLSLC(0, #extension GL_EXT_shader_explicit_arithmetic_types : require);
//
//     desc = (FFVulkanDescriptorSetBinding[]){
//         {
//             .name = "in_buf",
//             .stages = VK_SHADER_STAGE_COMPUTE_BIT,
//             .type = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
//             .buf_content = "int32_t inBuf[];",
//             .mem_quali = "readonly",
//             .dimensions = 1,
//         },
//         {
//             .name = "out_buf",
//             .stages = VK_SHADER_STAGE_COMPUTE_BIT,
//             .type = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
//             .buf_content = "int32_t outBuf[];",
//             .mem_quali = "writeonly",
//             .dimensions = 1,
//         },
//     };
//     RET(ff_vk_pipeline_descriptor_set_add(vkctx, pl, shd, desc, 2, 0, 0));
//
//     ff_vk_add_push_constant(pl, 0, sizeof(WaveletPushConst),
//                             VK_SHADER_STAGE_COMPUTE_BIT);
//
//     GLSLC(
//         0, layout(push_constant, std430) uniform pushConstants {  );
//             GLSLC(1, ivec2 plane_sizes[3];);
//             GLSLC(1, int plane_offs[3];);
//             GLSLC(1, int plane_strides[3];);
//             GLSLC(1, int dw[3];);
//             GLSLC(1, int wavelet_depth;);
//     GLSLC(0,
//         };);
//     GLSLC(0, );
//
//     GLSLD(get_idx);
//     GLSLD(daub97_low1);
//     GLSLD(daub97_low0);
//     GLSLD(daub97_high1);
//     GLSLD(daub97_high0);
//     GLSLD(daub97_low_x0);
//     GLSLD(daub97_high_x0);
//     GLSLD(daub97_low_x1);
//     GLSLD(daub97_high_x1);
//     GLSLD(daub97_horiz);
//
//     GLSLC(
//         0, void main() {                                                              );
//             GLSLC(1, int off_y = int(gl_WorkGroupSize.y * gl_NumWorkGroups.y););
//             GLSLC(1, int off_x = int(gl_WorkGroupSize.x * gl_NumWorkGroups.x););
//             GLSLC(1, int pic_z = int(gl_GlobalInvocationID.z););
//             GLSLC(1, );
//             GLSLC(1, uint h = int(plane_sizes[pic_z].y););
//             GLSLC(2, uint w = int(plane_sizes[pic_z].x););
//             GLSLC(1, );
//             GLSLC(1, int y = int(gl_GlobalInvocationID.y););
//             GLSLC(
//                 1, for (; y < h; y += off_y) {                                             );
//                     GLSLC(2, int x = int(gl_GlobalInvocationID.x););
//                     GLSLC(
//                         2, for (; 2 * x < w; x += off_x) {                                     );
//                             GLSLC(3, idwt_horiz(pic_z, x, y););
//     GLSLC(2,
//                         });
//     GLSLC(1,
//                 });
//     GLSLC(0,
//         });
//
//     RET(spv->compile_shader(spv, vkctx, shd, &spv_data, &spv_len, "main",
//                             &spv_opaque));
//     RET(ff_vk_shader_create(vkctx, shd, spv_data, spv_len, "main"));
//     RET(ff_vk_init_compute_pipeline(vkctx, pl, shd));
//     RET(ff_vk_exec_pipeline_register(vkctx, exec, pl));
//
// fail:
//     if (spv_opaque)
//         spv->free_shader(spv, &spv_opaque);
//
//     return err;
// }
//
// static av_always_inline int inline wavelet_daub97_pass(
//     DiracVulkanDecodeContext *dec, DiracContext *ctx, FFVkExecContext *exec,
//     VkBufferMemoryBarrier2 *buf_bar, int *nb_buf_bar) {
//     int err;
//     int barrier_num = *nb_buf_bar;
//     int wavelet_idx = DWT_DIRAC_DAUB9_7;
//     FFVulkanFunctions *vk = &dec->vkctx.vkfn;
//
//     FFVulkanPipeline *pl_hor = &dec->horiz_wavelet_pl[wavelet_idx];
//     FFVulkanPipeline *pl_vert = &dec->vert_wavelet_pl[wavelet_idx];
//
//     err = ff_vk_set_descriptor_buffer(&dec->vkctx, pl_vert, exec, 0, 0, 0,
//                                       &dec->tmp_buf, 0, dec->tmp_buf.size,
//                                       VK_FORMAT_UNDEFINED);
//     if (err < 0)
//         goto fail;
//     err = ff_vk_set_descriptor_buffer(
//         &dec->vkctx, pl_vert, exec, 0, 1, 0, &dec->tmp_interleave_buf, 0,
//         dec->tmp_interleave_buf.size, VK_FORMAT_UNDEFINED);
//     if (err < 0)
//         goto fail;
//
//     err = ff_vk_set_descriptor_buffer(
//         &dec->vkctx, pl_hor, exec, 0, 0, 0, &dec->tmp_interleave_buf, 0,
//         dec->tmp_interleave_buf.size, VK_FORMAT_UNDEFINED);
//     if (err < 0)
//         goto fail;
//     err = ff_vk_set_descriptor_buffer(&dec->vkctx, pl_hor, exec, 0, 1, 0,
//                                       &dec->tmp_buf, 0, dec->tmp_buf.size,
//                                       VK_FORMAT_UNDEFINED);
//     if (err < 0)
//         goto fail;
//
//     for (int i = ctx->wavelet_depth - 1; i >= 0; i--) {
//         dec->pConst.plane_strides[0] = ctx->plane[0].idwt.width << i;
//         dec->pConst.plane_strides[1] = ctx->plane[1].idwt.width << i;
//         dec->pConst.plane_strides[2] = ctx->plane[2].idwt.width << i;
//
//         dec->pConst.plane_offs[0] = 0;
//         dec->pConst.plane_offs[1] =
//             ctx->plane[0].idwt.width * ctx->plane[0].idwt.height;
//         dec->pConst.plane_offs[2] =
//             dec->pConst.plane_offs[1] +
//             ctx->plane[1].idwt.width * ctx->plane[1].idwt.height;
//
//         dec->pConst.dw[0] = ctx->plane[0].idwt.width >> (i + 1);
//         dec->pConst.dw[1] = ctx->plane[1].idwt.width >> (i + 1);
//         dec->pConst.dw[2] = ctx->plane[2].idwt.width >> (i + 1);
//
//         dec->pConst.real_plane_dims[0] = (ctx->plane[0].idwt.width) >> i;
//         dec->pConst.real_plane_dims[1] = (ctx->plane[0].idwt.height) >> i;
//         dec->pConst.real_plane_dims[2] = (ctx->plane[1].idwt.width) >> i;
//         dec->pConst.real_plane_dims[3] = (ctx->plane[1].idwt.height) >> i;
//         dec->pConst.real_plane_dims[4] = (ctx->plane[2].idwt.width) >> i;
//         dec->pConst.real_plane_dims[5] = (ctx->plane[2].idwt.height) >> i;
//
//         /* Vertical wavelet pass */
//         ff_vk_update_push_exec(&dec->vkctx, exec, pl_vert,
//                                VK_SHADER_STAGE_COMPUTE_BIT, 0,
//                                sizeof(WaveletPushConst), &dec->pConst);
//
//         barrier_num = *nb_buf_bar;
//         bar_read(buf_bar, nb_buf_bar, &dec->tmp_buf);
//         bar_write(buf_bar, nb_buf_bar, &dec->tmp_buf);
//         bar_read(buf_bar, nb_buf_bar, &dec->tmp_interleave_buf);
//         bar_write(buf_bar, nb_buf_bar, &dec->tmp_interleave_buf);
//
//         vk->CmdPipelineBarrier2(
//             exec->buf,
//             &(VkDependencyInfo){
//                 .sType = VK_STRUCTURE_TYPE_DEPENDENCY_INFO,
//                 .pBufferMemoryBarriers = buf_bar + barrier_num,
//                 .bufferMemoryBarrierCount = *nb_buf_bar - barrier_num,
//             });
//
//         ff_vk_exec_bind_pipeline(&dec->vkctx, exec, pl_vert);
//         vk->CmdDispatch(exec->buf, dec->pConst.real_plane_dims[0], 1, 3);
//
//         /* Horizontal wavelet pass */
//         ff_vk_update_push_exec(&dec->vkctx, exec, pl_hor,
//                                VK_SHADER_STAGE_COMPUTE_BIT, 0,
//                                sizeof(WaveletPushConst), &dec->pConst);
//
//         barrier_num = *nb_buf_bar;
//         bar_read(buf_bar, nb_buf_bar, &dec->tmp_buf);
//         bar_write(buf_bar, nb_buf_bar, &dec->tmp_buf);
//         bar_read(buf_bar, nb_buf_bar, &dec->tmp_interleave_buf);
//         bar_write(buf_bar, nb_buf_bar, &dec->tmp_interleave_buf);
//
//         vk->CmdPipelineBarrier2(
//             exec->buf,
//             &(VkDependencyInfo){
//                 .sType = VK_STRUCTURE_TYPE_DEPENDENCY_INFO,
//                 .pBufferMemoryBarriers = buf_bar + barrier_num,
//                 .bufferMemoryBarrierCount = *nb_buf_bar - barrier_num,
//             });
//
//         ff_vk_exec_bind_pipeline(&dec->vkctx, exec, pl_hor);
//         vk->CmdDispatch(exec->buf, dec->pConst.real_plane_dims[0] >> 4,
//                         dec->pConst.real_plane_dims[1] >> 3, 3);
//
//         barrier_num = *nb_buf_bar;
//         bar_read(buf_bar, nb_buf_bar, &dec->tmp_buf);
//         bar_write(buf_bar, nb_buf_bar, &dec->tmp_buf);
//         bar_read(buf_bar, nb_buf_bar, &dec->tmp_interleave_buf);
//         bar_write(buf_bar, nb_buf_bar, &dec->tmp_interleave_buf);
//
//         vk->CmdPipelineBarrier2(
//             exec->buf,
//             &(VkDependencyInfo){
//                 .sType = VK_STRUCTURE_TYPE_DEPENDENCY_INFO,
//                 .pBufferMemoryBarriers = buf_bar + barrier_num,
//                 .bufferMemoryBarrierCount = *nb_buf_bar - barrier_num,
//             });
//     }
//
//     return 0;
// fail:
//     ff_vk_exec_discard_deps(&dec->vkctx, exec);
//     return err;
// }
//
// /* ----- Deslauriers-Dubuc(9, 7) init and pipeline pass ----- */
//
// static const char dd97_high[] = {C(
//     0, int32_t dd97_high(int32_t v1, int32_t v2, int32_t v3, int32_t v4,
//                          int32_t v5) {  )
//       C(1,      return  v3 + ((9 * v4 + 9 * v2 - v5 - v1 + 8) >> 4);                        )
//       C(0,
//     })};
//
// static const char dd97_vert[] = {C(
//     0, void idwt_vert(int plane, int x, int y) {                                          )
//     C(1,    const int h = plane_sizes[plane].y;                                             )
//     C(1,                                                                                    )
//     C(1,    const int32_t out0 = legall_low_y(plane, x, y);                                 )
//     C(1,    const int32_t out_2 = (y - 2 > 0) ? legall_low_y(plane, x, y - 2) :             )
//     C(1,                                        legall_low_y(plane, x, 0);                  )
//     C(1,    const int32_t out2 = (y + 2 < h) ? legall_low_y(plane, x, y + 2) :              )
//     C(1,                                        legall_low_y(plane, x, h - 2);              )
//     C(1,    const int32_t out4 = (y + 4 < h) ? legall_low_y(plane, x, y + 4) :              )
//     C(1,                                        legall_low_y(plane, x, h - 2);              )
//     C(1,    const int32_t val1 = inBuf[getIdx(plane, x, y + 1)];                            )
//     C(1,                                                                                    )
//     C(1,    outBuf[getIdx(plane, x, y)]     = out0;                                         )
//     C(1,    outBuf[getIdx(plane, x, y + 1)] = dd97_high(out_2, out0, val1, out2, out4);     )
//     C(1,
//     })};
//
// static const char dd97_horiz[] = {C(
//     0, void idwt_horiz(int plane, int x, int y) {                                         )
//     C(1,    const int w = plane_sizes[plane].x;                                             )
//     C(1,    const int dw = w / 2 - 1;                                                       )
//     C(1,                                                                                    )
//     C(1,    const int32_t out0 = legall_low_x(plane, x, y);                                 )
//     C(1,                                                                                    )
//     C(1,    const int32_t out_1 = ((x - 1) > 0) ? legall_low_x(plane, x - 1, y) : out0;     )
//     C(1,    const int32_t val3 = inBuf[getIdx(plane, x + dw + 1, y)];                       )
//     C(1,    const int32_t out1 = ((x + 1) <= dw) ? legall_low_x(plane, x + 1, y) :          )
//     C(1,                                            legall_low_x(plane, dw, y);             )
//     C(1,    const int32_t out2 = ((x + 2) <= dw) ? legall_low_x(plane, x + 2, y) :          )
//     C(1,                                            legall_low_x(plane, dw, y);             )
//     C(1,    const int32_t res  = dd97_high(out_1, out0, val3, out1, out2);                  )
//     C(1,                                                                                    )
//     C(1,    outBuf[getIdx(plane, 2 * x, y)]     = (out0 + 1) >> 1;                          )
//     C(1,    outBuf[getIdx(plane, 2 * x + 1, y)] = (res + 1) >> 1;                           )
//     C(0,
//     })};
//
// static int init_wavelet_shd_dd97_vert(DiracVulkanDecodeContext *s,
//                                       FFVkSPIRVCompiler *spv) {
//     int err = 0;
//     uint8_t *spv_data;
//     size_t spv_len;
//     void *spv_opaque = NULL;
//     int wavelet_idx = DWT_DIRAC_DD9_7;
//     FFVulkanContext *vkctx = &s->vkctx;
//     FFVulkanDescriptorSetBinding *desc;
//     FFVkSPIRVShader *shd = &s->vert_wavelet_shd[wavelet_idx];
//     FFVulkanPipeline *pl = &s->vert_wavelet_pl[wavelet_idx];
//     FFVkExecPool *exec = &s->exec_pool;
//
//     RET(ff_vk_shader_init(pl, shd, "dd97_vert", VK_SHADER_STAGE_COMPUTE_BIT,
//                           0));
//
//     shd = &s->vert_wavelet_shd[wavelet_idx];
//     ff_vk_shader_set_compute_sizes(shd, 8, 8, 1);
//
//     GLSLC(0, #extension GL_EXT_scalar_block_layout : require);
//     GLSLC(0, #extension GL_EXT_shader_explicit_arithmetic_types : require);
//
//     desc = (FFVulkanDescriptorSetBinding[]){
//         {
//             .name = "in_buf",
//             .stages = VK_SHADER_STAGE_COMPUTE_BIT,
//             .type = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
//             .buf_content = "int32_t inBuf[];",
//             .mem_quali = "readonly",
//             .dimensions = 1,
//         },
//         {
//             .name = "out_buf",
//             .stages = VK_SHADER_STAGE_COMPUTE_BIT,
//             .type = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
//             .buf_content = "int32_t outBuf[];",
//             .mem_quali = "writeonly",
//             .dimensions = 1,
//         },
//     };
//     RET(ff_vk_pipeline_descriptor_set_add(vkctx, pl, shd, desc, 2, 0, 0));
//
//     ff_vk_add_push_constant(pl, 0, sizeof(WaveletPushConst),
//                             VK_SHADER_STAGE_COMPUTE_BIT);
//
//     GLSLC(
//         0, layout(push_constant, std430) uniform pushConstants {  );
//             GLSLC(1, ivec2 plane_sizes[3];);
//             GLSLC(1, int plane_offs[3];);
//             GLSLC(1, int plane_strides[3];);
//             GLSLC(1, int dw[3];);
//             GLSLC(1, int wavelet_depth;);
//     GLSLC(0,
//         };);
//     GLSLC(0, );
//
//     GLSLD(get_idx);
//     GLSLD(legall_low_y);
//     GLSLD(dd97_high);
//     GLSLD(dd97_vert);
//
//     GLSLC(
//         0, void main() {                                                              );
//             GLSLC(1, int off_y = int(gl_WorkGroupSize.y * gl_NumWorkGroups.y););
//             GLSLC(1, int off_x = int(gl_WorkGroupSize.x * gl_NumWorkGroups.x););
//             GLSLC(1, int pic_z = int(gl_GlobalInvocationID.z););
//             GLSLC(1, );
//             GLSLC(1, uint h = int(plane_sizes[pic_z].y););
//             GLSLC(2, uint w = int(plane_sizes[pic_z].x););
//             GLSLC(1, );
//             GLSLC(1, int y = int(gl_GlobalInvocationID.y););
//             GLSLC(
//                 1, for (; 2 * y < h; y += off_y) {                                         );
//                     GLSLC(2, int x = int(gl_GlobalInvocationID.x););
//                     GLSLC(
//                         2, for (; x < w; x += off_x) {                                         );
//                             GLSLC(3, idwt_vert(pic_z, x, 2 * y););
//     GLSLC(2,
//                         });
//     GLSLC(1,
//                 });
//     GLSLC(0,
//         });
//
//     RET(spv->compile_shader(spv, vkctx, shd, &spv_data, &spv_len, "main",
//                             &spv_opaque));
//     RET(ff_vk_shader_create(vkctx, shd, spv_data, spv_len, "main"));
//     RET(ff_vk_init_compute_pipeline(vkctx, pl, shd));
//     RET(ff_vk_exec_pipeline_register(vkctx, exec, pl));
//
// fail:
//     if (spv_opaque)
//         spv->free_shader(spv, &spv_opaque);
//
//     return err;
// }
//
// static int init_wavelet_shd_dd97_horiz(DiracVulkanDecodeContext *s,
//                                        FFVkSPIRVCompiler *spv) {
//     int err = 0;
//     uint8_t *spv_data;
//     size_t spv_len;
//     void *spv_opaque = NULL;
//     int wavelet_idx = DWT_DIRAC_DD9_7;
//     FFVulkanContext *vkctx = &s->vkctx;
//     FFVulkanDescriptorSetBinding *desc;
//     FFVkSPIRVShader *shd = &s->horiz_wavelet_shd[wavelet_idx];
//     FFVulkanPipeline *pl = &s->horiz_wavelet_pl[wavelet_idx];
//     FFVkExecPool *exec = &s->exec_pool;
//
//     RET(ff_vk_shader_init(pl, shd, "dd97_horiz", VK_SHADER_STAGE_COMPUTE_BIT,
//                           0));
//
//     shd = &s->horiz_wavelet_shd[wavelet_idx];
//     ff_vk_shader_set_compute_sizes(shd, 8, 8, 1);
//
//     GLSLC(0, #extension GL_EXT_debug_printf : enable);
//     GLSLC(0, #extension GL_EXT_scalar_block_layout : require);
//     GLSLC(0, #extension GL_EXT_shader_explicit_arithmetic_types : require);
//
//     desc = (FFVulkanDescriptorSetBinding[]){
//         {
//             .name = "in_buf",
//             .stages = VK_SHADER_STAGE_COMPUTE_BIT,
//             .type = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
//             .buf_content = "int32_t inBuf[];",
//             .mem_quali = "readonly",
//             .dimensions = 1,
//         },
//         {
//             .name = "out_buf",
//             .stages = VK_SHADER_STAGE_COMPUTE_BIT,
//             .type = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
//             .buf_content = "int32_t outBuf[];",
//             .mem_quali = "writeonly",
//             .dimensions = 1,
//         },
//     };
//     RET(ff_vk_pipeline_descriptor_set_add(vkctx, pl, shd, desc, 2, 0, 0));
//
//     ff_vk_add_push_constant(pl, 0, sizeof(WaveletPushConst),
//                             VK_SHADER_STAGE_COMPUTE_BIT);
//
//     GLSLC(
//         0, layout(push_constant, std430) uniform pushConstants {  );
//             GLSLC(1, ivec2 plane_sizes[3];);
//             GLSLC(1, int plane_offs[3];);
//             GLSLC(1, int plane_strides[3];);
//             GLSLC(1, int dw[3];);
//             GLSLC(1, int wavelet_depth;);
//     GLSLC(0,
//         };);
//     GLSLC(0, );
//
//     GLSLD(get_idx);
//     GLSLD(legall_low_x);
//     GLSLD(dd97_high);
//     GLSLD(dd97_horiz);
//
//     GLSLC(
//         0, void main() {                                                              );
//             GLSLC(1, int off_y = int(gl_WorkGroupSize.y * gl_NumWorkGroups.y););
//             GLSLC(1, int off_x = int(gl_WorkGroupSize.x * gl_NumWorkGroups.x););
//             GLSLC(1, int pic_z = int(gl_GlobalInvocationID.z););
//             GLSLC(1, );
//             GLSLC(1, uint h = int(plane_sizes[pic_z].y););
//             GLSLC(2, uint w = int(plane_sizes[pic_z].x););
//             GLSLC(1, );
//             GLSLC(1, int y = int(gl_GlobalInvocationID.y););
//             GLSLC(
//                 1, for (; y < h; y += off_y) {                                             );
//                     GLSLC(2, int x = int(gl_GlobalInvocationID.x););
//                     GLSLC(
//                         2, for (; 2 * x < w; x += off_x) {                                     );
//                             GLSLC(3, idwt_horiz(pic_z, x, y););
//     GLSLC(2,
//                         });
//     GLSLC(1,
//                 });
//     GLSLC(0,
//         });
//
//     RET(spv->compile_shader(spv, vkctx, shd, &spv_data, &spv_len, "main",
//                             &spv_opaque));
//     RET(ff_vk_shader_create(vkctx, shd, spv_data, spv_len, "main"));
//     RET(ff_vk_init_compute_pipeline(vkctx, pl, shd));
//     RET(ff_vk_exec_pipeline_register(vkctx, exec, pl));
//
// fail:
//     if (spv_opaque)
//         spv->free_shader(spv, &spv_opaque);
//
//     return err;
// }
//
// static av_always_inline int inline wavelet_dd97_pass(
//     DiracVulkanDecodeContext *dec, DiracContext *ctx, FFVkExecContext *exec,
//     VkBufferMemoryBarrier2 *buf_bar, int *nb_buf_bar) {
//     int err;
//     int barrier_num = *nb_buf_bar;
//     int wavelet_idx = DWT_DIRAC_DD9_7;
//     FFVulkanFunctions *vk = &dec->vkctx.vkfn;
//
//     FFVulkanPipeline *pl_hor = &dec->horiz_wavelet_pl[wavelet_idx];
//     FFVulkanPipeline *pl_vert = &dec->vert_wavelet_pl[wavelet_idx];
//
//     err = ff_vk_set_descriptor_buffer(&dec->vkctx, pl_vert, exec, 0, 0, 0,
//                                       &dec->tmp_buf, 0, dec->tmp_buf.size,
//                                       VK_FORMAT_UNDEFINED);
//     if (err < 0)
//         goto fail;
//     err = ff_vk_set_descriptor_buffer(
//         &dec->vkctx, pl_vert, exec, 0, 1, 0, &dec->tmp_interleave_buf, 0,
//         dec->tmp_interleave_buf.size, VK_FORMAT_UNDEFINED);
//     if (err < 0)
//         goto fail;
//
//     err = ff_vk_set_descriptor_buffer(
//         &dec->vkctx, pl_hor, exec, 0, 0, 0, &dec->tmp_interleave_buf, 0,
//         dec->tmp_interleave_buf.size, VK_FORMAT_UNDEFINED);
//     if (err < 0)
//         goto fail;
//     err = ff_vk_set_descriptor_buffer(&dec->vkctx, pl_hor, exec, 0, 1, 0,
//                                       &dec->tmp_buf, 0, dec->tmp_buf.size,
//                                       VK_FORMAT_UNDEFINED);
//     if (err < 0)
//         goto fail;
//
//     for (int i = ctx->wavelet_depth - 1; i >= 0; i--) {
//         dec->pConst.plane_strides[0] = ctx->plane[0].idwt.width << i;
//         dec->pConst.plane_strides[1] = ctx->plane[1].idwt.width << i;
//         dec->pConst.plane_strides[2] = ctx->plane[2].idwt.width << i;
//
//         dec->pConst.plane_offs[0] = 0;
//         dec->pConst.plane_offs[1] =
//             ctx->plane[0].idwt.width * ctx->plane[0].idwt.height;
//         dec->pConst.plane_offs[2] =
//             dec->pConst.plane_offs[1] +
//             ctx->plane[1].idwt.width * ctx->plane[1].idwt.height;
//
//         dec->pConst.dw[0] = ctx->plane[0].idwt.width >> (i + 1);
//         dec->pConst.dw[1] = ctx->plane[1].idwt.width >> (i + 1);
//         dec->pConst.dw[2] = ctx->plane[2].idwt.width >> (i + 1);
//
//         dec->pConst.real_plane_dims[0] = (ctx->plane[0].idwt.width) >> i;
//         dec->pConst.real_plane_dims[1] = (ctx->plane[0].idwt.height) >> i;
//         dec->pConst.real_plane_dims[2] = (ctx->plane[1].idwt.width) >> i;
//         dec->pConst.real_plane_dims[3] = (ctx->plane[1].idwt.height) >> i;
//         dec->pConst.real_plane_dims[4] = (ctx->plane[2].idwt.width) >> i;
//         dec->pConst.real_plane_dims[5] = (ctx->plane[2].idwt.height) >> i;
//
//         /* Vertical wavelet pass */
//         ff_vk_update_push_exec(&dec->vkctx, exec, pl_vert,
//                                VK_SHADER_STAGE_COMPUTE_BIT, 0,
//                                sizeof(WaveletPushConst), &dec->pConst);
//
//         barrier_num = *nb_buf_bar;
//         bar_read(buf_bar, nb_buf_bar, &dec->tmp_buf);
//         bar_write(buf_bar, nb_buf_bar, &dec->tmp_buf);
//         bar_read(buf_bar, nb_buf_bar, &dec->tmp_interleave_buf);
//         bar_write(buf_bar, nb_buf_bar, &dec->tmp_interleave_buf);
//
//         vk->CmdPipelineBarrier2(
//             exec->buf,
//             &(VkDependencyInfo){
//                 .sType = VK_STRUCTURE_TYPE_DEPENDENCY_INFO,
//                 .pBufferMemoryBarriers = buf_bar + barrier_num,
//                 .bufferMemoryBarrierCount = *nb_buf_bar - barrier_num,
//             });
//
//         ff_vk_exec_bind_pipeline(&dec->vkctx, exec, pl_vert);
//         vk->CmdDispatch(exec->buf, dec->pConst.real_plane_dims[0] >> 3,
//                         dec->pConst.real_plane_dims[1] >> 4, 3);
//
//         /* Horizontal wavelet pass */
//         ff_vk_update_push_exec(&dec->vkctx, exec, pl_hor,
//                                VK_SHADER_STAGE_COMPUTE_BIT, 0,
//                                sizeof(WaveletPushConst), &dec->pConst);
//
//         barrier_num = *nb_buf_bar;
//         bar_read(buf_bar, nb_buf_bar, &dec->tmp_buf);
//         bar_write(buf_bar, nb_buf_bar, &dec->tmp_buf);
//         bar_read(buf_bar, nb_buf_bar, &dec->tmp_interleave_buf);
//         bar_write(buf_bar, nb_buf_bar, &dec->tmp_interleave_buf);
//
//         vk->CmdPipelineBarrier2(
//             exec->buf,
//             &(VkDependencyInfo){
//                 .sType = VK_STRUCTURE_TYPE_DEPENDENCY_INFO,
//                 .pBufferMemoryBarriers = buf_bar + barrier_num,
//                 .bufferMemoryBarrierCount = *nb_buf_bar - barrier_num,
//             });
//
//         ff_vk_exec_bind_pipeline(&dec->vkctx, exec, pl_hor);
//         vk->CmdDispatch(exec->buf, dec->pConst.real_plane_dims[0] >> 4,
//                         dec->pConst.real_plane_dims[1] >> 3, 3);
//
//         barrier_num = *nb_buf_bar;
//         bar_read(buf_bar, nb_buf_bar, &dec->tmp_buf);
//         bar_write(buf_bar, nb_buf_bar, &dec->tmp_buf);
//         bar_read(buf_bar, nb_buf_bar, &dec->tmp_interleave_buf);
//         bar_write(buf_bar, nb_buf_bar, &dec->tmp_interleave_buf);
//
//         vk->CmdPipelineBarrier2(
//             exec->buf,
//             &(VkDependencyInfo){
//                 .sType = VK_STRUCTURE_TYPE_DEPENDENCY_INFO,
//                 .pBufferMemoryBarriers = buf_bar + barrier_num,
//                 .bufferMemoryBarrierCount = *nb_buf_bar - barrier_num,
//             });
//     }
//
//     return 0;
// fail:
//     ff_vk_exec_discard_deps(&dec->vkctx, exec);
//     return err;
// }
//
// /* ----- Deslauriers-Dubuc(13, 7) init and pipeline pass ----- */
// static const char dd137_low[] = {C(
//     0, int32_t dd137_low(int32_t v0, int32_t v1, int32_t v2, int32_t v3,
//                          int32_t v4) {    )
//     C(0,    return v2 - ((9 * v1 + 9 * v3 - v4 - v0 + 16) >> 5);                            )
//     C(0,
//     })};
//
// static const char dd137_low_y[] = {C(
//     0, int32_t dd137_low_y(int plane, int x, int y) {                     )
//     C(1,    const int h = plane_sizes[plane].y;                             )
//     C(1,                                                                    )
//     C(1,    const int y0 = (x > 3) ? (y - 3) : 1;                           )
//     C(1,    const int32_t v0 = inBuf[getIdx(plane, x, y0)];                 )
//     C(1,                                                                    )
//     C(1,    const int y1 = (y > 1) ? (y - 1) : 1;                           )
//     C(1,    const int32_t v1 = inBuf[getIdx(plane, x, y1)];                 )
//     C(1,                                                                    )
//     C(1,    const int y2 = y;                                               )
//     C(1,    const int32_t v2 = inBuf[getIdx(plane, x, y2)];                 )
//     C(1,                                                                    )
//     C(1,    const int y3 = y + 1;                                           )
//     C(1,    const int32_t v3 = inBuf[getIdx(plane, x, y3)];                 )
//     C(1,                                                                    )
//     C(1,    const int y4 = (y + 3 < h) ? (y + 3) : (h - 1);                 )
//     C(1,    const int32_t v4 = inBuf[getIdx(plane, x, y4)];                 )
//     C(1,                                                                    )
//     C(1,    return dd137_low(v0, v1, v2, v3, v4);                           )
//     C(0,
//     })};
//
// static const char dd137_vert[] = {C(
//     0, void idwt_vert(int plane, int x, int y) {                                          )
//     C(1,    const int h = plane_sizes[plane].y;                                             )
//     C(1,                                                                                    )
//     C(1,    const int32_t out0 = dd137_low_y(plane, x, y);                                  )
//     C(1,    const int32_t out_2 = (y - 2 > 0) ? dd137_low_y(plane, x, y - 2) :              )
//     C(1,                                        dd137_low_y(plane, x, 0);                   )
//     C(1,    const int32_t out2 = (y + 2 < h) ? dd137_low_y(plane, x, y + 2) :               )
//     C(1,                                        dd137_low_y(plane, x, h - 2);               )
//     C(1,    const int32_t out4 = (y + 4 < h) ? dd137_low_y(plane, x, y + 4) :               )
//     C(1,                                        dd137_low_y(plane, x, h - 2);               )
//     C(1,    const int32_t val1 = inBuf[getIdx(plane, x, y + 1)];                            )
//     C(1,                                                                                    )
//     C(1,    outBuf[getIdx(plane, x, y)]     = out0;                                         )
//     C(1,    outBuf[getIdx(plane, x, y + 1)] = dd97_high(out_2, out0, val1, out2, out4);     )
//     C(1,
//     })};
//
// static const char dd137_low_x[] = {C(
//     0, int32_t dd137_low_x(int plane, int x, int y) {                     )
//     C(1,    const int w = plane_sizes[plane].x;                             )
//     C(1,    const int dw = w / 2;                                           )
//     C(1,                                                                    )
//     C(1,    const int x0 = (x > 1) ? x : dw;                                )
//     C(1,    const int32_t v0 = inBuf[getIdx(plane, x0, y)];                 )
//     C(1,                                                                    )
//     C(1,    const int x1 = (x > 1) ? (x + dw - 2) : dw;                     )
//     C(1,    const int32_t v1 = inBuf[getIdx(plane, x1, y)];                 )
//     C(1,                                                                    )
//     C(1,    const int x2 = x;                                               )
//     C(1,    const int32_t v2 = inBuf[getIdx(plane, x2, y)];                 )
//     C(1,                                                                    )
//     C(1,    const int x3 = x + dw;                                          )
//     C(1,    const int32_t v3 = inBuf[getIdx(plane, x3, y)];                 )
//     C(1,                                                                    )
//     C(1,    const int x4 = (x != dw - 1) ? (x + dw + 1) : (dw - 1);         )
//     C(1,    const int32_t v4 = inBuf[getIdx(plane, x4, y)];                 )
//     C(1,                                                                    )
//     C(1,    return dd137_low(v0, v1, v2, v3, v4);                           )
//     C(0,
//     })};
//
// static const char dd137_horiz[] = {C(
//     0, void idwt_horiz(int plane, int x, int y) {                                         )
//     C(1,    const int w = plane_sizes[plane].x;                                             )
//     C(1,    const int dw = w / 2 - 1;                                                       )
//     C(1,                                                                                    )
//     C(1,    const int32_t out0 = dd137_low_x(plane, x, y);                                  )
//     C(1,                                                                                    )
//     C(1,    const int32_t out_1 = ((x - 1) > 0) ? dd137_low_x(plane, x - 1, y) : out0;      )
//     C(1,    const int32_t val3 = inBuf[getIdx(plane, x + dw + 1, y)];                       )
//     C(1,    const int32_t out1 = ((x + 1) <= dw) ? dd137_low_x(plane, x + 1, y) :           )
//     C(1,                                            dd137_low_x(plane, dw, y);              )
//     C(1,    const int32_t out2 = ((x + 2) <= dw) ? dd137_low_x(plane, x + 2, y) :           )
//     C(1,                                            dd137_low_x(plane, dw, y);              )
//     C(1,    const int32_t res  = dd97_high(out_1, out0, val3, out1, out2);                  )
//     C(1,                                                                                    )
//     C(1,    outBuf[getIdx(plane, 2 * x, y)]     = (out0 + 1) >> 1;                          )
//     C(1,    outBuf[getIdx(plane, 2 * x + 1, y)] = (res + 1) >> 1;                           )
//     C(0,
//     })};
//
// static int init_wavelet_shd_dd137_vert(DiracVulkanDecodeContext *s,
//                                        FFVkSPIRVCompiler *spv) {
//     int err = 0;
//     uint8_t *spv_data;
//     size_t spv_len;
//     void *spv_opaque = NULL;
//     int wavelet_idx = DWT_DIRAC_DD13_7;
//     FFVulkanContext *vkctx = &s->vkctx;
//     FFVulkanDescriptorSetBinding *desc;
//     FFVkSPIRVShader *shd = &s->vert_wavelet_shd[wavelet_idx];
//     FFVulkanPipeline *pl = &s->vert_wavelet_pl[wavelet_idx];
//     FFVkExecPool *exec = &s->exec_pool;
//
//     RET(ff_vk_shader_init(pl, shd, "dd137_vert", VK_SHADER_STAGE_COMPUTE_BIT,
//                           0));
//
//     shd = &s->vert_wavelet_shd[wavelet_idx];
//     ff_vk_shader_set_compute_sizes(shd, 8, 8, 1);
//
//     GLSLC(0, #extension GL_EXT_scalar_block_layout : require);
//     GLSLC(0, #extension GL_EXT_shader_explicit_arithmetic_types : require);
//
//     desc = (FFVulkanDescriptorSetBinding[]){
//         {
//             .name = "in_buf",
//             .stages = VK_SHADER_STAGE_COMPUTE_BIT,
//             .type = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
//             .buf_content = "int32_t inBuf[];",
//             .mem_quali = "readonly",
//             .dimensions = 1,
//         },
//         {
//             .name = "out_buf",
//             .stages = VK_SHADER_STAGE_COMPUTE_BIT,
//             .type = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
//             .buf_content = "int32_t outBuf[];",
//             .mem_quali = "writeonly",
//             .dimensions = 1,
//         },
//     };
//     RET(ff_vk_pipeline_descriptor_set_add(vkctx, pl, shd, desc, 2, 0, 0));
//
//     ff_vk_add_push_constant(pl, 0, sizeof(WaveletPushConst),
//                             VK_SHADER_STAGE_COMPUTE_BIT);
//
//     GLSLC(
//         0, layout(push_constant, std430) uniform pushConstants {  );
//             GLSLC(1, ivec2 plane_sizes[3];);
//             GLSLC(1, int plane_offs[3];);
//             GLSLC(1, int plane_strides[3];);
//             GLSLC(1, int dw[3];);
//             GLSLC(1, int wavelet_depth;);
//     GLSLC(0,
//         };);
//     GLSLC(0, );
//
//     GLSLD(get_idx);
//     GLSLD(dd97_high);
//     GLSLD(dd137_low);
//     GLSLD(dd137_low_y);
//     GLSLD(dd137_vert);
//
//     GLSLC(
//         0, void main() {                                                              );
//             GLSLC(1, int off_y = int(gl_WorkGroupSize.y * gl_NumWorkGroups.y););
//             GLSLC(1, int off_x = int(gl_WorkGroupSize.x * gl_NumWorkGroups.x););
//             GLSLC(1, int pic_z = int(gl_GlobalInvocationID.z););
//             GLSLC(1, );
//             GLSLC(1, uint h = int(plane_sizes[pic_z].y););
//             GLSLC(2, uint w = int(plane_sizes[pic_z].x););
//             GLSLC(1, );
//             GLSLC(1, int y = int(gl_GlobalInvocationID.y););
//             GLSLC(
//                 1, for (; 2 * y < h; y += off_y) {                                         );
//                     GLSLC(2, int x = int(gl_GlobalInvocationID.x););
//                     GLSLC(
//                         2, for (; x < w; x += off_x) {                                         );
//                             GLSLC(3, idwt_vert(pic_z, x, 2 * y););
//     GLSLC(2,
//                         });
//     GLSLC(1,
//                 });
//     GLSLC(0,
//         });
//
//     RET(spv->compile_shader(spv, vkctx, shd, &spv_data, &spv_len, "main",
//                             &spv_opaque));
//     RET(ff_vk_shader_create(vkctx, shd, spv_data, spv_len, "main"));
//     RET(ff_vk_init_compute_pipeline(vkctx, pl, shd));
//     RET(ff_vk_exec_pipeline_register(vkctx, exec, pl));
//
// fail:
//     if (spv_opaque)
//         spv->free_shader(spv, &spv_opaque);
//
//     return err;
// }
//
// static int init_wavelet_shd_dd137_horiz(DiracVulkanDecodeContext *s,
//                                         FFVkSPIRVCompiler *spv) {
//     int err = 0;
//     uint8_t *spv_data;
//     size_t spv_len;
//     void *spv_opaque = NULL;
//     int wavelet_idx = DWT_DIRAC_DD13_7;
//     FFVulkanContext *vkctx = &s->vkctx;
//     FFVulkanDescriptorSetBinding *desc;
//     FFVkSPIRVShader *shd = &s->horiz_wavelet_shd[wavelet_idx];
//     FFVulkanPipeline *pl = &s->horiz_wavelet_pl[wavelet_idx];
//     FFVkExecPool *exec = &s->exec_pool;
//
//     RET(ff_vk_shader_init(pl, shd, "dd137_horiz", VK_SHADER_STAGE_COMPUTE_BIT,
//                           0));
//
//     shd = &s->horiz_wavelet_shd[wavelet_idx];
//     ff_vk_shader_set_compute_sizes(shd, 8, 8, 1);
//
//     GLSLC(0, #extension GL_EXT_debug_printf : enable);
//     GLSLC(0, #extension GL_EXT_scalar_block_layout : require);
//     GLSLC(0, #extension GL_EXT_shader_explicit_arithmetic_types : require);
//
//     desc = (FFVulkanDescriptorSetBinding[]){
//         {
//             .name = "in_buf",
//             .stages = VK_SHADER_STAGE_COMPUTE_BIT,
//             .type = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
//             .buf_content = "int32_t inBuf[];",
//             .mem_quali = "readonly",
//             .dimensions = 1,
//         },
//         {
//             .name = "out_buf",
//             .stages = VK_SHADER_STAGE_COMPUTE_BIT,
//             .type = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
//             .buf_content = "int32_t outBuf[];",
//             .mem_quali = "writeonly",
//             .dimensions = 1,
//         },
//     };
//     RET(ff_vk_pipeline_descriptor_set_add(vkctx, pl, shd, desc, 2, 0, 0));
//
//     ff_vk_add_push_constant(pl, 0, sizeof(WaveletPushConst),
//                             VK_SHADER_STAGE_COMPUTE_BIT);
//
//     GLSLC(
//         0, layout(push_constant, std430) uniform pushConstants {  );
//             GLSLC(1, ivec2 plane_sizes[3];);
//             GLSLC(1, int plane_offs[3];);
//             GLSLC(1, int plane_strides[3];);
//             GLSLC(1, int dw[3];);
//             GLSLC(1, int wavelet_depth;);
//     GLSLC(0,
//         };);
//     GLSLC(0, );
//
//     GLSLD(get_idx);
//     GLSLD(dd97_high);
//     GLSLD(dd137_low);
//     GLSLD(dd137_low_x);
//     GLSLD(dd137_horiz);
//
//     GLSLC(
//         0, void main() {                                                              );
//             GLSLC(1, int off_y = int(gl_WorkGroupSize.y * gl_NumWorkGroups.y););
//             GLSLC(1, int off_x = int(gl_WorkGroupSize.x * gl_NumWorkGroups.x););
//             GLSLC(1, int pic_z = int(gl_GlobalInvocationID.z););
//             GLSLC(1, );
//             GLSLC(1, uint h = int(plane_sizes[pic_z].y););
//             GLSLC(2, uint w = int(plane_sizes[pic_z].x););
//             GLSLC(1, );
//             GLSLC(1, int y = int(gl_GlobalInvocationID.y););
//             GLSLC(
//                 1, for (; y < h; y += off_y) {                                             );
//                     GLSLC(2, int x = int(gl_GlobalInvocationID.x););
//                     GLSLC(
//                         2, for (; 2 * x < w; x += off_x) {                                     );
//                             GLSLC(3, idwt_horiz(pic_z, x, y););
//     GLSLC(2,
//                         });
//     GLSLC(1,
//                 });
//     GLSLC(0,
//         });
//
//     RET(spv->compile_shader(spv, vkctx, shd, &spv_data, &spv_len, "main",
//                             &spv_opaque));
//     RET(ff_vk_shader_create(vkctx, shd, spv_data, spv_len, "main"));
//     RET(ff_vk_init_compute_pipeline(vkctx, pl, shd));
//     RET(ff_vk_exec_pipeline_register(vkctx, exec, pl));
//
// fail:
//     if (spv_opaque)
//         spv->free_shader(spv, &spv_opaque);
//
//     return err;
// }
//
// static av_always_inline int inline wavelet_dd137_pass(
//     DiracVulkanDecodeContext *dec, DiracContext *ctx, FFVkExecContext *exec,
//     VkBufferMemoryBarrier2 *buf_bar, int *nb_buf_bar) {
//     int err;
//     int barrier_num = *nb_buf_bar;
//     int wavelet_idx = DWT_DIRAC_DD13_7;
//     FFVulkanFunctions *vk = &dec->vkctx.vkfn;
//
//     FFVulkanPipeline *pl_hor = &dec->horiz_wavelet_pl[wavelet_idx];
//     FFVulkanPipeline *pl_vert = &dec->vert_wavelet_pl[wavelet_idx];
//
//     err = ff_vk_set_descriptor_buffer(&dec->vkctx, pl_vert, exec, 0, 0, 0,
//                                       &dec->tmp_buf, 0, dec->tmp_buf.size,
//                                       VK_FORMAT_UNDEFINED);
//     if (err < 0)
//         goto fail;
//     err = ff_vk_set_descriptor_buffer(
//         &dec->vkctx, pl_vert, exec, 0, 1, 0, &dec->tmp_interleave_buf, 0,
//         dec->tmp_interleave_buf.size, VK_FORMAT_UNDEFINED);
//     if (err < 0)
//         goto fail;
//
//     err = ff_vk_set_descriptor_buffer(
//         &dec->vkctx, pl_hor, exec, 0, 0, 0, &dec->tmp_interleave_buf, 0,
//         dec->tmp_interleave_buf.size, VK_FORMAT_UNDEFINED);
//     if (err < 0)
//         goto fail;
//     err = ff_vk_set_descriptor_buffer(&dec->vkctx, pl_hor, exec, 0, 1, 0,
//                                       &dec->tmp_buf, 0, dec->tmp_buf.size,
//                                       VK_FORMAT_UNDEFINED);
//     if (err < 0)
//         goto fail;
//
//     for (int i = ctx->wavelet_depth - 1; i >= 0; i--) {
//         dec->pConst.plane_strides[0] = ctx->plane[0].idwt.width << i;
//         dec->pConst.plane_strides[1] = ctx->plane[1].idwt.width << i;
//         dec->pConst.plane_strides[2] = ctx->plane[2].idwt.width << i;
//
//         dec->pConst.plane_offs[0] = 0;
//         dec->pConst.plane_offs[1] =
//             ctx->plane[0].idwt.width * ctx->plane[0].idwt.height;
//         dec->pConst.plane_offs[2] =
//             dec->pConst.plane_offs[1] +
//             ctx->plane[1].idwt.width * ctx->plane[1].idwt.height;
//
//         dec->pConst.dw[0] = ctx->plane[0].idwt.width >> (i + 1);
//         dec->pConst.dw[1] = ctx->plane[1].idwt.width >> (i + 1);
//         dec->pConst.dw[2] = ctx->plane[2].idwt.width >> (i + 1);
//
//         dec->pConst.real_plane_dims[0] = (ctx->plane[0].idwt.width) >> i;
//         dec->pConst.real_plane_dims[1] = (ctx->plane[0].idwt.height) >> i;
//         dec->pConst.real_plane_dims[2] = (ctx->plane[1].idwt.width) >> i;
//         dec->pConst.real_plane_dims[3] = (ctx->plane[1].idwt.height) >> i;
//         dec->pConst.real_plane_dims[4] = (ctx->plane[2].idwt.width) >> i;
//         dec->pConst.real_plane_dims[5] = (ctx->plane[2].idwt.height) >> i;
//
//         /* Vertical wavelet pass */
//         ff_vk_update_push_exec(&dec->vkctx, exec, pl_vert,
//                                VK_SHADER_STAGE_COMPUTE_BIT, 0,
//                                sizeof(WaveletPushConst), &dec->pConst);
//
//         barrier_num = *nb_buf_bar;
//         bar_read(buf_bar, nb_buf_bar, &dec->tmp_buf);
//         bar_write(buf_bar, nb_buf_bar, &dec->tmp_buf);
//         bar_read(buf_bar, nb_buf_bar, &dec->tmp_interleave_buf);
//         bar_write(buf_bar, nb_buf_bar, &dec->tmp_interleave_buf);
//
//         vk->CmdPipelineBarrier2(
//             exec->buf,
//             &(VkDependencyInfo){
//                 .sType = VK_STRUCTURE_TYPE_DEPENDENCY_INFO,
//                 .pBufferMemoryBarriers = buf_bar + barrier_num,
//                 .bufferMemoryBarrierCount = *nb_buf_bar - barrier_num,
//             });
//
//         ff_vk_exec_bind_pipeline(&dec->vkctx, exec, pl_vert);
//         vk->CmdDispatch(exec->buf, dec->pConst.real_plane_dims[0] >> 3,
//                         dec->pConst.real_plane_dims[1] >> 4, 3);
//
//         /* Horizontal wavelet pass */
//         ff_vk_update_push_exec(&dec->vkctx, exec, pl_hor,
//                                VK_SHADER_STAGE_COMPUTE_BIT, 0,
//                                sizeof(WaveletPushConst), &dec->pConst);
//
//         barrier_num = *nb_buf_bar;
//         bar_read(buf_bar, nb_buf_bar, &dec->tmp_buf);
//         bar_write(buf_bar, nb_buf_bar, &dec->tmp_buf);
//         bar_read(buf_bar, nb_buf_bar, &dec->tmp_interleave_buf);
//         bar_write(buf_bar, nb_buf_bar, &dec->tmp_interleave_buf);
//
//         vk->CmdPipelineBarrier2(
//             exec->buf,
//             &(VkDependencyInfo){
//                 .sType = VK_STRUCTURE_TYPE_DEPENDENCY_INFO,
//                 .pBufferMemoryBarriers = buf_bar + barrier_num,
//                 .bufferMemoryBarrierCount = *nb_buf_bar - barrier_num,
//             });
//
//         ff_vk_exec_bind_pipeline(&dec->vkctx, exec, pl_hor);
//         vk->CmdDispatch(exec->buf, dec->pConst.real_plane_dims[0] >> 4,
//                         dec->pConst.real_plane_dims[1] >> 3, 3);
//
//         barrier_num = *nb_buf_bar;
//         bar_read(buf_bar, nb_buf_bar, &dec->tmp_buf);
//         bar_write(buf_bar, nb_buf_bar, &dec->tmp_buf);
//         bar_read(buf_bar, nb_buf_bar, &dec->tmp_interleave_buf);
//         bar_write(buf_bar, nb_buf_bar, &dec->tmp_interleave_buf);
//
//         vk->CmdPipelineBarrier2(
//             exec->buf,
//             &(VkDependencyInfo){
//                 .sType = VK_STRUCTURE_TYPE_DEPENDENCY_INFO,
//                 .pBufferMemoryBarriers = buf_bar + barrier_num,
//                 .bufferMemoryBarrierCount = *nb_buf_bar - barrier_num,
//             });
//     }
//
//     return 0;
// fail:
//     ff_vk_exec_discard_deps(&dec->vkctx, exec);
//     return err;
// }
//
// /* ----- Haar Wavelet init and pipeline pass ----- */
//
// static const char haari_horiz[] = {C(
//     0, void idwt_horiz(int plane, int x, int y) {                             )
//     C(1,    int offs0 = plane_offs[plane] + plane_strides[plane] * y + x;       )
//     C(1,    int offs1 = offs0 + plane_sizes[plane].x / 2;                       )
//     C(1,    int outIdx = plane_offs[plane] + plane_strides[plane] * y + x * 2;  )
//     C(1,    int32_t val_orig0 = inBuf[offs0];                                   )
//     C(1,    int32_t val_orig1 = inBuf[offs1];                                   )
//     C(1,    int32_t val_new0 = val_orig0 - ((val_orig1 + 1) >> 1);              )
//     C(1,    int32_t val_new1 = val_orig1 + val_new0;                            )
//     C(1,    outBuf[outIdx] = val_new0;                                          )
//     C(1,    outBuf[outIdx + 1] = val_new1;                                      )
//     C(0,
//     })};
//
// static const char haari_shift_horiz[] = {C(
//     0, void idwt_horiz(int plane, int x, int y) {                             )
//     C(1,    int offs0 = plane_offs[plane] + plane_strides[plane] * y + x;       )
//     C(1,    int offs1 = offs0 + plane_sizes[plane].x / 2;                       )
//     C(1,    int outIdx = plane_offs[plane] + plane_strides[plane] * y + x * 2;  )
//     C(1,    int32_t val_orig0 = inBuf[offs0];                                   )
//     C(1,    int32_t val_orig1 = inBuf[offs1];                                   )
//     C(1,    int32_t val_new0 = val_orig0 - ((val_orig1 + 1) >> 1);              )
//     C(1,    int32_t val_new1 = val_orig1 + val_new0;                            )
//     C(1,    outBuf[outIdx] = (val_new0 + 1) >> 1;                               )
//     C(1,    outBuf[outIdx + 1] = (val_new1 + 1) >> 1;                           )
//     C(0,
//     })};
//
// static const char haari_vert[] = {C(
//     0, void idwt_vert(int plane, int x, int y) {                              )
//     C(1,    int offs0 = plane_offs[plane] + plane_strides[plane] * y + x;       )
//     C(1,    int offs1 = plane_offs[plane] + plane_strides[plane] * (y + 1) + x; )
//     C(2,    int32_t val_orig0 = inBuf[offs0];                                   )
//     C(1,    int32_t val_orig1 = inBuf[offs1];                                   )
//     C(1,    int32_t val_new0 = val_orig0 - ((val_orig1 + 1) >> 1);              )
//     C(1,    int32_t val_new1 = val_orig1 + val_new0;                            )
//     C(1,    outBuf[offs0] = val_new0;                                           )
//     C(1,    outBuf[offs1] = val_new1;                                           )
//     C(0,
//     })};
//
// static int init_wavelet_shd_haari_vert(DiracVulkanDecodeContext *s,
//                                        FFVkSPIRVCompiler *spv, int shift) {
//     int err = 0;
//     uint8_t *spv_data;
//     size_t spv_len;
//     void *spv_opaque = NULL;
//     int wavelet_idx = DWT_DIRAC_HAAR0 + shift;
//     FFVulkanContext *vkctx = &s->vkctx;
//     FFVulkanDescriptorSetBinding *desc;
//     FFVkSPIRVShader *shd = &s->vert_wavelet_shd[wavelet_idx];
//     FFVulkanPipeline *pl = &s->vert_wavelet_pl[wavelet_idx];
//     FFVkExecPool *exec = &s->exec_pool;
//
//     RET(ff_vk_shader_init(pl, shd, "haari_vert", VK_SHADER_STAGE_COMPUTE_BIT,
//                           0));
//
//     shd = &s->vert_wavelet_shd[wavelet_idx];
//     ff_vk_shader_set_compute_sizes(shd, 8, 8, 1);
//
//     GLSLC(0, #extension GL_EXT_scalar_block_layout : require);
//     GLSLC(0, #extension GL_EXT_shader_explicit_arithmetic_types : require);
//
//     desc = (FFVulkanDescriptorSetBinding[]){
//         {
//             .name = "in_buf",
//             .stages = VK_SHADER_STAGE_COMPUTE_BIT,
//             .type = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
//             .buf_content = "int32_t inBuf[];",
//             .mem_quali = "readonly",
//             .dimensions = 1,
//         },
//         {
//             .name = "out_buf",
//             .stages = VK_SHADER_STAGE_COMPUTE_BIT,
//             .type = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
//             .buf_content = "int32_t outBuf[];",
//             .mem_quali = "writeonly",
//             .dimensions = 1,
//         },
//     };
//     RET(ff_vk_pipeline_descriptor_set_add(vkctx, pl, shd, desc, 2, 0, 0));
//
//     ff_vk_add_push_constant(pl, 0, sizeof(WaveletPushConst),
//                             VK_SHADER_STAGE_COMPUTE_BIT);
//
//     GLSLC(
//         0, layout(push_constant, std430) uniform pushConstants {  );
//             GLSLC(1, ivec2 plane_sizes[3];);
//             GLSLC(1, int plane_offs[3];);
//             GLSLC(1, int plane_strides[3];);
//             GLSLC(1, int dw[3];);
//             GLSLC(1, int wavelet_depth;);
//     GLSLC(0,
//         };);
//     GLSLC(0, );
//
//     GLSLD(haari_vert);
//
//     GLSLC(
//         0, void main() {                                                              );
//             GLSLC(1, int off_y = int(gl_WorkGroupSize.y * gl_NumWorkGroups.y););
//             GLSLC(1, int off_x = int(gl_WorkGroupSize.x * gl_NumWorkGroups.x););
//             GLSLC(1, int pic_z = int(gl_GlobalInvocationID.z););
//             GLSLC(1, );
//             GLSLC(1, uint h = int(plane_sizes[pic_z].y););
//             GLSLC(2, uint w = int(plane_sizes[pic_z].x););
//             GLSLC(1, );
//             GLSLC(1, int y = int(gl_GlobalInvocationID.y););
//             GLSLC(
//                 1, for (; 2 * y < h; y += off_y) {                                         );
//                     GLSLC(2, int x = int(gl_GlobalInvocationID.x););
//                     GLSLC(
//                         2, for (; x < w; x += off_x) {                                         );
//                             GLSLC(3, idwt_vert(pic_z, x, 2 * y););
//     GLSLC(2,
//                         });
//     GLSLC(1,
//                 });
//     GLSLC(0,
//         });
//
//     RET(spv->compile_shader(spv, vkctx, shd, &spv_data, &spv_len, "main",
//                             &spv_opaque));
//     RET(ff_vk_shader_create(vkctx, shd, spv_data, spv_len, "main"));
//     RET(ff_vk_init_compute_pipeline(vkctx, pl, shd));
//     RET(ff_vk_exec_pipeline_register(vkctx, exec, pl));
//
// fail:
//     if (spv_opaque)
//         spv->free_shader(spv, &spv_opaque);
//
//     return err;
// }
//
// static int init_wavelet_shd_haari_horiz(DiracVulkanDecodeContext *s,
//                                         FFVkSPIRVCompiler *spv, int shift) {
//     int err = 0;
//     uint8_t *spv_data;
//     size_t spv_len;
//     void *spv_opaque = NULL;
//     int wavelet_idx = DWT_DIRAC_HAAR0 + shift;
//     FFVulkanContext *vkctx = &s->vkctx;
//     FFVulkanDescriptorSetBinding *desc;
//     FFVkSPIRVShader *shd = &s->horiz_wavelet_shd[wavelet_idx];
//     FFVulkanPipeline *pl = &s->horiz_wavelet_pl[wavelet_idx];
//     FFVkExecPool *exec = &s->exec_pool;
//
//     RET(ff_vk_shader_init(pl, shd, "haari_horiz", VK_SHADER_STAGE_COMPUTE_BIT,
//                           0));
//
//     shd = &s->horiz_wavelet_shd[wavelet_idx];
//     ff_vk_shader_set_compute_sizes(shd, 8, 8, 1);
//
//     GLSLC(0, #extension GL_EXT_scalar_block_layout : require);
//     GLSLC(0, #extension GL_EXT_shader_explicit_arithmetic_types : require);
//
//     desc = (FFVulkanDescriptorSetBinding[]){
//         {
//             .name = "in_buf",
//             .stages = VK_SHADER_STAGE_COMPUTE_BIT,
//             .type = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
//             .buf_content = "int32_t inBuf[];",
//             .mem_quali = "readonly",
//             .dimensions = 1,
//         },
//         {
//             .name = "out_buf",
//             .stages = VK_SHADER_STAGE_COMPUTE_BIT,
//             .type = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
//             .buf_content = "int32_t outBuf[];",
//             .mem_quali = "writeonly",
//             .dimensions = 1,
//         },
//     };
//     RET(ff_vk_pipeline_descriptor_set_add(vkctx, pl, shd, desc, 2, 0, 0));
//
//     ff_vk_add_push_constant(pl, 0, sizeof(WaveletPushConst),
//                             VK_SHADER_STAGE_COMPUTE_BIT);
//
//     GLSLC(
//         0, layout(push_constant, std430) uniform pushConstants {  );
//             GLSLC(1, ivec2 plane_sizes[3];);
//             GLSLC(1, int plane_offs[3];);
//             GLSLC(1, int plane_strides[3];);
//             GLSLC(1, int dw[3];);
//             GLSLC(1, int wavelet_depth;);
//     GLSLC(0,
//         };);
//     GLSLC(0, );
//
//     GLSLD(shift ? haari_shift_horiz : haari_horiz);
//
//     GLSLC(
//         0, void main() {                                                              );
//             GLSLC(1, int off_y = int(gl_WorkGroupSize.y * gl_NumWorkGroups.y););
//             GLSLC(1, int off_x = int(gl_WorkGroupSize.x * gl_NumWorkGroups.x););
//             GLSLC(1, int pic_z = int(gl_GlobalInvocationID.z););
//             GLSLC(1, );
//             GLSLC(1, uint w = int(plane_sizes[pic_z].x););
//             GLSLC(1, uint h = int(plane_sizes[pic_z].y););
//             GLSLC(1, );
//             GLSLC(1, int y = int(gl_GlobalInvocationID.y););
//             GLSLC(
//                 1, for (; y < h; y += off_y) {                                             );
//                     GLSLC(2, int x = int(gl_GlobalInvocationID.x););
//                     GLSLC(
//                         2, for (; 2 * x < w; x += off_x) {                                     );
//                             GLSLC(3, idwt_horiz(pic_z, x, y););
//     GLSLC(2,
//                         });
//     GLSLC(1,
//                 });
//     GLSLC(0,
//         });
//
//     RET(spv->compile_shader(spv, vkctx, shd, &spv_data, &spv_len, "main",
//                             &spv_opaque));
//     RET(ff_vk_shader_create(vkctx, shd, spv_data, spv_len, "main"));
//     RET(ff_vk_init_compute_pipeline(vkctx, pl, shd));
//     RET(ff_vk_exec_pipeline_register(vkctx, exec, pl));
//
// fail:
//     if (spv_opaque)
//         spv->free_shader(spv, &spv_opaque);
//
//     return err;
// }
//
// static av_always_inline int inline wavelet_haari_pass(
//     DiracVulkanDecodeContext *dec, DiracContext *ctx, FFVkExecContext *exec,
//     VkBufferMemoryBarrier2 *buf_bar, int *nb_buf_bar, int shift) {
//     int err;
//     int barrier_num = *nb_buf_bar;
//
//     const int wavelet_idx = DWT_DIRAC_HAAR0 + shift;
//     FFVulkanFunctions *vk = &dec->vkctx.vkfn;
//
//     FFVulkanPipeline *pl_hor = &dec->horiz_wavelet_pl[wavelet_idx];
//     FFVulkanPipeline *pl_vert = &dec->vert_wavelet_pl[wavelet_idx];
//
//     err = ff_vk_set_descriptor_buffer(&dec->vkctx, pl_vert, exec, 0, 0, 0,
//                                       &dec->tmp_buf, 0, dec->tmp_buf.size,
//                                       VK_FORMAT_UNDEFINED);
//     if (err < 0)
//         goto fail;
//     err = ff_vk_set_descriptor_buffer(
//         &dec->vkctx, pl_vert, exec, 0, 1, 0, &dec->tmp_interleave_buf, 0,
//         dec->tmp_interleave_buf.size, VK_FORMAT_UNDEFINED);
//     if (err < 0)
//         goto fail;
//
//     err = ff_vk_set_descriptor_buffer(
//         &dec->vkctx, pl_hor, exec, 0, 0, 0, &dec->tmp_interleave_buf, 0,
//         dec->tmp_interleave_buf.size, VK_FORMAT_UNDEFINED);
//     if (err < 0)
//         goto fail;
//     err = ff_vk_set_descriptor_buffer(&dec->vkctx, pl_hor, exec, 0, 1, 0,
//                                       &dec->tmp_buf, 0, dec->tmp_buf.size,
//                                       VK_FORMAT_UNDEFINED);
//     if (err < 0)
//         goto fail;
//
//     for (int i = ctx->wavelet_depth - 1; i >= 0; i--) {
//         dec->pConst.plane_strides[0] = ctx->plane[0].idwt.width << i;
//         dec->pConst.plane_strides[1] = ctx->plane[1].idwt.width << i;
//         dec->pConst.plane_strides[2] = ctx->plane[2].idwt.width << i;
//
//         dec->pConst.plane_offs[0] = 0;
//         dec->pConst.plane_offs[1] =
//             ctx->plane[0].idwt.width * ctx->plane[0].idwt.height;
//         dec->pConst.plane_offs[2] =
//             dec->pConst.plane_offs[1] +
//             ctx->plane[1].idwt.width * ctx->plane[1].idwt.height;
//
//         dec->pConst.dw[0] = ctx->plane[0].idwt.width >> (i + 1);
//         dec->pConst.dw[1] = ctx->plane[1].idwt.width >> (i + 1);
//         dec->pConst.dw[2] = ctx->plane[2].idwt.width >> (i + 1);
//
//         dec->pConst.real_plane_dims[0] = ctx->plane[0].idwt.width >> i;
//         dec->pConst.real_plane_dims[1] = ctx->plane[0].idwt.height >> i;
//         dec->pConst.real_plane_dims[2] = ctx->plane[1].idwt.width >> i;
//         dec->pConst.real_plane_dims[3] = ctx->plane[1].idwt.height >> i;
//         dec->pConst.real_plane_dims[4] = ctx->plane[2].idwt.width >> i;
//         dec->pConst.real_plane_dims[5] = ctx->plane[2].idwt.height >> i;
//
//         dec->pConst.wavelet_depth = ctx->wavelet_depth;
//
//         /* Vertical wavelet pass */
//         ff_vk_update_push_exec(&dec->vkctx, exec, pl_vert,
//                                VK_SHADER_STAGE_COMPUTE_BIT, 0,
//                                sizeof(WaveletPushConst), &dec->pConst);
//
//         barrier_num = *nb_buf_bar;
//         bar_read(buf_bar, nb_buf_bar, &dec->tmp_buf);
//         bar_write(buf_bar, nb_buf_bar, &dec->tmp_buf);
//         bar_read(buf_bar, nb_buf_bar, &dec->tmp_interleave_buf);
//         bar_write(buf_bar, nb_buf_bar, &dec->tmp_interleave_buf);
//
//         vk->CmdPipelineBarrier2(
//             exec->buf,
//             &(VkDependencyInfo){
//                 .sType = VK_STRUCTURE_TYPE_DEPENDENCY_INFO,
//                 .pBufferMemoryBarriers = buf_bar + barrier_num,
//                 .bufferMemoryBarrierCount = *nb_buf_bar - barrier_num,
//             });
//
//         ff_vk_exec_bind_pipeline(&dec->vkctx, exec, pl_vert);
//         vk->CmdDispatch(exec->buf, dec->pConst.real_plane_dims[0] >> 1,
//                         dec->pConst.real_plane_dims[1] >> 2, 3);
//
//         /* Horizontal wavelet pass */
//         ff_vk_update_push_exec(&dec->vkctx, exec, pl_hor,
//                                VK_SHADER_STAGE_COMPUTE_BIT, 0,
//                                sizeof(WaveletPushConst), &dec->pConst);
//
//         barrier_num = *nb_buf_bar;
//         bar_read(buf_bar, nb_buf_bar, &dec->tmp_buf);
//         bar_write(buf_bar, nb_buf_bar, &dec->tmp_buf);
//         bar_read(buf_bar, nb_buf_bar, &dec->tmp_interleave_buf);
//         bar_write(buf_bar, nb_buf_bar, &dec->tmp_interleave_buf);
//
//         vk->CmdPipelineBarrier2(
//             exec->buf,
//             &(VkDependencyInfo){
//                 .sType = VK_STRUCTURE_TYPE_DEPENDENCY_INFO,
//                 .pBufferMemoryBarriers = buf_bar + barrier_num,
//                 .bufferMemoryBarrierCount = *nb_buf_bar - barrier_num,
//             });
//
//         ff_vk_exec_bind_pipeline(&dec->vkctx, exec, pl_hor);
//         vk->CmdDispatch(exec->buf, dec->pConst.real_plane_dims[0] >> 2,
//                         dec->pConst.real_plane_dims[1] >> 1, 3);
//     }
//
//     barrier_num = *nb_buf_bar;
//     bar_read(buf_bar, nb_buf_bar, &dec->tmp_buf);
//     bar_write(buf_bar, nb_buf_bar, &dec->tmp_buf);
//     bar_read(buf_bar, nb_buf_bar, &dec->tmp_interleave_buf);
//     bar_write(buf_bar, nb_buf_bar, &dec->tmp_interleave_buf);
//     vk->CmdPipelineBarrier2(
//         exec->buf, &(VkDependencyInfo){
//                        .sType = VK_STRUCTURE_TYPE_DEPENDENCY_INFO,
//                        .pBufferMemoryBarriers = buf_bar + barrier_num,
//                        .bufferMemoryBarrierCount = *nb_buf_bar - barrier_num,
//                    });
//
//     return 0;
// fail:
//     ff_vk_exec_discard_deps(&dec->vkctx, exec);
//     return err;
// }
//
// /* ----- Dequant Shader init and pipeline pass ----- */

extern const char *ff_source_vulkan_dirac_dequant_comp;

static int init_quant_shd(DiracVulkanDecodeContext *s, FFVkSPIRVCompiler *spv) {
    int err = 0;
    int dims[3] = {1, 1, 1};
    static const char *ext[] = {
        "GL_EXT_debug_printf",
        "GL_EXT_scalar_block_layout",
        "GL_EXT_shader_explicit_arithmetic_types",
    };

    FFVulkanDescriptorSetBinding *desc = (FFVulkanDescriptorSetBinding[]){
        {
            .name = "out_buf_0",
            .stages = VK_SHADER_STAGE_COMPUTE_BIT,
            .type = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
            .buf_content = "int32_t outBuf0[];",
            .mem_quali = "writeonly",
            .dimensions = 1,
        },
        {
            .name = "out_buf_1",
            .stages = VK_SHADER_STAGE_COMPUTE_BIT,
            .type = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
            .buf_content = "int32_t outBuf1[];",
            .mem_quali = "writeonly",
            .dimensions = 1,
        },
        {
            .name = "quant_in_buf",
            .type = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
            .stages = VK_SHADER_STAGE_COMPUTE_BIT,
            .buf_content = "int32_t inBuffer[];",
            .mem_quali = "readonly",
        },
        {
            .name = "quant_vals_buf",
            .type = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
            .stages = VK_SHADER_STAGE_COMPUTE_BIT,
            .buf_content = "int32_t quantMatrix[];",
            .mem_quali = "readonly",
        },
        {
            .name = "slices_buf",
            .type = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
            .stages = VK_SHADER_STAGE_COMPUTE_BIT,
            .buf_content = "Slice slices[];",
            .mem_quali = "readonly",
            .mem_layout = "std430",
        },
        {
            .name = "subband_buf",
            .type = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER,
            .stages = VK_SHADER_STAGE_COMPUTE_BIT,
            .buf_content = "SubbandOffset subband_offs[60];",
            .mem_quali = "readonly",
            .mem_layout = "std430",
        },
    };


    err = compile_shader(s, spv, &s->quant,
                        desc, 6, ext, 3,
                        "dequant", (const char *)ff_source_vulkan_dirac_dequant_comp,
                        dims, sizeof(WaveletPushConst));

    return err;
}

static av_always_inline int inline quant_pl_pass(
    DiracVulkanDecodeContext *dec, DiracContext *ctx, FFVkExecContext *exec,
    VkBufferMemoryBarrier2 *buf_bar, int *nb_buf_bar) {
    int err, nb_bar;
    FFVulkanFunctions *vk = &dec->vkctx.vkfn;

    err = ff_vk_shader_update_desc_buffer(&dec->vkctx, exec, &dec->quant,
                                                0, 0, 0, &dec->tmp_buf, 0,
                                                dec->tmp_buf.size, VK_FORMAT_UNDEFINED);
    if (err < 0)
        return err;

    err = ff_vk_shader_update_desc_buffer(&dec->vkctx, exec, &dec->quant,
                                                0, 1, 0, &dec->tmp_interleave_buf, 0,
                                                dec->tmp_interleave_buf.size,
                                                VK_FORMAT_UNDEFINED);
    if (err < 0)
        return err;

    err = ff_vk_shader_update_desc_buffer(&dec->vkctx, exec, &dec->quant,
                                                0, 2, 0, dec->quant_val_buf, 0,
                                                dec->quant_val_buf->size, VK_FORMAT_UNDEFINED);
    if (err < 0)
        return err;

    err = ff_vk_shader_update_desc_buffer(&dec->vkctx, exec, &dec->quant,
                                                0, 3, 0, dec->quant_buf, 0,
                                                dec->quant_buf->size, VK_FORMAT_UNDEFINED);
    if (err < 0)
        return err;

    err = ff_vk_shader_update_desc_buffer(&dec->vkctx, exec, &dec->quant,
                                                0, 4, 0, dec->slice_buf, 0,
                                                dec->slice_buf->size, VK_FORMAT_UNDEFINED);
    if (err < 0)
        return err;

    err = ff_vk_shader_update_desc_buffer(&dec->vkctx, exec, &dec->quant,
                                                0, 5, 0, &dec->subband_info, 0,
                                                dec->subband_info.size, VK_FORMAT_UNDEFINED);
    if (err < 0)
        return err;

    dec->pConst.real_plane_dims[0] = ctx->plane[0].idwt.width;
    dec->pConst.real_plane_dims[1] = ctx->plane[0].idwt.height;
    dec->pConst.real_plane_dims[2] = ctx->plane[1].idwt.width;
    dec->pConst.real_plane_dims[3] = ctx->plane[1].idwt.height;
    dec->pConst.real_plane_dims[4] = ctx->plane[2].idwt.width;
    dec->pConst.real_plane_dims[5] = ctx->plane[2].idwt.height;

    dec->pConst.plane_strides[0] = ctx->plane[0].idwt.width;
    dec->pConst.plane_strides[1] = ctx->plane[1].idwt.width;
    dec->pConst.plane_strides[2] = ctx->plane[2].idwt.width;

    dec->pConst.plane_offs[0] = 0;
    dec->pConst.plane_offs[1] =
        ctx->plane[0].idwt.width * ctx->plane[0].idwt.height;
    dec->pConst.plane_offs[2] =
        dec->pConst.plane_offs[1] +
        ctx->plane[1].idwt.width * ctx->plane[1].idwt.height;

    dec->pConst.wavelet_depth = ctx->wavelet_depth;

    ff_vk_shader_update_push_const(&dec->vkctx, exec, &dec->quant,
                                        VK_SHADER_STAGE_COMPUTE_BIT, 0,
                                        sizeof(WaveletPushConst), &dec->pConst);

    bar_read(buf_bar, nb_buf_bar, dec->quant_val_buf);
    bar_read(buf_bar, nb_buf_bar, dec->slice_buf);
    bar_read(buf_bar, nb_buf_bar, dec->quant_buf);
    bar_read(buf_bar, nb_buf_bar, &dec->subband_info);

    bar_write(buf_bar, nb_buf_bar, &dec->tmp_buf);
    bar_write(buf_bar, nb_buf_bar, &dec->tmp_interleave_buf);
    bar_read(buf_bar, nb_buf_bar, &dec->tmp_buf);
    bar_read(buf_bar, nb_buf_bar, &dec->tmp_interleave_buf);

    vk->CmdPipelineBarrier2(exec->buf,
                            &(VkDependencyInfo){
                                .sType = VK_STRUCTURE_TYPE_DEPENDENCY_INFO,
                                .pBufferMemoryBarriers = buf_bar,
                                .bufferMemoryBarrierCount = *nb_buf_bar,
                            });

    ff_vk_exec_bind_shader(&dec->vkctx, exec, &dec->quant);
    vk->CmdDispatch(exec->buf, ctx->num_x * ctx->num_y, 3, ctx->wavelet_depth);

    nb_bar = *nb_buf_bar;
    bar_write(buf_bar, nb_buf_bar, &dec->tmp_buf);
    bar_write(buf_bar, nb_buf_bar, &dec->tmp_interleave_buf);
    bar_read(buf_bar, nb_buf_bar, &dec->tmp_buf);
    bar_read(buf_bar, nb_buf_bar, &dec->tmp_interleave_buf);

    vk->CmdPipelineBarrier2(
        exec->buf, &(VkDependencyInfo){
                       .sType = VK_STRUCTURE_TYPE_DEPENDENCY_INFO,
                       .pBufferMemoryBarriers = buf_bar + nb_bar,
                       .bufferMemoryBarrierCount = *nb_buf_bar - nb_bar,
                   });

    return 0;
}

static int vulkan_dirac_uninit(AVCodecContext *avctx) {
    free_common(avctx);

    return 0;
}

static inline int wavelet_init(DiracVulkanDecodeContext *dec,
                               FFVkSPIRVCompiler *spv) {
    int err;

//     err = init_wavelet_shd_daub97_horiz(dec, spv);
//     if (err < 0) {
//         return err;
//     }
//
//     err = init_wavelet_shd_daub97_vert(dec, spv);
//     if (err < 0) {
//         return err;
//     }
//
//     err = init_wavelet_shd_haari_vert(dec, spv, 0);
//     if (err < 0) {
//         return err;
//     }
//
//     err = init_wavelet_shd_haari_horiz(dec, spv, 0);
//     if (err < 0) {
//         return err;
//     }
//
//     err = init_wavelet_shd_haari_vert(dec, spv, 1);
//     if (err < 0) {
//         return err;
//     }
//
//     err = init_wavelet_shd_haari_horiz(dec, spv, 1);
//     if (err < 0) {
//         return err;
//     }

    err = init_wavelet_shd_legall_vert(dec, spv);
    if (err < 0) {
        return err;
    }

    err = init_wavelet_shd_legall_horiz(dec, spv);
    if (err < 0) {
        return err;
    }

//     err = init_wavelet_shd_dd97_vert(dec, spv);
//     if (err < 0) {
//         return err;
//     }
//
//     err = init_wavelet_shd_dd97_horiz(dec, spv);
//     if (err < 0) {
//         return err;
//     }
//
//     err = init_wavelet_shd_fidelity_vert(dec, spv);
//     if (err < 0) {
//         return err;
//     }
//
//     err = init_wavelet_shd_fidelity_horiz(dec, spv);
//     if (err < 0) {
//         return err;
//     }
//
//     err = init_wavelet_shd_dd137_vert(dec, spv);
//     if (err < 0) {
//         return err;
//     }
//
//     err = init_wavelet_shd_dd137_horiz(dec, spv);
//     if (err < 0) {
//         return err;
//     }

    return 0;
}

static int vulkan_dirac_init(AVCodecContext *avctx) {
    int err = 0;
    DiracVulkanDecodeContext *dec = avctx->internal->hwaccel_priv_data;
    FFVulkanContext *s;
    FFVkSPIRVCompiler *spv;

    spv = ff_vk_spirv_init();
    if (!spv) {
        av_log(avctx, AV_LOG_ERROR, "Unable to initialize SPIR-V compiler!\n");
        return AVERROR_EXTERNAL;
    }

    err = ff_decode_get_hw_frames_ctx(avctx, AV_HWDEVICE_TYPE_VULKAN);
    if (err < 0)
        goto fail;

    /* Initialize contexts */
    s = &dec->vkctx;

    err = ff_vk_init(s, avctx, NULL, avctx->hw_frames_ctx);
    if (err < 0)
        goto fail;

    /* Create queue context */
    dec->qf = ff_vk_qf_find(&dec->vkctx, VK_QUEUE_COMPUTE_BIT, 0);
    if (!dec->qf) {
        av_log(avctx, AV_LOG_ERROR, "Device has no compute queues!\n");
        return err;
    }

    err = ff_vk_exec_pool_init(s, dec->qf, &dec->exec_pool, 8, 0, 0, 0, NULL);
    if (err < 0)
        goto fail;

    err = ff_vk_init_sampler(&dec->vkctx, &dec->sampler, 1, VK_FILTER_LINEAR);
    if (err < 0) {
        goto fail;
    }

    av_log(avctx, AV_LOG_VERBOSE, "Vulkan decoder initialization sucessful\n");

    err = init_quant_shd(dec, spv);
    if (err < 0) {
        goto fail;
    }

    err = init_cpy_shd(dec, spv, 0);
    if (err < 0) {
        goto fail;
    }

    err = init_cpy_shd(dec, spv, 1);
    if (err < 0) {
        goto fail;
    }

    err = init_cpy_shd(dec, spv, 2);
    if (err < 0) {
        goto fail;
    }

    err = wavelet_init(dec, spv);
    if (err < 0) {
        goto fail;
    }

    dec->quant_val_buf_vk_ptr = NULL;
    dec->slice_buf_vk_ptr = NULL;
    dec->quant_buf_vk_ptr = NULL;

    dec->av_quant_val_buf = NULL;
    dec->av_quant_buf = NULL;
    dec->av_slice_buf = NULL;

    dec->thread_buf_size = 0;
    dec->n_slice_bufs = 0;

    err = ff_vk_create_buf(&dec->vkctx, &dec->subband_info,
                           sizeof(SubbandOffset) * MAX_DWT_LEVELS * 12, NULL,
                           NULL,
                           VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT |
                               VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT,
                           VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT |
                               VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
    if (err < 0)
        goto fail;

    err = ff_vk_map_buffer(&dec->vkctx, &dec->subband_info,
                           (uint8_t **)&dec->subband_info_ptr, 0);
    if (err < 0)
        goto fail;

    return 0;

fail:
    if (spv) {
        spv->uninit(&spv);
    }
    vulkan_dirac_uninit(avctx);

    return err;
}

static int vulkan_decode_bootstrap(AVCodecContext *avctx,
                                   AVBufferRef *frames_ref) {
    DiracVulkanDecodeContext *dec = avctx->internal->hwaccel_priv_data;
    AVHWFramesContext *frames = (AVHWFramesContext *)frames_ref->data;
    AVHWDeviceContext *device = (AVHWDeviceContext *)frames->device_ref->data;
    AVVulkanDeviceContext *hwctx = device->hwctx;

    dec->vkctx.extensions = ff_vk_extensions_to_mask(
        hwctx->enabled_dev_extensions, hwctx->nb_enabled_dev_extensions);

    return 0;
}

static int vulkan_dirac_frame_params(AVCodecContext *avctx,
                                     AVBufferRef *hw_frames_ctx) {
    int err;
    AVHWFramesContext *frames_ctx = (AVHWFramesContext *)hw_frames_ctx->data;
    AVVulkanFramesContext *hwfc = frames_ctx->hwctx;
    DiracContext *s = avctx->priv_data;

    frames_ctx->sw_format = s->sof_pix_fmt;
    frames_ctx->width = avctx->width;
    frames_ctx->height = avctx->height;
    frames_ctx->format = AV_PIX_FMT_VULKAN;

    err = vulkan_decode_bootstrap(avctx, hw_frames_ctx);
    if (err < 0)
        return err;


    for (int i = 0; i < AV_NUM_DATA_POINTERS; i++) {
        hwfc->format[i] = av_vkfmt_from_pixfmt(frames_ctx->sw_format)[i];
    }

    hwfc->tiling = VK_IMAGE_TILING_OPTIMAL;
    hwfc->usage =   VK_IMAGE_USAGE_TRANSFER_SRC_BIT |
                    VK_IMAGE_USAGE_TRANSFER_SRC_BIT |
                    VK_IMAGE_USAGE_STORAGE_BIT;

    return err;
}

static void vulkan_dirac_free_frame_priv(AVRefStructOpaque _hwctx, void *data) {
    DiracVulkanDecodePicture *dp = data;

    /* Free frame resources */
    av_free(dp);
}

static void setup_subbands(DiracContext *ctx, DiracVulkanDecodeContext *dec) {
    SubbandOffset *offs = dec->subband_info_ptr;
    memset(offs, 0, dec->subband_info.size);

    for (int plane = 0; plane < 3; plane++) {
        Plane *p = &ctx->plane[plane];
        int w = p->idwt.width;
        int s = FFALIGN(p->idwt.width, 8);

        for (int level = ctx->wavelet_depth - 1; level >= 0; level--) {
            w >>= 1;
            s <<= 1;
            for (int orient = 0; orient < 4; orient++) {
                const int idx = plane * MAX_DWT_LEVELS * 4 + level * 4 + orient;
                SubbandOffset *off = &offs[idx];
                off->stride = s;
                off->base_off = 0;

                if (orient & 1)
                    off->base_off += w;
                if (orient > 1)
                    off->base_off += (s >> 1);
            }
        }
    }
}

static int vulkan_dirac_start_frame(AVCodecContext *avctx,
                                    av_unused const uint8_t *buffer,
                                    av_unused uint32_t size) {
    int err;
    DiracVulkanDecodeContext *s = avctx->internal->hwaccel_priv_data;
    DiracContext *c = avctx->priv_data;
    DiracVulkanDecodePicture *pic = c->hwaccel_picture_private;
    WaveletPushConst *pConst = &s->pConst;

    pic->frame = c->current_picture;
    setup_subbands(c, s);

    pConst->real_plane_dims[0] = c->plane[0].idwt.width;
    pConst->real_plane_dims[1] = c->plane[0].idwt.height;
    pConst->real_plane_dims[2] = c->plane[1].idwt.width;
    pConst->real_plane_dims[3] = c->plane[1].idwt.height;
    pConst->real_plane_dims[4] = c->plane[2].idwt.width;
    pConst->real_plane_dims[5] = c->plane[2].idwt.height;

    pConst->plane_strides[0] = c->plane[0].idwt.width;
    pConst->plane_strides[1] = c->plane[1].idwt.width;
    pConst->plane_strides[0] = c->plane[0].idwt.width;

    pConst->plane_offs[0] = 0;
    pConst->plane_offs[1] = c->plane[0].idwt.width * c->plane[0].idwt.height;
    pConst->plane_offs[2] = pConst->plane_offs[1] +
                            c->plane[1].idwt.width * c->plane[1].idwt.height;

    pConst->wavelet_depth = c->wavelet_depth;

    if (s->quant_buf_vk_ptr == NULL || s->slice_buf_vk_ptr == NULL ||
        s->quant_val_buf_vk_ptr == NULL ||
        c->num_x * c->num_y != s->n_slice_bufs) {
        err = alloc_quant_buf(c, s);
        if (err < 0)
            return err;
        err = alloc_dequant_buf(c, s);
        if (err < 0)
            return err;
        err = alloc_slices_buf(c, s);
        if (err < 0)
            return err;
        err = alloc_tmp_bufs(c, s);
        if (err < 0)
            return err;
    }

    return 0;
}

static int vulkan_dirac_end_frame(AVCodecContext *avctx) {
    int err, nb_img_bar = 0, nb_buf_bar = 0;
    DiracVulkanDecodeContext *dec = avctx->internal->hwaccel_priv_data;
    DiracContext *ctx = avctx->priv_data;
    VkImageView views[AV_NUM_DATA_POINTERS];
    VkBufferMemoryBarrier2 buf_bar[80];
    VkImageMemoryBarrier2 img_bar[80];
    DiracVulkanDecodePicture *pic = ctx->hwaccel_picture_private;
    FFVkExecContext *exec = ff_vk_exec_get(&dec->vkctx, &dec->exec_pool);
    enum dwt_type wavelet_idx = ctx->wavelet_idx + 2;

    ff_vk_exec_start(&dec->vkctx, exec);

    err = quant_pl_pass(dec, ctx, exec, buf_bar, &nb_buf_bar);
    if (err < 0)
        goto fail;

    err = ff_vk_exec_add_dep_frame(&dec->vkctx, exec, ctx->current_picture->avframe,
                                   VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT,
                                   VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT);
    if (err < 0)
        goto fail;

    err = ff_vk_create_imageviews(&dec->vkctx, exec,
                                    views, pic->frame->avframe, FF_VK_REP_FLOAT);
    if (err < 0)
        goto fail;

    switch (wavelet_idx) {
    // case DWT_DIRAC_DAUB9_7:
    //     err = wavelet_daub97_pass(dec, ctx, exec, buf_bar, &nb_buf_bar);
    //     break;
    //
    // case DWT_DIRAC_FIDELITY:
    //     err = wavelet_fidelity_pass(dec, ctx, exec, buf_bar, &nb_buf_bar);
    //     break;
    //
    // case DWT_DIRAC_DD9_7:
    //     err = wavelet_dd97_pass(dec, ctx, exec, buf_bar, &nb_buf_bar);
    //     break;
    //
    // case DWT_DIRAC_DD13_7:
    //     err = wavelet_dd137_pass(dec, ctx, exec, buf_bar, &nb_buf_bar);
    //     break;
    //
    case DWT_DIRAC_LEGALL5_3:
        err = wavelet_legall_pass(dec, ctx, exec, buf_bar, &nb_buf_bar);
        break;
    //
    // case DWT_DIRAC_HAAR0:
    //     err = wavelet_haari_pass(dec, ctx, exec, buf_bar, &nb_buf_bar, 0);
    //     break;
    //
    // case DWT_DIRAC_HAAR1:
    //     err = wavelet_haari_pass(dec, ctx, exec, buf_bar, &nb_buf_bar, 1);
    //     break;
    //
    default:
        err = AVERROR_PATCHWELCOME;
        break;
    }

    err = cpy_to_image_pass(dec, ctx, exec, views, buf_bar, &nb_buf_bar,
                            img_bar, &nb_img_bar, (ctx->bit_depth - 8) >> 1);
    if (err < 0)
        goto fail;

    err = ff_vk_exec_submit(&dec->vkctx, exec);
    if (err < 0)
        goto fail;

    ff_vk_exec_wait(&dec->vkctx, exec);

    return 0;

fail:
    ff_vk_exec_discard_deps(&dec->vkctx, exec);
    return err;
}

static int vulkan_dirac_update_thread_context(AVCodecContext *dst,
                                              const AVCodecContext *src) {
    int err;
    DiracVulkanDecodeContext *src_ctx = src->internal->hwaccel_priv_data;
    DiracVulkanDecodeContext *dst_ctx = dst->internal->hwaccel_priv_data;
    FFVkSPIRVCompiler *spv;

    spv = ff_vk_spirv_init();
    if (!spv) {
        av_log(dst, AV_LOG_ERROR, "Unable to initialize SPIR-V compiler!\n");
        return AVERROR_EXTERNAL;
    }

    memset(dst_ctx, 0, sizeof(DiracVulkanDecodeContext));

    dst_ctx->vkctx = src_ctx->vkctx;
    dst_ctx->qf = ff_vk_qf_find(&dst_ctx->vkctx, VK_QUEUE_COMPUTE_BIT, 0);
    if (!dst_ctx->qf) {
        av_log(dst, AV_LOG_ERROR, "Device has no compute queues!\n");
        return err;
    }

    err = ff_vk_exec_pool_init(&dst_ctx->vkctx, dst_ctx->qf, &dst_ctx->exec_pool, 8, 0, 0, 0, NULL);
    if (err < 0)
        goto fail;

    err = ff_vk_init_sampler(&dst_ctx->vkctx, &dst_ctx->sampler, 1, VK_FILTER_LINEAR);
    if (err < 0) {
        goto fail;
    }

    err = init_quant_shd(dst_ctx, spv);
    if (err < 0) {
        goto fail;
    }

    err = init_cpy_shd(dst_ctx, spv, 0);
    if (err < 0) {
        goto fail;
    }

    err = init_cpy_shd(dst_ctx, spv, 1);
    if (err < 0) {
        goto fail;
    }

    err = init_cpy_shd(dst_ctx, spv, 2);
    if (err < 0) {
        goto fail;
    }

    err = wavelet_init(dst_ctx, spv);
    if (err < 0) {
        goto fail;
    }

    dst_ctx->quant_val_buf_vk_ptr = NULL;
    dst_ctx->slice_buf_vk_ptr = NULL;
    dst_ctx->quant_buf_vk_ptr = NULL;

    dst_ctx->av_quant_val_buf = NULL;
    dst_ctx->av_quant_buf = NULL;
    dst_ctx->av_slice_buf = NULL;

    dst_ctx->thread_buf_size = 0;
    dst_ctx->n_slice_bufs = 0;

    err = ff_vk_create_buf(&dst_ctx->vkctx, &dst_ctx->subband_info,
                           sizeof(SubbandOffset) * MAX_DWT_LEVELS * 12, NULL,
                           NULL,
                           VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT |
                               VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT,
                           VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT |
                               VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
    if (err < 0)
        goto fail;

    err = ff_vk_map_buffer(&dst_ctx->vkctx, &dst_ctx->subband_info,
                           (uint8_t **)&dst_ctx->subband_info_ptr, 0);
    if (err < 0)
        goto fail;

    return 0;

fail:
    if (spv) {
        spv->uninit(&spv);
    }
    vulkan_dirac_uninit(dst);

    return err;
}

static inline int decode_hq_slice(const DiracContext *s, int jobnr) {
    int i, level, orientation, quant_idx;
    DiracVulkanDecodeContext *dec = s->avctx->internal->hwaccel_priv_data;
    int32_t *qfactor = &dec->quant_buf_vk_ptr[jobnr * 8 * MAX_DWT_LEVELS];
    int32_t *qoffset = &dec->quant_buf_vk_ptr[jobnr * 8 * MAX_DWT_LEVELS + 4];
    int32_t *quant_val_base = dec->quant_val_buf_vk_ptr;
    DiracSlice *slice = &s->slice_params_buf[jobnr];
    SliceCoeffVk *slice_vk = &dec->slice_buf_vk_ptr[jobnr * 3 * MAX_DWT_LEVELS];
    GetBitContext *gb = &slice->gb;

    skip_bits_long(gb, 8 * s->highquality.prefix_bytes);
    quant_idx = get_bits(gb, 8);

    if (quant_idx > DIRAC_MAX_QUANT_INDEX - 1) {
        av_log(s->avctx, AV_LOG_ERROR, "Invalid quantization index - %i\n",
               quant_idx);
        return AVERROR_INVALIDDATA;
    }

    /* Slice quantization (slice_quantizers() in the specs) */
    for (level = 0; level < s->wavelet_depth; level++) {
        for (orientation = !!level; orientation < 4; orientation++) {
            const int quant =
                FFMAX(quant_idx - s->lowdelay.quant[level][orientation], 0);
            qfactor[level * 8 + orientation] = ff_dirac_qscale_tab[quant];
            qoffset[level * 8 + orientation] =
                ff_dirac_qoffset_intra_tab[quant] + 2;
        }
    }

    /* Luma + 2 Chroma planes */
    for (i = 0; i < 3; i++) {
        int coef_num, coef_par;
        int64_t length = s->highquality.size_scaler * get_bits(gb, 8);
        int64_t bits_end = get_bits_count(gb) + 8 * length;
        const uint8_t *addr = align_get_bits(gb);
        int offs = dec->slice_vals_size * (3 * jobnr + i);
        uint8_t *tmp_buf = (uint8_t *)&quant_val_base[offs];

        if (length * 8 > get_bits_left(gb)) {
            av_log(s->avctx, AV_LOG_ERROR, "end too far away\n");
            return AVERROR_INVALIDDATA;
        }

        coef_num = subband_coeffs(s, slice->slice_x, slice->slice_y, i, offs,
                                  &slice_vk[MAX_DWT_LEVELS * i]);

        coef_par = ff_dirac_golomb_read_32bit(addr, length, tmp_buf, coef_num);

        if (coef_num > coef_par) {
            const int start_b = coef_par * sizeof(int32_t);
            const int end_b = coef_num * sizeof(int32_t);
            memset(&tmp_buf[start_b], 0, end_b - start_b);
        }

        skip_bits_long(gb, bits_end - get_bits_count(gb));
    }

    return 0;
}

static int decode_hq_slice_row(AVCodecContext *avctx, void *arg, int jobnr,
                               int threadnr) {
    const DiracContext *s = avctx->priv_data;
    int i, jobn = s->num_x * jobnr;

    for (i = 0; i < s->num_x; i++) {
        decode_hq_slice(s, jobn);
        jobn++;
    }

    return 0;
}

static int vulkan_dirac_decode_slice(AVCodecContext *avctx, const uint8_t *data,
                                     uint32_t size) {
    DiracContext *s = avctx->priv_data;

    for (int i = 0; i < s->num_y; i++) {
        decode_hq_slice_row(avctx, NULL, i, 0);
    }

    return 0;
}

const FFHWAccel ff_dirac_vulkan_hwaccel = {
    .p.name = "dirac_vulkan",
    .p.type = AVMEDIA_TYPE_VIDEO,
    .p.id = AV_CODEC_ID_DIRAC,
    .p.pix_fmt = AV_PIX_FMT_VULKAN,
    .start_frame = &vulkan_dirac_start_frame,
    .end_frame = &vulkan_dirac_end_frame,
    .decode_slice = &vulkan_dirac_decode_slice,
    .free_frame_priv = &vulkan_dirac_free_frame_priv,
    .uninit = &vulkan_dirac_uninit,
    .init = &vulkan_dirac_init,
    .frame_params = &vulkan_dirac_frame_params,
    .frame_priv_data_size = sizeof(DiracVulkanDecodePicture),
    .decode_params = &ff_vk_params_invalidate,
    .flush = &ff_vk_decode_flush,
    .update_thread_context = &vulkan_dirac_update_thread_context,
    .priv_data_size = sizeof(DiracVulkanDecodeContext),
    .caps_internal = HWACCEL_CAP_ASYNC_SAFE | HWACCEL_CAP_THREAD_SAFE,
};
