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

#include "diracdec.h"
#include "vulkan.h"
#include "vulkan_decode.h"
#include "libavfilter/vulkan_spirv.h"
#include "libavutil/vulkan_loader.h"
#include "libavcodec/pthread_internal.h"
#include "libavcodec/dirac_vlc.c"


typedef struct SubbandOffset {
    int32_t base_off;
    int32_t stride;
    int32_t pad0;
    int32_t pad1;
} SubbandOffset;

typedef struct SliceCoeffVk {
    int32_t left;
    int32_t top;
    int32_t tot_h;
    int32_t tot_v;
    int32_t tot;
    int32_t offs;
    int32_t pad0;
    int32_t pad1;
} SliceCoeffVk;

typedef struct WaveletPushConst {
    int real_plane_dims[6];
    int plane_offs[3];
    int plane_strides[3];
    int dw[3];
    int wavelet_depth;
} WaveletPushConst;

typedef  struct DiracVulkanDecodeContext {
    FFVulkanContext vkctx;
    VkSamplerYcbcrConversion yuv_sampler;
    VkSampler sampler;

    FFVulkanPipeline vert_wavelet_pl[7];
    FFVkSPIRVShader vert_wavelet_shd[7];

    FFVulkanPipeline vert_interleave_pl[7];
    FFVkSPIRVShader vert_interleave_shd[7];

    FFVulkanPipeline horiz_wavelet_pl[7];
    FFVkSPIRVShader horiz_wavelet_shd[7];

    FFVulkanPipeline horiz_interleave_pl[7];
    FFVkSPIRVShader horiz_interleave_shd[7];

    FFVulkanPipeline cpy_to_image_pl;
    FFVkSPIRVShader cpy_to_image_shd;

    FFVulkanPipeline quant_pl;
    FFVkSPIRVShader quant_shd;

    FFVkQueueFamilyCtx qf;
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

typedef  struct DiracVulkanDecodePicture {
    DiracFrame *frame;
} DiracVulkanDecodePicture;

static const char dd97i_vert[] = {
    C(0, void idwt_vert(int i1, int i2) {                       )
    C(0,                                                        )
    C(0,                                                        )
    C(0,                                                        )
    C(0,                                                        )
    C(0,                                                        )
    C(0,                                                        )
    C(0, }                                                      )

};

static void free_common(AVCodecContext *avctx)
{
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

    ff_vk_pipeline_free(s, &dec->quant_pl);
    ff_vk_shader_free(s, &dec->quant_shd);

    ff_vk_pipeline_free(s, &dec->cpy_to_image_pl);
    ff_vk_shader_free(s, &dec->cpy_to_image_shd);

    // TODO: Add freeing all pipelines and shaders for wavelets
    for (int i = 0; i < 7; i++) {
        ff_vk_pipeline_free(s, &dec->horiz_interleave_pl[i]);
        ff_vk_shader_free(s, &dec->horiz_interleave_shd[i]);

        ff_vk_pipeline_free(s, &dec->vert_interleave_pl[i]);
        ff_vk_shader_free(s, &dec->vert_interleave_shd[i]);

        ff_vk_pipeline_free(s, &dec->vert_wavelet_pl[i]);
        ff_vk_shader_free(s, &dec->vert_wavelet_shd[i]);

        ff_vk_pipeline_free(s, &dec->horiz_wavelet_pl[i]);
        ff_vk_shader_free(s, &dec->horiz_wavelet_shd[i]);
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
                                                int *nb_buf_bar,
                                                FFVkBuffer *buf) {
    buf_bar[(*nb_buf_bar)++] = (VkBufferMemoryBarrier2) {
        .sType = VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER_2,
        .srcStageMask = buf->stage,
        .dstStageMask = VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT,
        .srcAccessMask = buf->access,
        .dstAccessMask = VK_ACCESS_2_SHADER_STORAGE_READ_BIT,
        .srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
        .dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
        .buffer = buf->buf,
        .size = buf->size,
        .offset = 0,
    };
}

static av_always_inline inline void bar_write(VkBufferMemoryBarrier2 *buf_bar,
                                                int *nb_buf_bar,
                                                FFVkBuffer *buf) {
    buf_bar[(*nb_buf_bar)++] = (VkBufferMemoryBarrier2) {
        .sType = VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER_2,
        .srcStageMask = buf->stage,
        .dstStageMask = VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT,
        .srcAccessMask = buf->access,
        .dstAccessMask = VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT,
        .srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
        .dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
        .buffer = buf->buf,
        .size = buf->size,
        .offset = 0,
    };
}

static inline int alloc_tmp_bufs(DiracContext *ctx, DiracVulkanDecodeContext *dec) {
    int err, plane_size;

    plane_size = sizeof(int32_t) *
        (ctx->plane[0].idwt.width * ctx->plane[0].idwt.height) * 3;

    if (dec->tmp_buf.buf != NULL) {
        ff_vk_free_buf(&dec->vkctx, &dec->tmp_buf);
        ff_vk_free_buf(&dec->vkctx, &dec->tmp_interleave_buf);
    }

    err = ff_vk_create_buf(&dec->vkctx, &dec->tmp_buf, plane_size,
                             NULL,
                             NULL,
                            VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT |
                                VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
                            VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
    if (err < 0)
        return err;

    err = ff_vk_create_buf(&dec->vkctx, &dec->tmp_interleave_buf, plane_size,
                             NULL,
                             NULL,
                            VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT |
                                VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
                            VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
    if (err < 0)
        return err;

    return 0;
}

static inline int alloc_host_mapped_buf(DiracVulkanDecodeContext *dec, size_t req_size,
                                 void **mem, AVBufferRef **avbuf, FFVkBuffer **buf) {
    // FFVulkanFunctions *vk = &dec->vkctx.vkfn;
    // VkResult ret;
    int err;

    err = ff_vk_create_avbuf(&dec->vkctx, avbuf, req_size,
                             NULL,
                             NULL,
                            // &create_desc,
                            // &import_desc,
                            VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT |
                                VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
                            VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT |
                                VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);
    if (err < 0)
        return err;

    *buf = (FFVkBuffer*)(*avbuf)->data;
    err = ff_vk_map_buffer(&dec->vkctx, *buf,
                           (uint8_t **)mem, 0);
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
                                &dec->av_slice_buf,
                                &dec->slice_buf);
    if (err < 0)
        return err;

    err = ff_vk_set_descriptor_buffer(&dec->vkctx, &dec->quant_pl,
                                    NULL, 1, 2, 0,
                                    dec->slice_buf->address,
                                    dec->slice_buf->size,
                                    VK_FORMAT_UNDEFINED);
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
                                &dec->av_quant_buf,
                                &dec->quant_buf);
    if (err < 0)
        return err;

    err = ff_vk_set_descriptor_buffer(&dec->vkctx, &dec->quant_pl,
                                    NULL, 1, 1, 0,
                                    dec->quant_buf->address,
                                    dec->quant_buf->size,
                                    VK_FORMAT_UNDEFINED);
    if (err < 0)
        return err;

    return 0;
}

static int subband_coeffs(const DiracContext *s, int x, int y, int p, int off,
                          SliceCoeffVk *c)
{
    int level, coef = 0;
    for (level = 0; level <= s->wavelet_depth; level++) {
        SliceCoeffVk *o = &c[level];
        const SubBand *b = &s->plane[p].band[level][3]; /* orientation doens't matter */
        o->top   = b->height * y / s->num_y;
        o->left  = b->width  * x / s->num_x;
        o->tot_h = ((b->width  * (x + 1)) / s->num_x) - o->left;
        o->tot_v = ((b->height * (y + 1)) / s->num_y) - o->top;
        o->tot   = o->tot_h * o->tot_v;
        o->offs  = off + coef;
        coef    += o->tot * (4 - !!level);
    }
    return coef;
}

static int alloc_quant_buf(DiracContext *ctx, DiracVulkanDecodeContext *dec) {
    int err, length = ctx->num_y * ctx->num_x, coef_buf_size;
    SliceCoeffVk tmp[MAX_DWT_LEVELS];
    coef_buf_size = subband_coeffs(ctx, ctx->num_x - 1, ctx->num_y - 1, 0, 0, tmp) + 8;
    coef_buf_size = coef_buf_size + 512;
    dec->slice_vals_size = coef_buf_size;
    coef_buf_size *= sizeof(int32_t);

    if (dec->quant_val_buf_vk_ptr) {
        av_buffer_unref(&dec->av_quant_val_buf);
    }

    dec->thread_buf_size = coef_buf_size;

    dec->quant_val_buf_size = dec->thread_buf_size * 3 * length;
    err = alloc_host_mapped_buf(dec, dec->quant_val_buf_size,
                                (void **)&dec->quant_val_buf_vk_ptr,
                                &dec->av_quant_val_buf,
                                &dec->quant_val_buf);
    if (err < 0)
        return err;

    err = ff_vk_set_descriptor_buffer(&dec->vkctx, &dec->quant_pl,
                                    NULL, 1, 0, 0,
                                    dec->quant_val_buf->address,
                                    dec->quant_val_buf->size,
                                    VK_FORMAT_UNDEFINED);
    if (err < 0)
        return err;

    return 0;
}

/* ----- Copy Shader init and pipeline pass ----- */

static int init_cpy_shd(DiracVulkanDecodeContext *s, FFVkSPIRVCompiler *spv)
{
    int err = 0;
    uint8_t *spv_data;
    size_t spv_len;
    void *spv_opaque = NULL;
    const int planes = av_pix_fmt_count_planes(s->vkctx.output_format);
    FFVulkanContext *vkctx = &s->vkctx;
    FFVulkanDescriptorSetBinding *desc;
    FFVkSPIRVShader *shd = &s->cpy_to_image_shd;
    FFVulkanPipeline *pl = &s->cpy_to_image_pl;
    FFVkExecPool *exec = &s->exec_pool;

    RET(ff_vk_shader_init(pl, shd, "cpy_to_image", VK_SHADER_STAGE_COMPUTE_BIT, 0));

    shd = &s->cpy_to_image_shd;
    ff_vk_shader_set_compute_sizes(shd, 8, 8, 3);

    GLSLC(0, #extension GL_EXT_debug_printf : enable);
    GLSLC(0, #extension GL_EXT_scalar_block_layout : enable);
    GLSLC(0, #extension GL_EXT_shader_explicit_arithmetic_types : enable);

    desc = (FFVulkanDescriptorSetBinding[])
    {
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
          .mem_layout = "rgba32f",
          .mem_quali = "writeonly",
          .dimensions = 2,
          .elems = planes,
        },
    };
    RET(ff_vk_pipeline_descriptor_set_add(vkctx, pl, shd, desc, 2, 0, 0));

    ff_vk_add_push_constant(pl, 0, sizeof(WaveletPushConst), VK_SHADER_STAGE_COMPUTE_BIT);

    GLSLC(0, layout(push_constant, std430) uniform pushConstants {  );
    GLSLC(1,     ivec2 plane_sizes[3];                              );
    GLSLC(1,     int plane_offs[3];                                 );
    GLSLC(1,     int plane_strides[3];                              );
    GLSLC(1,     int dw[3];                                         );
    GLSLC(1,     int wavelet_depth;                                 );
    GLSLC(0, };                                                     );
    GLSLC(0,                                                        );

    GLSLC(0, void main() {                                                              );
    GLSLC(1,    int y = int(gl_GlobalInvocationID.y);                                   );
    GLSLC(1,    int plane = int(gl_GlobalInvocationID.z);                               );
    GLSLC(1,    ivec2 iSize = imageSize(out_img[plane]);                                );
    GLSLC(1,    int off_y = int(gl_WorkGroupSize.y * gl_NumWorkGroups.y);               );
    GLSLC(1,    int off_x = int(gl_WorkGroupSize.x * gl_NumWorkGroups.x);               );
    GLSLC(1,                                                                            );
    GLSLC(1,    for (; y < uint(iSize.y); y += off_y) {                                 );
    GLSLC(2,        int x = int(gl_GlobalInvocationID.x);                               );
    GLSLC(2,        for (; x < uint(iSize.x); x += off_x) {                             );
    GLSLC(3,            int idx = plane_offs[plane] + y * plane_strides[plane] + x;     );
    GLSLC(3,            float val = mod(float(inBuf[idx] + 128), 256.0) / 255.0;        );
    GLSLC(3,            imageStore(out_img[plane], ivec2(x, y), vec4(val));             );
    GLSLC(2,        }                                                                   );
    GLSLC(1,    }                                                                       );
    GLSLC(0, }                                                                          );

    RET(spv->compile_shader(spv, vkctx, shd, &spv_data, &spv_len, "main", &spv_opaque));
    RET(ff_vk_shader_create(vkctx, shd, spv_data, spv_len, "main"));
    RET(ff_vk_init_compute_pipeline(vkctx, pl, shd));
    RET(ff_vk_exec_pipeline_register(vkctx, exec, pl));

fail:
    if (spv_opaque)
        spv->free_shader(spv, &spv_opaque);

    return err;
}

static av_always_inline int inline cpy_to_image_pass(DiracVulkanDecodeContext *dec,
                          DiracContext *ctx,
                          FFVkExecContext *exec,
                          VkImageView *views,
                          VkBufferMemoryBarrier2 *buf_bar,
                          int *nb_buf_bar,
                          VkImageMemoryBarrier2 *img_bar,
                          int *nb_img_bar) {
    int err, prev_nb_bar = *nb_buf_bar, prev_nb_img_bar = *nb_img_bar;
    FFVulkanFunctions *vk = &dec->vkctx.vkfn;
    DiracVulkanDecodePicture *pic = ctx->hwaccel_picture_private;

    err = ff_vk_set_descriptor_buffer(&dec->vkctx, &dec->cpy_to_image_pl,
                                    exec, 0, 0, 0,
                                    dec->tmp_buf.address,
                                    dec->tmp_buf.size,
                                    VK_FORMAT_UNDEFINED);
    if (err < 0)
        return err;

    ff_vk_update_descriptor_img_array(&dec->vkctx, &dec->cpy_to_image_pl,
                                      exec, pic->frame->avframe, views, 0, 1,
                                      VK_IMAGE_LAYOUT_GENERAL,
                                      dec->sampler);

    dec->pConst.real_plane_dims[0] = ctx->plane[0].idwt.width;
    dec->pConst.real_plane_dims[1] = ctx->plane[0].idwt.height;
    dec->pConst.real_plane_dims[2] = ctx->plane[1].idwt.width;
    dec->pConst.real_plane_dims[3] = ctx->plane[1].idwt.height;
    dec->pConst.real_plane_dims[4] = ctx->plane[2].idwt.width;
    dec->pConst.real_plane_dims[5] = ctx->plane[2].idwt.height;

    dec->pConst.plane_strides[0] = ctx->plane[0].idwt.stride >> (1 + ctx->pshift);
    dec->pConst.plane_strides[1] = ctx->plane[1].idwt.stride >> (1 + ctx->pshift);
    dec->pConst.plane_strides[2] = ctx->plane[2].idwt.stride >> (1 + ctx->pshift);

    dec->pConst.plane_offs[0] = 0;
    dec->pConst.plane_offs[1] = ctx->plane[0].idwt.width * ctx->plane[0].idwt.height;
    dec->pConst.plane_offs[2] = 2 * dec->pConst.plane_offs[1];

    ff_vk_update_push_exec(&dec->vkctx, exec, &dec->cpy_to_image_pl,
                           VK_SHADER_STAGE_COMPUTE_BIT,
                           0, sizeof(WaveletPushConst), &dec->pConst);

    bar_read(buf_bar, nb_buf_bar, &dec->tmp_buf);
    bar_write(buf_bar, nb_buf_bar, &dec->tmp_buf);
    bar_read(buf_bar, nb_buf_bar, &dec->tmp_interleave_buf);
    bar_write(buf_bar, nb_buf_bar, &dec->tmp_interleave_buf);

    ff_vk_frame_barrier(&dec->vkctx, exec, pic->frame->avframe,
                        img_bar, nb_img_bar,
                        VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT,
                        VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT,
                        VK_ACCESS_SHADER_READ_BIT,
                        VK_IMAGE_LAYOUT_GENERAL,
                        VK_QUEUE_FAMILY_IGNORED);

    ff_vk_frame_barrier(&dec->vkctx, exec, pic->frame->avframe,
                        img_bar, nb_img_bar,
                        VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT,
                        VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT,
                        VK_ACCESS_SHADER_WRITE_BIT,
                        VK_IMAGE_LAYOUT_GENERAL,
                        VK_QUEUE_FAMILY_IGNORED);

    ff_vk_exec_bind_pipeline(&dec->vkctx, exec, &dec->cpy_to_image_pl);

    vk->CmdPipelineBarrier2(exec->buf, &(VkDependencyInfo) {
            .sType = VK_STRUCTURE_TYPE_DEPENDENCY_INFO,
            .pBufferMemoryBarriers = buf_bar + prev_nb_bar,
            .bufferMemoryBarrierCount = *nb_buf_bar - prev_nb_bar,
            .pImageMemoryBarriers = img_bar + prev_nb_img_bar,
            .imageMemoryBarrierCount = *nb_img_bar - prev_nb_img_bar,
        });


    vk->CmdDispatch(exec->buf,
                    ctx->plane[0].width >> 5,
                    ctx->plane[0].height >> 5,
                    1);

    prev_nb_img_bar = *nb_img_bar;
    ff_vk_frame_barrier(&dec->vkctx, exec, pic->frame->avframe,
                        img_bar, nb_img_bar,
                        VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT,
                        VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT,
                        VK_ACCESS_SHADER_READ_BIT,
                        VK_IMAGE_LAYOUT_GENERAL,
                        VK_QUEUE_FAMILY_IGNORED);

    ff_vk_frame_barrier(&dec->vkctx, exec, pic->frame->avframe,
                        img_bar, nb_img_bar,
                        VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT,
                        VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT,
                        VK_ACCESS_SHADER_WRITE_BIT,
                        VK_IMAGE_LAYOUT_GENERAL,
                        VK_QUEUE_FAMILY_IGNORED);

    vk->CmdPipelineBarrier2(exec->buf, &(VkDependencyInfo) {
            .sType = VK_STRUCTURE_TYPE_DEPENDENCY_INFO,
            .pImageMemoryBarriers = img_bar + prev_nb_img_bar,
            .imageMemoryBarrierCount = *nb_img_bar - prev_nb_img_bar,
        });

    return 0;
}

/* ----- LeGall Wavelet init and pipeline pass ----- */

static const char get_idx[] = {
      C(0, int getIdx(int plane, int x, int y) {                            )
      C(1,      return plane_offs[plane] + plane_strides[plane] * y + x;    )
      C(0, }                                                                )
};

static const char legall_low_y[] = {
    C(0, int32_t legall_low_y(int plane, int x, int y) {                    )
    C(1,    const int h = plane_sizes[plane].y;                             )
    C(1,                                                                    )
    C(1,    const int y_1 = int((y - 1) >= 0) * (y - 1) + int((y - 1) < 0); )
    C(1,    const int32_t val_1 = inBuf[getIdx(plane, x, y_1)];             )
    C(1,    const int y0 = y;                                               )
    C(1,    const int32_t val0 = inBuf[getIdx(plane, x, y0)];               )
    C(1,    const int y1 = y + 1;                                           )
    C(1,    const int32_t val1 = inBuf[getIdx(plane, x, y1)];               )
    C(1,    return val0 - ((val1 + val_1 + 2) >> 2);                        )
    C(0, }                                                                  )
};

static const char legall_high[] = {
      C(0, int32_t legall_high(int32_t v1, int32_t v2, int32_t v3) {        )
      C(1,      return v1 + ((v2 + v3 + 1) >> 1);                           )
      C(0, }                                                                )
};

static const char legall_vert[] = {
    C(0, void idwt_vert(int plane, int x, int y) {                                      )
    C(1,    const int h = plane_sizes[plane].y;                                         )
    C(1,                                                                                )
    C(1,    const int32_t out0 = legall_low_y(plane, x, y);                             )
    C(1,    const int32_t tmp1 = int((y + 2) < h) * legall_low_y(plane, x, y + 2) +     )
    C(1,                            int((y + 2) >= h) * out0;                           )
    C(1,                                                                                )
    C(1,    const int y1 = y + 1;                                                       )
    C(1,    const int32_t val1 = inBuf[getIdx(plane, x, y1)];                           )
    C(1,                                                                                )
    C(1,    const int32_t out1 = legall_high(val1, out0, tmp1);                         )
    C(1,                                                                                )
    C(1,    outBuf[getIdx(plane, x, y)]     = out0;                                     )
    C(1,    outBuf[getIdx(plane, x, y + 1)] = out1;                                     )
    C(0, }                                                                              )
};

static const char legall_low_x[] = {
    C(0, int32_t legall_low_x(int plane, int x, int y) {                    )
    C(1,    const int w = plane_sizes[plane].x;                             )
    C(1,                                                                    )
    C(1,    const int x_1 = int((x - 1) >= 0) * (x - 1) + int((x - 1) < 0); )
    C(1,    const int32_t val_1 = inBuf[getIdx(plane, x_1, y)];             )
    C(1,    const int x0 = x;                                               )
    C(1,    const int32_t val0 = inBuf[getIdx(plane, x0, y)];               )
    C(1,    const int x1 = x + 1;                                           )
    C(1,    const int32_t val1 = inBuf[getIdx(plane, x1, y)];               )
    C(1,    return val0 - ((val1 + val_1 + 2) >> 2);                        )
    C(0, }                                                                  )
};

static const char legall_horiz[] = {
    C(0, void idwt_horiz(int plane, int x, int y) {                                     )
    C(1,    const int w = plane_sizes[plane].x;                                         )
    C(1,                                                                                )
    C(1,    const int32_t out0 = legall_low_x(plane, x, y);                             )
    C(1,    const int32_t tmp1 = int((x + 2) < 2) * legall_low_x(plane, x + 2, y) +     )
    C(1,                            int((x + 2) <= w) * out0;                           )
    C(1,                                                                                )
    C(1,    const int x1 = x + 1;                                                       )
    C(1,    const int32_t val1 = inBuf[getIdx(plane, x1, y)];                           )
    C(1,                                                                                )
    C(1,    const int32_t out1 = legall_high(val1, out0, tmp1);                         )
    C(1,                                                                                )
    C(1,    outBuf[getIdx(plane, 2 * x, y)]     = (out0 + 1) >> 1;                      )
    C(1,    outBuf[getIdx(plane, 2 * x + 1, y)] = (out1 + 1) >> 1;                      )
    C(0, }                                                                              )
};

static int init_wavelet_shd_legall_vert(DiracVulkanDecodeContext *s, FFVkSPIRVCompiler *spv)
{
    int err = 0;
    uint8_t *spv_data;
    size_t spv_len;
    void *spv_opaque = NULL;
    int wavelet_idx = DWT_DIRAC_LEGALL5_3;
    FFVulkanContext *vkctx = &s->vkctx;
    FFVulkanDescriptorSetBinding *desc;
    FFVkSPIRVShader *shd = &s->vert_wavelet_shd[wavelet_idx];
    FFVulkanPipeline *pl = &s->vert_wavelet_pl[wavelet_idx];
    FFVkExecPool *exec = &s->exec_pool;

    RET(ff_vk_shader_init(pl, shd, "legall_vert", VK_SHADER_STAGE_COMPUTE_BIT, 0));

    shd = &s->vert_wavelet_shd[wavelet_idx];
    ff_vk_shader_set_compute_sizes(shd, 8, 8, 1);

    GLSLC(0, #extension GL_EXT_scalar_block_layout : enable);
    GLSLC(0, #extension GL_EXT_shader_explicit_arithmetic_types : enable);

    desc = (FFVulkanDescriptorSetBinding[])
    {
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
    RET(ff_vk_pipeline_descriptor_set_add(vkctx, pl, shd, desc, 2, 0, 0));

    ff_vk_add_push_constant(pl, 0, sizeof(WaveletPushConst), VK_SHADER_STAGE_COMPUTE_BIT);

    GLSLC(0, layout(push_constant, std430) uniform pushConstants {  );
    GLSLC(1,     ivec2 plane_sizes[3];                              );
    GLSLC(1,     int plane_offs[3];                                 );
    GLSLC(1,     int plane_strides[3];                              );
    GLSLC(1,     int dw[3];                                         );
    GLSLC(1,     int wavelet_depth;                                 );
    GLSLC(0, };                                                     );
    GLSLC(0,                                                        );

    GLSLD(get_idx);
    GLSLD(legall_low_y);
    GLSLD(legall_high);
    GLSLD(legall_vert);

    GLSLC(0, void main() {                                                                  );
    GLSLC(1,    int off_y = int(gl_WorkGroupSize.y * gl_NumWorkGroups.y);                   );
    GLSLC(1,    int off_x = int(gl_WorkGroupSize.x * gl_NumWorkGroups.x);                   );
    GLSLC(1,                                                                                );
    GLSLC(1,    for (int pic_z = 0; pic_z < 3; pic_z++) {                                   );
    GLSLC(2,        uint h = int(plane_sizes[pic_z].y / 2);                                 );
    GLSLC(2,        uint w = int(plane_sizes[pic_z].x);                                     );
    GLSLC(2,                                                                                );
    GLSLC(2,        int x = int(gl_GlobalInvocationID.x);                                   );
    GLSLC(2,        for (; x < w; x += off_x) {                                             );
    GLSLC(3,            int y = int(gl_GlobalInvocationID.y);                               );
    GLSLC(3,            for (; y < h; y += off_y) {                                         );
    GLSLC(4,                idwt_vert(pic_z, x, 2 * y);                                     );
    GLSLC(3,            }                                                                   );
    GLSLC(2,        }                                                                       );
    GLSLC(1,    }                                                                           );
    GLSLC(0, }                                                                              );

    RET(spv->compile_shader(spv, vkctx, shd, &spv_data, &spv_len, "main", &spv_opaque));
    RET(ff_vk_shader_create(vkctx, shd, spv_data, spv_len, "main"));
    RET(ff_vk_init_compute_pipeline(vkctx, pl, shd));
    RET(ff_vk_exec_pipeline_register(vkctx, exec, pl));

fail:
    if (spv_opaque)
        spv->free_shader(spv, &spv_opaque);

    return err;
}

static int init_wavelet_shd_legall_horiz(DiracVulkanDecodeContext *s, FFVkSPIRVCompiler *spv)
{
    int err = 0;
    uint8_t *spv_data;
    size_t spv_len;
    void *spv_opaque = NULL;
    int wavelet_idx = DWT_DIRAC_LEGALL5_3;
    FFVulkanContext *vkctx = &s->vkctx;
    FFVulkanDescriptorSetBinding *desc;
    FFVkSPIRVShader *shd = &s->horiz_wavelet_shd[wavelet_idx];
    FFVulkanPipeline *pl = &s->horiz_wavelet_pl[wavelet_idx];
    FFVkExecPool *exec = &s->exec_pool;

    RET(ff_vk_shader_init(pl, shd, "legall_horiz", VK_SHADER_STAGE_COMPUTE_BIT, 0));

    shd = &s->horiz_wavelet_shd[wavelet_idx];
    ff_vk_shader_set_compute_sizes(shd, 8, 8, 1);

    GLSLC(0, #extension GL_EXT_scalar_block_layout : enable);
    GLSLC(0, #extension GL_EXT_shader_explicit_arithmetic_types : enable);

    desc = (FFVulkanDescriptorSetBinding[])
    {
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
    RET(ff_vk_pipeline_descriptor_set_add(vkctx, pl, shd, desc, 2, 0, 0));

    ff_vk_add_push_constant(pl, 0, sizeof(WaveletPushConst), VK_SHADER_STAGE_COMPUTE_BIT);

    GLSLC(0, layout(push_constant, std430) uniform pushConstants {  );
    GLSLC(1,     ivec2 plane_sizes[3];                              );
    GLSLC(1,     int plane_offs[3];                                 );
    GLSLC(1,     int plane_strides[3];                              );
    GLSLC(1,     int dw[3];                                         );
    GLSLC(1,     int wavelet_depth;                                 );
    GLSLC(0, };                                                     );
    GLSLC(0,                                                        );

    GLSLD(get_idx);
    GLSLD(legall_low_x);
    GLSLD(legall_high);
    GLSLD(legall_horiz);

    GLSLC(0, void main() {                                                                  );
    GLSLC(1,    int off_y = int(gl_WorkGroupSize.y * gl_NumWorkGroups.y);                   );
    GLSLC(1,    int off_x = int(gl_WorkGroupSize.x * gl_NumWorkGroups.x);                   );
    GLSLC(1,                                                                                );
    GLSLC(1,    for (int pic_z = 0; pic_z < 3; pic_z++) {                                   );
    GLSLC(2,        uint h = int(plane_sizes[pic_z].y);                                     );
    GLSLC(2,        uint w = int(plane_sizes[pic_z].x / 2);                                 );
    GLSLC(1,                                                                                );
    GLSLC(2,        int y = int(gl_GlobalInvocationID.y);                                   );
    GLSLC(2,        for (; y < h; y += off_y) {                                             );
    GLSLC(3,            int x = int(gl_GlobalInvocationID.x);                               );
    GLSLC(3,            for (; x < w; x += off_x) {                                         );
    GLSLC(4,                idwt_horiz(pic_z, x, y);                                        );
    GLSLC(3,            }                                                                   );
    GLSLC(2,        }                                                                       );
    GLSLC(1,    }                                                                           );
    GLSLC(0, }                                                                              );

    RET(spv->compile_shader(spv, vkctx, shd, &spv_data, &spv_len, "main", &spv_opaque));
    RET(ff_vk_shader_create(vkctx, shd, spv_data, spv_len, "main"));
    RET(ff_vk_init_compute_pipeline(vkctx, pl, shd));
    RET(ff_vk_exec_pipeline_register(vkctx, exec, pl));

fail:
    if (spv_opaque)
        spv->free_shader(spv, &spv_opaque);

    return err;
}

static av_always_inline int inline wavelet_legall_pass(DiracVulkanDecodeContext *dec,
                          DiracContext *ctx,
                          FFVkExecContext *exec,
                          VkBufferMemoryBarrier2 *buf_bar,
                          int *nb_buf_bar) {
    int err;
    int barrier_num = *nb_buf_bar;
    int wavelet_idx = DWT_DIRAC_LEGALL5_3;
    FFVulkanFunctions *vk = &dec->vkctx.vkfn;

    FFVulkanPipeline *pl_hor = &dec->horiz_wavelet_pl[wavelet_idx];
    FFVulkanPipeline *pl_vert = &dec->vert_wavelet_pl[wavelet_idx];

    for (int i = ctx->wavelet_depth - 1; i >= 0; i--) {
        dec->pConst.plane_strides[0] = (ctx->plane[0].idwt.stride << i) >> (1 + ctx->pshift);
        dec->pConst.plane_strides[1] = (ctx->plane[1].idwt.stride << i) >> (1 + ctx->pshift);
        dec->pConst.plane_strides[2] = (ctx->plane[2].idwt.stride << i) >> (1 + ctx->pshift);

        dec->pConst.dw[0] = ctx->plane[0].idwt.width >> (i + 1);
        dec->pConst.dw[1] = ctx->plane[1].idwt.width >> (i + 1);
        dec->pConst.dw[2] = ctx->plane[2].idwt.width >> (i + 1);

        dec->pConst.real_plane_dims[0] = ((1 << i) + ctx->plane[0].idwt.width)  >> i;
        dec->pConst.real_plane_dims[1] = ((1 << i) + ctx->plane[0].idwt.height) >> i;
        dec->pConst.real_plane_dims[2] = ((1 << i) + ctx->plane[1].idwt.width)  >> i;
        dec->pConst.real_plane_dims[3] = ((1 << i) + ctx->plane[1].idwt.height) >> i;
        dec->pConst.real_plane_dims[4] = ((1 << i) + ctx->plane[2].idwt.width)  >> i;
        dec->pConst.real_plane_dims[5] = ((1 << i) + ctx->plane[2].idwt.height) >> i;

        /* Vertical wavelet pass */
        ff_vk_exec_bind_pipeline(&dec->vkctx, exec, pl_vert);
        ff_vk_update_push_exec(&dec->vkctx, exec, pl_vert,
                               VK_SHADER_STAGE_COMPUTE_BIT,
                               0, sizeof(WaveletPushConst), &dec->pConst);

        err = ff_vk_set_descriptor_buffer(&dec->vkctx, pl_vert, exec,
                                          0, 0, 0,
                                          dec->tmp_buf.address,
                                          dec->tmp_buf.size,
                                          VK_FORMAT_UNDEFINED);
        if (err < 0)
            goto fail;
        err = ff_vk_set_descriptor_buffer(&dec->vkctx, pl_vert, exec,
                                          0, 1, 0,
                                          dec->tmp_interleave_buf.address,
                                          dec->tmp_interleave_buf.size,
                                          VK_FORMAT_UNDEFINED);
        if (err < 0)
            goto fail;

        barrier_num = *nb_buf_bar;
        bar_read(buf_bar, nb_buf_bar, &dec->tmp_buf);
        bar_write(buf_bar, nb_buf_bar, &dec->tmp_buf);
        bar_read(buf_bar, nb_buf_bar, &dec->tmp_interleave_buf);
        bar_write(buf_bar, nb_buf_bar, &dec->tmp_interleave_buf);

        vk->CmdPipelineBarrier2(exec->buf, &(VkDependencyInfo) {
                .sType = VK_STRUCTURE_TYPE_DEPENDENCY_INFO,
                .pBufferMemoryBarriers = buf_bar + barrier_num,
                .bufferMemoryBarrierCount = *nb_buf_bar - barrier_num,
        });

        vk->CmdDispatch(exec->buf,
                        dec->pConst.real_plane_dims[0] >> 3,
                        dec->pConst.real_plane_dims[1] >> 4,
                        1);

        /* Horizontal wavelet pass */
        ff_vk_exec_bind_pipeline(&dec->vkctx, exec, pl_hor);
        ff_vk_update_push_exec(&dec->vkctx, exec, pl_hor,
                            VK_SHADER_STAGE_COMPUTE_BIT,
                            0, sizeof(WaveletPushConst), &dec->pConst);

        err = ff_vk_set_descriptor_buffer(&dec->vkctx, pl_hor, exec,
                                          0, 0, 0,
                                          dec->tmp_interleave_buf.address,
                                          dec->tmp_interleave_buf.size,
                                          VK_FORMAT_UNDEFINED);
        if (err < 0)
            goto fail;
        err = ff_vk_set_descriptor_buffer(&dec->vkctx, pl_hor, exec,
                                          0, 1, 0,
                                          dec->tmp_buf.address,
                                          dec->tmp_buf.size,
                                          VK_FORMAT_UNDEFINED);
        if (err < 0)
            goto fail;

        barrier_num = *nb_buf_bar;
        bar_read(buf_bar, nb_buf_bar, &dec->tmp_buf);
        bar_write(buf_bar, nb_buf_bar, &dec->tmp_buf);
        bar_read(buf_bar, nb_buf_bar, &dec->tmp_interleave_buf);
        bar_write(buf_bar, nb_buf_bar, &dec->tmp_interleave_buf);

        vk->CmdPipelineBarrier2(exec->buf, &(VkDependencyInfo) {
                .sType = VK_STRUCTURE_TYPE_DEPENDENCY_INFO,
                .pBufferMemoryBarriers = buf_bar + barrier_num,
                .bufferMemoryBarrierCount = *nb_buf_bar - barrier_num,
        });

        vk->CmdDispatch(exec->buf,
                        dec->pConst.real_plane_dims[0] >> 4,
                        dec->pConst.real_plane_dims[1] >> 3,
                        1);

        barrier_num = *nb_buf_bar;
        bar_read(buf_bar, nb_buf_bar, &dec->tmp_buf);
        bar_write(buf_bar, nb_buf_bar, &dec->tmp_buf);
        bar_read(buf_bar, nb_buf_bar, &dec->tmp_interleave_buf);
        bar_write(buf_bar, nb_buf_bar, &dec->tmp_interleave_buf);

        vk->CmdPipelineBarrier2(exec->buf, &(VkDependencyInfo) {
                .sType = VK_STRUCTURE_TYPE_DEPENDENCY_INFO,
                .pBufferMemoryBarriers = buf_bar + barrier_num,
                .bufferMemoryBarrierCount = *nb_buf_bar - barrier_num,
        });
    }

    return 0;
fail:
    ff_vk_exec_discard_deps(&dec->vkctx, exec);
    return err;
}


/* ----- Haar Wavelet init and pipeline pass ----- */

static const char haari_horiz[] = {
    C(0, void idwt_horiz(int plane, int x, int y) {                             )
    C(1,    int offs0 = plane_offs[plane] + plane_strides[plane] * y + x;       )
    C(1,    int offs1 = offs0 + plane_sizes[plane].x / 2;                       )
    C(1,    int outIdx = plane_offs[plane] + plane_strides[plane] * y + x * 2;  )
    C(1,    int32_t val_orig0 = inBuf[offs0];                                   )
    C(1,    int32_t val_orig1 = inBuf[offs1];                                   )
    C(1,    int32_t val_new0 = val_orig0 - ((val_orig1 + 1) >> 1);              )
    C(1,    int32_t val_new1 = val_orig1 + val_new0;                            )
    C(1,    outBuf[outIdx] = val_new0;                                          )
    C(1,    outBuf[outIdx + 1] = val_new1;                                      )
    C(0, }                                                                      )
};

static const char haari_shift_horiz[] = {
    C(0, void idwt_horiz(int plane, int x, int y) {                             )
    C(1,    int offs0 = plane_offs[plane] + plane_strides[plane] * y + x;       )
    C(1,    int offs1 = offs0 + plane_sizes[plane].x / 2;                       )
    C(1,    int outIdx = plane_offs[plane] + plane_strides[plane] * y + x * 2;  )
    C(1,    int32_t val_orig0 = inBuf[offs0];                                   )
    C(1,    int32_t val_orig1 = inBuf[offs1];                                   )
    C(1,    int32_t val_new0 = val_orig0 - ((val_orig1 + 1) >> 1);              )
    C(1,    int32_t val_new1 = val_orig1 + val_new0;                            )
    C(1,    outBuf[outIdx] = (val_new0 + 1) >> 1;                               )
    C(1,    outBuf[outIdx + 1] = (val_new1 + 1) >> 1;                           )
    C(0, }                                                                      )
};

static const char haari_vert[] = {
    C(0, void idwt_vert(int plane, int x, int y) {                              )
    C(1,    int offs0 = plane_offs[plane] + plane_strides[plane] * y + x;       )
    C(1,    int offs1 = plane_offs[plane] + plane_strides[plane] * (y + 1) + x; )
    C(2,    int32_t val_orig0 = inBuf[offs0];                                   )
    C(1,    int32_t val_orig1 = inBuf[offs1];                                   )
    C(1,    int32_t val_new0 = val_orig0 - ((val_orig1 + 1) >> 1);              )
    C(1,    int32_t val_new1 = val_orig1 + val_new0;                            )
    C(1,    outBuf[offs0] = val_new0;                                           )
    C(1,    outBuf[offs1] = val_new1;                                           )
    C(0, }                                                                      )
};

static int init_wavelet_shd_haari_vert(DiracVulkanDecodeContext *s, FFVkSPIRVCompiler *spv, int shift)
{
    int err = 0;
    uint8_t *spv_data;
    size_t spv_len;
    void *spv_opaque = NULL;
    int wavelet_idx = DWT_DIRAC_HAAR0 + shift;
    FFVulkanContext *vkctx = &s->vkctx;
    FFVulkanDescriptorSetBinding *desc;
    FFVkSPIRVShader *shd = &s->vert_wavelet_shd[wavelet_idx];
    FFVulkanPipeline *pl = &s->vert_wavelet_pl[wavelet_idx];
    FFVkExecPool *exec = &s->exec_pool;

    RET(ff_vk_shader_init(pl, shd, "haari_vert", VK_SHADER_STAGE_COMPUTE_BIT, 0));

    shd = &s->vert_wavelet_shd[wavelet_idx];
    ff_vk_shader_set_compute_sizes(shd, 8, 8, 1);

    GLSLC(0, #extension GL_EXT_scalar_block_layout : enable);
    GLSLC(0, #extension GL_EXT_shader_explicit_arithmetic_types : enable);

    desc = (FFVulkanDescriptorSetBinding[])
    {
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
    RET(ff_vk_pipeline_descriptor_set_add(vkctx, pl, shd, desc, 2, 0, 0));

    ff_vk_add_push_constant(pl, 0, sizeof(WaveletPushConst), VK_SHADER_STAGE_COMPUTE_BIT);

    GLSLC(0, layout(push_constant, std430) uniform pushConstants {  );
    GLSLC(1,     ivec2 plane_sizes[3];                              );
    GLSLC(1,     int plane_offs[3];                                 );
    GLSLC(1,     int plane_strides[3];                              );
    GLSLC(1,     int dw[3];                                         );
    GLSLC(1,     int wavelet_depth;                                 );
    GLSLC(0, };                                                     );
    GLSLC(0,                                                        );

    GLSLD(haari_vert);

    GLSLC(0, void main() {                                                                  );
    GLSLC(1,    int off_y = int(gl_WorkGroupSize.y * gl_NumWorkGroups.y);                   );
    GLSLC(1,    int off_x = int(gl_WorkGroupSize.x * gl_NumWorkGroups.x);                   );
    GLSLC(1,                                                                                );
    GLSLC(1,    for (int pic_z = 0; pic_z < 3; pic_z++) {                                   );
    GLSLC(2,        uint h = int(plane_sizes[pic_z].y / 2);                                 );
    GLSLC(2,        uint w = int(plane_sizes[pic_z].x);                                     );
    GLSLC(1,                                                                                );
    GLSLC(2,        int y = int(gl_GlobalInvocationID.y);                                   );
    GLSLC(2,        for (; y < h; y += off_y) {                                             );
    GLSLC(3,            int x = int(gl_GlobalInvocationID.x);                               );
    GLSLC(3,            for (; x < w; x += off_x) {                                         );
    GLSLC(4,                idwt_vert(pic_z, x, 2 * y);                                     );
    GLSLC(3,            }                                                                   );
    GLSLC(2,        }                                                                       );
    GLSLC(1,    }                                                                           );
    GLSLC(0, }                                                                              );

    RET(spv->compile_shader(spv, vkctx, shd, &spv_data, &spv_len, "main", &spv_opaque));
    RET(ff_vk_shader_create(vkctx, shd, spv_data, spv_len, "main"));
    RET(ff_vk_init_compute_pipeline(vkctx, pl, shd));
    RET(ff_vk_exec_pipeline_register(vkctx, exec, pl));

fail:
    if (spv_opaque)
        spv->free_shader(spv, &spv_opaque);

    return err;
}

static int init_wavelet_shd_haari_horiz(DiracVulkanDecodeContext *s, FFVkSPIRVCompiler *spv, int shift)
{
    int err = 0;
    uint8_t *spv_data;
    size_t spv_len;
    void *spv_opaque = NULL;
    int wavelet_idx = DWT_DIRAC_HAAR0 + shift;
    FFVulkanContext *vkctx = &s->vkctx;
    FFVulkanDescriptorSetBinding *desc;
    FFVkSPIRVShader *shd = &s->horiz_wavelet_shd[wavelet_idx];
    FFVulkanPipeline *pl = &s->horiz_wavelet_pl[wavelet_idx];
    FFVkExecPool *exec = &s->exec_pool;

    RET(ff_vk_shader_init(pl, shd, "haari_horiz", VK_SHADER_STAGE_COMPUTE_BIT, 0));

    shd = &s->horiz_wavelet_shd[wavelet_idx];
    ff_vk_shader_set_compute_sizes(shd, 8, 8, 1);

    GLSLC(0, #extension GL_EXT_scalar_block_layout : enable);
    GLSLC(0, #extension GL_EXT_shader_explicit_arithmetic_types : enable);

    desc = (FFVulkanDescriptorSetBinding[])
    {
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
    RET(ff_vk_pipeline_descriptor_set_add(vkctx, pl, shd, desc, 2, 0, 0));

    ff_vk_add_push_constant(pl, 0, sizeof(WaveletPushConst), VK_SHADER_STAGE_COMPUTE_BIT);

    GLSLC(0, layout(push_constant, std430) uniform pushConstants {  );
    GLSLC(1,     ivec2 plane_sizes[3];                              );
    GLSLC(1,     int plane_offs[3];                                 );
    GLSLC(1,     int plane_strides[3];                              );
    GLSLC(1,     int dw[3];                                         );
    GLSLC(1,     int wavelet_depth;                                 );
    GLSLC(0, };                                                     );
    GLSLC(0,                                                        );

    GLSLD(shift ? haari_shift_horiz : haari_horiz);

    GLSLC(0, void main() {                                                                  );
    GLSLC(1,    int off_y = int(gl_WorkGroupSize.y * gl_NumWorkGroups.y);                   );
    GLSLC(1,    int off_x = int(gl_WorkGroupSize.x * gl_NumWorkGroups.x);                   );
    GLSLC(1,                                                                                );
    GLSLC(1,    for (int pic_z = 0; pic_z < 3; pic_z++) {                                   );
    GLSLC(2,        uint h = int(plane_sizes[pic_z].y);                                     );
    GLSLC(2,        uint w = int(plane_sizes[pic_z].x / 2);                                 );
    GLSLC(1,                                                                                );
    GLSLC(2,        int y = int(gl_GlobalInvocationID.y);                                   );
    GLSLC(2,        for (; y < h; y += off_y) {                                             );
    GLSLC(3,            int x = int(gl_GlobalInvocationID.x);                               );
    GLSLC(3,            for (; x < w; x += off_x) {                                         );
    GLSLC(4,                idwt_horiz(pic_z, x, y);                                        );
    GLSLC(3,            }                                                                   );
    GLSLC(2,        }                                                                       );
    GLSLC(1,    }                                                                           );
    GLSLC(0, }                                                                              );

    RET(spv->compile_shader(spv, vkctx, shd, &spv_data, &spv_len, "main", &spv_opaque));
    RET(ff_vk_shader_create(vkctx, shd, spv_data, spv_len, "main"));
    RET(ff_vk_init_compute_pipeline(vkctx, pl, shd));
    RET(ff_vk_exec_pipeline_register(vkctx, exec, pl));

fail:
    if (spv_opaque)
        spv->free_shader(spv, &spv_opaque);

    return err;
}

static av_always_inline int inline wavelet_haari_pass(DiracVulkanDecodeContext *dec,
                          DiracContext *ctx,
                          FFVkExecContext *exec,
                          VkBufferMemoryBarrier2 *buf_bar,
                          int *nb_buf_bar, int shift) {
    int err;
    int barrier_num = *nb_buf_bar;
    const int wavelet_idx = DWT_DIRAC_HAAR0 + shift;
    FFVulkanFunctions *vk = &dec->vkctx.vkfn;

    FFVulkanPipeline *pl_hor = &dec->horiz_wavelet_pl[wavelet_idx];
    FFVulkanPipeline *pl_vert = &dec->vert_wavelet_pl[wavelet_idx];

    for (int i = ctx->wavelet_depth - 1; i >= 0; i--) {
        dec->pConst.plane_strides[0] = (ctx->plane[0].idwt.stride >> (1 + ctx->pshift)) << i;
        dec->pConst.plane_strides[1] = (ctx->plane[1].idwt.stride >> (1 + ctx->pshift)) << i;
        dec->pConst.plane_strides[2] = (ctx->plane[2].idwt.stride >> (1 + ctx->pshift)) << i;

        dec->pConst.dw[0] = ctx->plane[0].idwt.width >> (i + 1);
        dec->pConst.dw[1] = ctx->plane[1].idwt.width >> (i + 1);
        dec->pConst.dw[2] = ctx->plane[2].idwt.width >> (i + 1);

        dec->pConst.real_plane_dims[0] = ((1 << i) + ctx->plane[0].idwt.width)  >> i;
        dec->pConst.real_plane_dims[1] = ((1 << i) + ctx->plane[0].idwt.height) >> i;
        dec->pConst.real_plane_dims[2] = ((1 << i) + ctx->plane[1].idwt.width)  >> i;
        dec->pConst.real_plane_dims[3] = ((1 << i) + ctx->plane[1].idwt.height) >> i;
        dec->pConst.real_plane_dims[4] = ((1 << i) + ctx->plane[2].idwt.width)  >> i;
        dec->pConst.real_plane_dims[5] = ((1 << i) + ctx->plane[2].idwt.height) >> i;

        dec->pConst.plane_offs[0] = 0;
        dec->pConst.plane_offs[1] = ctx->plane[0].idwt.width * ctx->plane[0].idwt.height;
        dec->pConst.plane_offs[2] = 2 * dec->pConst.plane_offs[1];

        dec->pConst.wavelet_depth = ctx->wavelet_depth;

        /* Vertical wavelet pass */
        ff_vk_exec_bind_pipeline(&dec->vkctx, exec, pl_vert);
        ff_vk_update_push_exec(&dec->vkctx, exec, pl_vert,
                               VK_SHADER_STAGE_COMPUTE_BIT,
                               0, sizeof(WaveletPushConst), &dec->pConst);

        err = ff_vk_set_descriptor_buffer(&dec->vkctx, pl_vert, exec,
                                          0, 0, 0,
                                          dec->tmp_buf.address,
                                          dec->tmp_buf.size,
                                          VK_FORMAT_UNDEFINED);
        if (err < 0)
            goto fail;
        err = ff_vk_set_descriptor_buffer(&dec->vkctx, pl_vert, exec,
                                          0, 1, 0,
                                          dec->tmp_interleave_buf.address,
                                          dec->tmp_interleave_buf.size,
                                          VK_FORMAT_UNDEFINED);
        if (err < 0)
            goto fail;

        barrier_num = *nb_buf_bar;
        bar_read(buf_bar, nb_buf_bar, &dec->tmp_buf);
        bar_write(buf_bar, nb_buf_bar, &dec->tmp_buf);
        bar_read(buf_bar, nb_buf_bar, &dec->tmp_interleave_buf);
        bar_write(buf_bar, nb_buf_bar, &dec->tmp_interleave_buf);

        vk->CmdPipelineBarrier2(exec->buf, &(VkDependencyInfo) {
                .sType = VK_STRUCTURE_TYPE_DEPENDENCY_INFO,
                .pBufferMemoryBarriers = buf_bar + barrier_num,
                .bufferMemoryBarrierCount = *nb_buf_bar - barrier_num,
        });

        vk->CmdDispatch(exec->buf,
                        dec->pConst.real_plane_dims[0] >> 3,
                        dec->pConst.real_plane_dims[1] >> 4,
                        1);

        /* Horizontal wavelet pass */
        ff_vk_update_push_exec(&dec->vkctx, exec, pl_hor,
                            VK_SHADER_STAGE_COMPUTE_BIT,
                            0, sizeof(WaveletPushConst), &dec->pConst);

        err = ff_vk_set_descriptor_buffer(&dec->vkctx, pl_hor, exec,
                                          0, 0, 0,
                                          dec->tmp_interleave_buf.address,
                                          dec->tmp_interleave_buf.size,
                                          VK_FORMAT_UNDEFINED);
        if (err < 0)
            goto fail;
        err = ff_vk_set_descriptor_buffer(&dec->vkctx, pl_hor, exec,
                                          0, 1, 0,
                                          dec->tmp_buf.address,
                                          dec->tmp_buf.size,
                                          VK_FORMAT_UNDEFINED);
        if (err < 0)
            goto fail;

        ff_vk_exec_bind_pipeline(&dec->vkctx, exec, pl_hor);

        barrier_num = *nb_buf_bar;
        bar_read(buf_bar, nb_buf_bar, &dec->tmp_buf);
        bar_write(buf_bar, nb_buf_bar, &dec->tmp_buf);
        bar_read(buf_bar, nb_buf_bar, &dec->tmp_interleave_buf);
        bar_write(buf_bar, nb_buf_bar, &dec->tmp_interleave_buf);

        vk->CmdPipelineBarrier2(exec->buf, &(VkDependencyInfo) {
                .sType = VK_STRUCTURE_TYPE_DEPENDENCY_INFO,
                .pBufferMemoryBarriers = buf_bar + barrier_num,
                .bufferMemoryBarrierCount = *nb_buf_bar - barrier_num,
        });

        vk->CmdDispatch(exec->buf,
                        dec->pConst.real_plane_dims[0] >> 4,
                        dec->pConst.real_plane_dims[1] >> 3,
                        1);
    }

    barrier_num = *nb_buf_bar;
    bar_read(buf_bar, nb_buf_bar, &dec->tmp_buf);
    bar_write(buf_bar, nb_buf_bar, &dec->tmp_buf);
    bar_read(buf_bar, nb_buf_bar, &dec->tmp_interleave_buf);
    bar_write(buf_bar, nb_buf_bar, &dec->tmp_interleave_buf);


    return 0;
fail:
    ff_vk_exec_discard_deps(&dec->vkctx, exec);
    return err;
}

/* ----- Dequant Shader init and pipeline pass ----- */

static const char dequant[] = {
    C(0, void dequant(int outIdx, int idx, int qf, int qs) {        )
    C(1,    int32_t val = inBuffer[idx];                            )
    C(1,    val = sign(val) * ((abs(val) * qf + qs) >> 2);          )
    C(1,    outBuf0[outIdx] = outBuf1[outIdx] = val;                )
    C(0, }                                                          )
};

static const char proc_slice[] = {
    C(0, void proc_slice(int slice_idx) {                                                   )
    C(1,    const int plane = int(gl_GlobalInvocationID.x);                                 )
    C(1,    const int level = int(gl_GlobalInvocationID.y);                                 )
    C(1,    if (level >= wavelet_depth) return;                                             )
    C(1,    const int base_idx = slice_idx * DWT_LEVELS * 8;                                )
    C(1,    const int base_slice_idx = slice_idx * DWT_LEVELS * 3 + plane * DWT_LEVELS;     )
    C(1,                                                                                    )
    C(1,    const Slice s = slices[base_slice_idx + level];                                 )
    C(1,    int offs = s.offs;                                                              )
    C(1,                                                                                    )
    C(1,    for(int orient = int(level > 0); orient < 4; orient++) {                        )
    C(2,        int32_t qf = quantMatrix[base_idx + level * 8 + orient];                    )
    C(2,        int32_t qs = quantMatrix[base_idx + level * 8 + orient + 4];                )
    C(2,                                                                                    )
    C(2,        const int subband_idx = plane * DWT_LEVELS * 4                              )
    C(2,                                        + 4 * level + orient;                       )
    C(2,                                                                                    )
    C(2,        const SubbandOffset sub_off = subband_offs[subband_idx];                    )
    C(2,        int img_idx = plane_offs[plane] + sub_off.base_off                          )
    C(2,                                        + s.top * sub_off.stride + s.left;          )
    C(2,                                                                                    )
    C(2,        for(int y = 0; y < s.tot_v; y++) {                                          )
    C(3,            int img_x = img_idx;                                                    )
    C(3,            for(int x = 0; x < s.tot_h; x++) {                                      )
    C(4,                dequant(img_x, offs, qf, qs);                                       )
    C(4,                img_x++;                                                            )
    C(4,                offs++;                                                             )
    C(3,            }                                                                       )
    C(3,            img_idx += sub_off.stride;                                              )
    C(2,        }                                                                           )
    C(1,    }                                                                               )
    C(0, }                                                                                  )
};

static int init_quant_shd(DiracVulkanDecodeContext *s, FFVkSPIRVCompiler *spv)
{
    int err = 0;
    uint8_t *spv_data;
    size_t spv_len;
    void *spv_opaque = NULL;
    // const int planes = av_pix_fmt_count_planes(s->vkctx.output_format);
    FFVulkanContext *vkctx = &s->vkctx;
    FFVulkanDescriptorSetBinding *desc;
    FFVkSPIRVShader *shd = &s->quant_shd;
    FFVulkanPipeline *pl = &s->quant_pl;
    FFVkExecPool *exec = &s->exec_pool;

    RET(ff_vk_shader_init(pl, shd, "dequant", VK_SHADER_STAGE_COMPUTE_BIT, 0));

    shd = &s->quant_shd;
    ff_vk_shader_set_compute_sizes(shd, 3, MAX_DWT_LEVELS, 1);

    GLSLC(0, #extension GL_EXT_debug_printf : enable);
    GLSLC(0, #extension GL_EXT_scalar_block_layout : enable);
    GLSLC(0, #extension GL_EXT_shader_explicit_arithmetic_types : enable);

    desc = (FFVulkanDescriptorSetBinding[])
    {
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
    };
    RET(ff_vk_pipeline_descriptor_set_add(vkctx, pl, shd, desc, 2, 0, 0));

    GLSLC(0, struct Slice {         );
    GLSLC(1,     int32_t left;      );
    GLSLC(1,     int32_t top;       );
    GLSLC(1,     int32_t tot_h;     );
    GLSLC(1,     int32_t tot_v;     );
    GLSLC(1,     int32_t tot;       );
    GLSLC(1,     int32_t offs;      );
    GLSLC(1,     int32_t pad0;      );
    GLSLC(1,     int32_t pad1;      );
    GLSLC(0, };                     );

    GLSLC(0, struct SubbandOffset {     );
    GLSLC(1,     int32_t base_off;      );
    GLSLC(1,     int32_t stride;        );
    GLSLC(1,     int32_t pad0;          );
    GLSLC(1,     int32_t pad1;          );
    GLSLC(0, };                         );

    desc = (FFVulkanDescriptorSetBinding[])
    {
        {
            .name = "quant_in_buf",
            .type = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
            .stages = VK_SHADER_STAGE_COMPUTE_BIT,
            .buf_content = "int32_t inBuffer[];",
            .mem_quali = "readonly",
            .mem_layout = "std430",
        },
        {
            .name = "quant_vals_buf",
            .type = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
            .stages = VK_SHADER_STAGE_COMPUTE_BIT,
            .buf_content = "int32_t quantMatrix[];",
            .mem_quali = "readonly",
            .mem_layout = "std430",
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
    RET(ff_vk_pipeline_descriptor_set_add(vkctx, pl, shd, desc, 4, 1, 0));

    ff_vk_add_push_constant(pl, 0, sizeof(WaveletPushConst), VK_SHADER_STAGE_COMPUTE_BIT);

    GLSLC(0, layout(push_constant, std430) uniform pushConstants {  );
    GLSLC(1,     ivec2 plane_sizes[3];                              );
    GLSLC(1,     int plane_offs[3];                                 );
    GLSLC(1,     int plane_strides[3];                              );
    GLSLC(1,     int dw[3];                                         );
    GLSLC(1,     int wavelet_depth;                                 );
    GLSLC(0, };                                                     );
    GLSLC(0,                                                        );

    GLSLF(0, #define DWT_LEVELS %i, MAX_DWT_LEVELS                  );

    GLSLD(dequant);
    GLSLD(proc_slice);
    GLSLC(0, void main()                                                            );
    GLSLC(0, {                                                                      );
    GLSLC(1,    int idx = int(gl_GlobalInvocationID.z);                             );
    GLSLC(1,    proc_slice(idx);                                                    );
    GLSLC(0, }                                                                      );

    RET(spv->compile_shader(spv, vkctx, shd, &spv_data, &spv_len, "main", &spv_opaque));
    RET(ff_vk_shader_create(vkctx, shd, spv_data, spv_len, "main"));
    RET(ff_vk_init_compute_pipeline(vkctx, pl, shd));
    RET(ff_vk_exec_pipeline_register(vkctx, exec, pl));

fail:
    if (spv_opaque)
        spv->free_shader(spv, &spv_opaque);

    return err;
}

static av_always_inline int inline quant_pl_pass(DiracVulkanDecodeContext *dec,
                          DiracContext *ctx,
                          FFVkExecContext *exec,
                          VkBufferMemoryBarrier2 *buf_bar,
                          int *nb_buf_bar) {
    int err;
    FFVulkanFunctions *vk = &dec->vkctx.vkfn;

    ff_vk_exec_bind_pipeline(&dec->vkctx, exec, &dec->quant_pl);

    err = ff_vk_set_descriptor_buffer(&dec->vkctx, &dec->quant_pl,
                                    exec, 0, 0, 0,
                                    dec->tmp_buf.address,
                                    dec->tmp_buf.size,
                                    VK_FORMAT_UNDEFINED);
    if (err < 0)
        return err;

    err = ff_vk_set_descriptor_buffer(&dec->vkctx, &dec->quant_pl,
                                    exec, 0, 1, 0,
                                    dec->tmp_interleave_buf.address,
                                    dec->tmp_interleave_buf.size,
                                    VK_FORMAT_UNDEFINED);
    if (err < 0)
        return err;

    ff_vk_update_push_exec(&dec->vkctx, exec, &dec->quant_pl,
                           VK_SHADER_STAGE_COMPUTE_BIT,
                           0, sizeof(WaveletPushConst), &dec->pConst);

    bar_read(buf_bar, nb_buf_bar, dec->quant_val_buf);
    bar_read(buf_bar, nb_buf_bar, dec->slice_buf);
    bar_read(buf_bar, nb_buf_bar, dec->quant_buf);
    bar_read(buf_bar, nb_buf_bar, &dec->subband_info);

    bar_write(buf_bar, nb_buf_bar, &dec->tmp_buf);
    bar_write(buf_bar, nb_buf_bar, &dec->tmp_interleave_buf);

    vk->CmdPipelineBarrier2(exec->buf, &(VkDependencyInfo) {
            .sType = VK_STRUCTURE_TYPE_DEPENDENCY_INFO,
            .pBufferMemoryBarriers = buf_bar,
            .bufferMemoryBarrierCount = *nb_buf_bar,
        });

    vk->CmdDispatch(exec->buf, 1, 1, ctx->num_x * ctx->num_y);

    return 0;
}

static int vulkan_dirac_uninit(AVCodecContext *avctx) {
    // DiracContext *d = avctx->priv_data;
    // if (d->hwaccel_picture_private) {
    //     av_freep(d->hwaccel_picture_private);
    // }

    free_common(avctx);

    return 0;
}

static inline int wavelet_init(DiracVulkanDecodeContext *dec,
                                FFVkSPIRVCompiler *spv) {
    int err;

    err = init_wavelet_shd_haari_vert(dec, spv, 0);
    if (err < 0) {
        return err;
    }

    err = init_wavelet_shd_haari_horiz(dec, spv, 0);
    if (err < 0) {
        return err;
    }

    err = init_wavelet_shd_haari_vert(dec, spv, 1);
    if (err < 0) {
        return err;
    }

    err = init_wavelet_shd_haari_horiz(dec, spv, 1);
    if (err < 0) {
        return err;
    }

    err = init_wavelet_shd_legall_vert(dec, spv);
    if (err < 0) {
        return err;
    }

    err = init_wavelet_shd_legall_horiz(dec, spv);
    if (err < 0) {
        return err;
    }

    return 0;
}

static int vulkan_dirac_init(AVCodecContext *avctx)
{
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

    s->frames_ref = av_buffer_ref(avctx->hw_frames_ctx);
    s->frames = (AVHWFramesContext *)s->frames_ref->data;
    s->hwfc = s->frames->hwctx;

    s->device = (AVHWDeviceContext *)s->frames->device_ref->data;
    s->hwctx = s->device->hwctx;

    err = ff_vk_load_props(s);
    if (err < 0)
        goto fail;

    /* Create queue context */
    ff_vk_qf_init(s, &dec->qf, VK_QUEUE_COMPUTE_BIT);

    err = ff_vk_exec_pool_init(s, &dec->qf, &dec->exec_pool, 1, 0, 0, 0, NULL);

    err = ff_vk_init_sampler(&dec->vkctx, &dec->sampler, 1, VK_FILTER_LINEAR);
    if (err < 0) {
        goto fail;
    }

    av_log(avctx, AV_LOG_VERBOSE, "Vulkan decoder initialization sucessful\n");

    err = init_quant_shd(dec, spv);
    if (err < 0) {
        goto fail;
    }

    err = init_cpy_shd(dec, spv);
    if (err < 0) {
        goto fail;
    }

    err = wavelet_init(dec, spv);
    if (err < 0) {
      goto fail;
    }

    dec->quant_val_buf_vk_ptr = NULL;
    dec->slice_buf_vk_ptr     = NULL;
    dec->quant_buf_vk_ptr     = NULL;

    dec->av_quant_val_buf = NULL;
    dec->av_quant_buf     = NULL;
    dec->av_slice_buf     = NULL;

    dec->thread_buf_size = 0;
    dec->n_slice_bufs    = 0;

    err = ff_vk_create_buf(&dec->vkctx, &dec->subband_info,
                         sizeof(SubbandOffset) * MAX_DWT_LEVELS * 12, NULL, NULL,
                         VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT |
                         VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT,
                         VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT |
                            VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
    if (err < 0)
        return err;

    err = ff_vk_map_buffer(&dec->vkctx, &dec->subband_info,
                           (uint8_t **)&dec->subband_info_ptr, 0);
    if (err < 0)
        return err;

    err = ff_vk_set_descriptor_buffer(&dec->vkctx, &dec->quant_pl,
                                    NULL, 1, 3, 0,
                                    dec->subband_info.address,
                                    dec->subband_info.size,
                                    VK_FORMAT_UNDEFINED);
    if (err < 0)
        return err;

    return 0;

fail:
    if (spv)
    {
        spv->uninit(&spv);
    }
    vulkan_dirac_uninit(avctx);

    return err;
}

static int vulkan_decode_bootstrap(AVCodecContext *avctx, AVBufferRef *frames_ref)
{
    int err;
    DiracVulkanDecodeContext *dec = avctx->internal->hwaccel_priv_data;
    AVHWFramesContext *frames = (AVHWFramesContext *)frames_ref->data;
    AVHWDeviceContext *device = (AVHWDeviceContext *)frames->device_ref->data;
    AVVulkanDeviceContext *hwctx = device->hwctx;

    dec->vkctx.extensions = ff_vk_extensions_to_mask(hwctx->enabled_dev_extensions,
                                                 hwctx->nb_enabled_dev_extensions);

    err = ff_vk_load_functions(device, &dec->vkctx.vkfn, dec->vkctx.extensions, 1, 1);
    if (err < 0) {
        free_common(avctx);
        return err;
    }

    return 0;
}

static int vulkan_dirac_frame_params(AVCodecContext *avctx, AVBufferRef *hw_frames_ctx)
{
    int err;
    AVHWFramesContext *frames_ctx = (AVHWFramesContext*)hw_frames_ctx->data;
    AVVulkanFramesContext *hwfc = frames_ctx->hwctx;
    DiracContext *s = avctx->priv_data;

    frames_ctx->sw_format = s->sof_pix_fmt;

    err = vulkan_decode_bootstrap(avctx, hw_frames_ctx);
    if (err < 0)
        return err;

    frames_ctx->width  = avctx->coded_width;
    frames_ctx->height = avctx->coded_height;
    frames_ctx->format = AV_PIX_FMT_VULKAN;

    for (int i = 0; i < AV_NUM_DATA_POINTERS; i++) {
        hwfc->format[i]    = av_vkfmt_from_pixfmt(frames_ctx->sw_format)[i];
    }
    hwfc->tiling       = VK_IMAGE_TILING_OPTIMAL;
    hwfc->usage        = VK_IMAGE_USAGE_SAMPLED_BIT |
                         VK_IMAGE_USAGE_TRANSFER_DST_BIT |
                         VK_IMAGE_USAGE_TRANSFER_SRC_BIT |
                         VK_IMAGE_USAGE_STORAGE_BIT;

    return err;
}


static void vulkan_dirac_free_frame_priv(FFRefStructOpaque _hwctx, void *data)
{
    // AVHWDeviceContext *hwctx = _hwctx.nc;
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
        int s = p->idwt.stride >> (1 + ctx->pshift);
        /*int s = FFALIGN(p->idwt.width, 8);*/

        for (int level = ctx->wavelet_depth - 1; level >= 0; level--) {
            w >>= 1;
            s <<= 1;
            for (int orient = !!level; orient < 4; orient++) {
                const int idx = plane * MAX_DWT_LEVELS * 4 + level * 4 + orient;
                SubbandOffset *off = &offs[idx];
                off->stride = s;
                off->base_off = 0;

                if (orient & 1)
                    off->base_off += w;
                if (orient > 1)
                    off->base_off += (s>>1);

                /*SubBand *b = &p->band[level][orient];*/
                /*int w = (b->ibuf - p->idwt.buf) >> (1 + b->pshift);*/
                /*off->stride = b->stride >> (1 + b->pshift);*/
                /*off->base_off = w;*/
            }
        }
    }
}

static int vulkan_dirac_start_frame(AVCodecContext          *avctx,
                               av_unused const uint8_t *buffer,
                               av_unused uint32_t       size)
{
    int err;
    DiracVulkanDecodeContext *s = avctx->internal->hwaccel_priv_data;
    DiracContext *c = avctx->priv_data;
    DiracVulkanDecodePicture *pic = c->hwaccel_picture_private;
    WaveletPushConst *pConst = &s->pConst;

    pic->frame = c->current_picture;
    setup_subbands(c, s);

    pConst->real_plane_dims[0] = c->plane[0].width;
    pConst->real_plane_dims[1] = c->plane[0].height;
    pConst->real_plane_dims[2] = c->plane[1].width;
    pConst->real_plane_dims[3] = c->plane[1].height;
    pConst->real_plane_dims[4] = c->plane[2].width;
    pConst->real_plane_dims[5] = c->plane[2].height;

    pConst->plane_strides[0] = c->plane[0].idwt.stride >> (1 + c->pshift);
    pConst->plane_strides[1] = c->plane[1].idwt.stride >> (1 + c->pshift);
    pConst->plane_strides[0] = c->plane[0].idwt.stride >> (1 + c->pshift);

    pConst->plane_offs[0] = 0;
    pConst->plane_offs[1] = c->plane[0].idwt.width * c->plane[0].idwt.height;
    pConst->plane_offs[2] = 2 * pConst->plane_offs[1];

    pConst->wavelet_depth = c->wavelet_depth;

    if (s->quant_buf_vk_ptr == NULL ||
            s->slice_buf_vk_ptr == NULL ||
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
    DiracVulkanDecodeContext*dec = avctx->internal->hwaccel_priv_data;
    DiracContext *ctx = avctx->priv_data;
    VkImageView views[AV_NUM_DATA_POINTERS];
    VkBufferMemoryBarrier2 buf_bar[80];
    VkImageMemoryBarrier2 img_bar[80];
    DiracVulkanDecodePicture *pic = ctx->hwaccel_picture_private;
    FFVkExecContext *exec = ff_vk_exec_get(&dec->exec_pool);

    ff_vk_exec_start(&dec->vkctx, exec);

    err = ff_vk_exec_add_dep_frame(&dec->vkctx, exec, ctx->current_picture->avframe,
                                 VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT,
                                 VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT);
    if (err < 0)
        goto fail;

    err = ff_vk_create_imageviews(&dec->vkctx, exec, views, pic->frame->avframe);
    if (err < 0)
        goto fail;

    err = quant_pl_pass(dec, ctx, exec, buf_bar, &nb_buf_bar);
    if (err < 0)
        goto fail;

    /*err = wavelet_haari_pass(dec, ctx, exec, buf_bar, &nb_buf_bar, 1);*/
    /*if (err < 0)*/
    /*    goto fail;*/

    err = wavelet_legall_pass(dec, ctx, exec, buf_bar, &nb_buf_bar);
    if (err < 0)
        goto fail;

    err = cpy_to_image_pass(dec, ctx, exec, views,
                            buf_bar, &nb_buf_bar, img_bar, &nb_img_bar);
    if (err < 0)
        goto fail;

    err = ff_vk_exec_submit(&dec->vkctx, exec);
    if (err < 0)
        return err;

    ff_vk_exec_wait(&dec->vkctx, exec);

    return 0;

fail:
    ff_vk_exec_discard_deps(&dec->vkctx, exec);
    return err;
}

static int vulkan_dirac_update_thread_context(AVCodecContext *dst, const AVCodecContext *src)
{
    // int err;
    DiracVulkanDecodeContext *src_ctx = src->internal->hwaccel_priv_data;
    DiracVulkanDecodeContext *dst_ctx = dst->internal->hwaccel_priv_data;

    dst_ctx->vkctx = src_ctx->vkctx;
    dst_ctx->yuv_sampler = src_ctx->yuv_sampler;
    dst_ctx->sampler = src_ctx->sampler;
    dst_ctx->qf = src_ctx->qf;
    dst_ctx->quant_pl = src_ctx->quant_pl;

    return 0;
}

static inline int decode_hq_slice(const DiracContext *s, int jobnr)
{
    int i, level, orientation, quant_idx;
    DiracVulkanDecodeContext *dec = s->avctx->internal->hwaccel_priv_data;
    int32_t *qfactor = &dec->quant_buf_vk_ptr[jobnr * 8 * MAX_DWT_LEVELS];
    int32_t *qoffset = &dec->quant_buf_vk_ptr[jobnr * 8 * MAX_DWT_LEVELS + 4];
    int32_t *quant_val_base = dec->quant_val_buf_vk_ptr;
    DiracSlice *slice = &s->slice_params_buf[jobnr];
    SliceCoeffVk *slice_vk = &dec->slice_buf_vk_ptr[jobnr * 3 * MAX_DWT_LEVELS];
    GetBitContext *gb = &slice->gb;

    skip_bits_long(gb, 8*s->highquality.prefix_bytes);
    quant_idx = get_bits(gb, 8);

    if (quant_idx > DIRAC_MAX_QUANT_INDEX - 1) {
        av_log(s->avctx, AV_LOG_ERROR, "Invalid quantization index - %i\n", quant_idx);
        return AVERROR_INVALIDDATA;
    }

    /* Slice quantization (slice_quantizers() in the specs) */
    for (level = 0; level < s->wavelet_depth; level++) {
        for (orientation = !!level; orientation < 4; orientation++) {
            const int quant = FFMAX(quant_idx - s->lowdelay.quant[level][orientation], 0);
            qfactor[level * 8 + orientation] = ff_dirac_qscale_tab[quant];
            qoffset[level * 8 + orientation] = ff_dirac_qoffset_intra_tab[quant] + 2;
        }
    }

    /* Luma + 2 Chroma planes */
    for (i = 0; i < 3; i++) {
        int coef_num, coef_par;
        int64_t length = s->highquality.size_scaler*get_bits(gb, 8);
        int64_t bits_end = get_bits_count(gb) + 8*length;
        const uint8_t *addr = align_get_bits(gb);
        int offs = dec->slice_vals_size * (3 * jobnr + i);
        uint8_t *tmp_buf = (uint8_t *)&quant_val_base[offs];

        if (length*8 > get_bits_left(gb)) {
            av_log(s->avctx, AV_LOG_ERROR, "end too far away\n");
            return AVERROR_INVALIDDATA;
        }

        coef_num = subband_coeffs(s, slice->slice_x, slice->slice_y,
                                    i, offs, &slice_vk[MAX_DWT_LEVELS * i]);

        coef_par = ff_dirac_golomb_read_32bit(addr, length,
                                                tmp_buf, coef_num);

        if (coef_num > coef_par) {
            const int start_b = coef_par * sizeof(int32_t);
            const int end_b   = coef_num * sizeof(int32_t);
            memset(&tmp_buf[start_b], 0, end_b - start_b);
        }

        skip_bits_long(gb, bits_end - get_bits_count(gb));
    }

    return 0;
}

static int decode_hq_slice_row(AVCodecContext *avctx, void *arg, int jobnr, int threadnr)
{
    const DiracContext *s = avctx->priv_data;
    int i, jobn = s->num_x * jobnr;

    for (i = 0; i < s->num_x; i++) {
        decode_hq_slice(s, jobn);
        jobn++;
    }

    return 0;
}

static int vulkan_dirac_decode_slice(AVCodecContext *avctx,
                               const uint8_t  *data,
                               uint32_t        size)
{
    DiracContext *s = avctx->priv_data;

    for (int i = 0; i < s->num_y; i++) {
      decode_hq_slice_row(avctx, NULL, i, 0);
    }

    return 0;
}

const FFHWAccel ff_dirac_vulkan_hwaccel = {
    .p.name                = "vulkan_dirac",
    .p.type                = AVMEDIA_TYPE_VIDEO,
    .p.id                  = AV_CODEC_ID_DIRAC,
    .p.pix_fmt             = AV_PIX_FMT_VULKAN,
    .start_frame           = &vulkan_dirac_start_frame,
    .end_frame             = &vulkan_dirac_end_frame,
    .decode_slice          = &vulkan_dirac_decode_slice,
    .free_frame_priv       = &vulkan_dirac_free_frame_priv,
    .uninit                = &vulkan_dirac_uninit,
    .init                  = &vulkan_dirac_init,
    .frame_params          = &vulkan_dirac_frame_params,
    .frame_priv_data_size  = sizeof(DiracVulkanDecodePicture),
    .decode_params         = &ff_vk_params_invalidate,
    .flush                 = &ff_vk_decode_flush,
    .update_thread_context = &vulkan_dirac_update_thread_context,
    .priv_data_size        = sizeof(DiracVulkanDecodeContext),
    .caps_internal         = HWACCEL_CAP_ASYNC_SAFE | HWACCEL_CAP_THREAD_SAFE,
};
