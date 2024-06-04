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

typedef  struct DiracVulkanDecodeContext {
    FFVulkanContext vkctx;
    VkSamplerYcbcrConversion yuv_sampler;
    VkSampler sampler;

    FFVulkanPipeline wavelet_pl[MAX_AUTO_THREADS];
    FFVkSPIRVShader wavelet_shd[MAX_AUTO_THREADS];

    FFVulkanPipeline quant_pl[MAX_AUTO_THREADS];
    FFVkSPIRVShader quant_shd[MAX_AUTO_THREADS];

    FFVkQueueFamilyCtx qf;
    FFVkExecPool exec_pool;

    FFVkBuffer thread_buf_vk;
    uint8_t *thread_buf_ptr;
    uint8_t *mapped_thread_buf_ptr;
    size_t offset;

    AVBufferPool *quant_buf_pool;
    AVBufferPool *slice_coeff_buf_pool;

    AVBufferRef *av_quant_bufs[MAX_AUTO_THREADS];
    AVBufferRef *av_slice_coeffs_bufs[MAX_AUTO_THREADS];

    FFVkBuffer *quant_bufs[MAX_AUTO_THREADS];
    FFVkBuffer *slice_coeffs_bufs[MAX_AUTO_THREADS];
} DiracVulkanDecodeContext;

typedef  struct DiracVulkanDecodePicture {
    VkImageView views[AV_NUM_DATA_POINTERS];
    VkImageAspectFlags              img_aspect;

    VkSemaphore                     sem;
    uint64_t                        sem_value;

    PFN_vkWaitSemaphores            wait_semaphores;
    PFN_vkDestroyImageView          destroy_image_view;

    DiracFrame *frame;
} DiracVulkanDecodePicture;

static const char dequant[] = {
    C(0, void dequant(int idx, ivec2 pos) {                        )
    C(1,     int val = inBuffer[idx];                              )
    C(1,     if (val < 0) {                                        )
    C(2,         val = -(((-val)*qf + qs) >> 2);                   )
    C(1,     } else if (val > 0) {                                 )
    C(2,         val = ((val*qf + qs) >> 2);                       )
    C(1,     }                                                     )
    C(1,     imageStore(out_img[plane], pos, vec4(val));           )
    C(0, }                                                         )
};

static void free_common(AVCodecContext *avctx)
{
    DiracVulkanDecodeContext *dec = avctx->internal->hwaccel_priv_data;
    // FFVulkanContext *s = &dec->vkctx;
    // FFVulkanFunctions *vk = &dec->vkctx.vkfn;

    /* Wait on and free execution pool */
    // if (dec->exec_pool.cmd_bufs) {
    //     ff_vk_exec_pool_free(&dec->vkctx, &dec->exec_pool);
    // }
    //
    // for (int i = 0; i < MAX_AUTO_THREADS; i++) {
    //     ff_vk_pipeline_free(s, &dec->quant_pl[i]);
    //     ff_vk_shader_free(s, &dec->quant_shd[i]);
    //
    //     ff_vk_pipeline_free(&dec->vkctx, &dec->wavelet_pl[i]);
    //     ff_vk_shader_free(&dec->vkctx, &dec->wavelet_shd[i]);
    // }

    //
    // if (dec->yuv_sampler)
    //     vk->DestroySamplerYcbcrConversion(s->hwctx->act_dev, dec->yuv_sampler,
    //                                       s->hwctx->alloc);
    // if (dec->sampler)
    //     vk->DestroySampler(s->hwctx->act_dev, dec->sampler, s->hwctx->alloc);

    if (dec->thread_buf_ptr) {
        // ff_vk_unmap_buffer(&dec->vkctx, dec->thread_buf_vk, 0);
        // av_buffer_unref(&dec->thread_buf_vk_ref);
        ff_vk_free_buf(&dec->vkctx, &dec->thread_buf_vk);
    }

    if (dec->slice_coeff_buf_pool)
        av_buffer_pool_uninit(&dec->slice_coeff_buf_pool);

    if (dec->quant_buf_pool)
        av_buffer_pool_uninit(&dec->quant_buf_pool);

    // ff_vk_uninit(s);
}

static int alloc_data_bufs(DiracVulkanDecodeContext *dec) {
    int err;
    for (int i = 0; i < MAX_AUTO_THREADS; i++) {
        err = ff_vk_get_pooled_buffer(&dec->vkctx, &dec->quant_buf_pool,
                                        &dec->av_quant_bufs[i],
                                        VK_BUFFER_USAGE_STORAGE_BUFFER_BIT |
                                        VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT,
                                        NULL,
                                        MAX_DWT_LEVELS * 8,
                                        VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT |
                                        VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
        if (err < 0)
            goto free_quants;
        dec->quant_bufs[i] = (FFVkBuffer *)dec->av_quant_bufs[i]->data;

        err = ff_vk_get_pooled_buffer(&dec->vkctx, &dec->slice_coeff_buf_pool,
                                        &dec->av_slice_coeffs_bufs[i],
                                        VK_BUFFER_USAGE_STORAGE_BUFFER_BIT |
                                        VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT,
                                        NULL,
                                        MAX_DWT_LEVELS * sizeof(SliceCoeffs),
                                        VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT |
                                        VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
        if (err < 0)
            goto free_slices;
        dec->slice_coeffs_bufs[i] = (FFVkBuffer *)dec->av_slice_coeffs_bufs[i]->data;

    }

    return 0;

free_slices:
    if (dec->slice_coeff_buf_pool)
        av_buffer_pool_uninit(&dec->slice_coeff_buf_pool);

free_quants:
    if (dec->quant_buf_pool)
        av_buffer_pool_uninit(&dec->quant_buf_pool);

    return err;
}

static int alloc_thread_buf(DiracContext *ctx, DiracVulkanDecodeContext *dec) {
    FFVulkanContext *s = &dec->vkctx;
    FFVulkanFunctions *vk = &s->vkfn;
    VkExternalMemoryBufferCreateInfo create_desc = {
        .sType = VK_STRUCTURE_TYPE_EXTERNAL_MEMORY_BUFFER_CREATE_INFO,
        .handleTypes = VK_EXTERNAL_MEMORY_HANDLE_TYPE_HOST_ALLOCATION_BIT_EXT,
    };

    VkImportMemoryHostPointerInfoEXT import_desc = {
        .sType = VK_STRUCTURE_TYPE_IMPORT_MEMORY_HOST_POINTER_INFO_EXT,
        .handleType = VK_EXTERNAL_MEMORY_HANDLE_TYPE_HOST_ALLOCATION_BIT_EXT,
    };

    VkMemoryHostPointerPropertiesEXT p_props = {
        .sType = VK_STRUCTURE_TYPE_MEMORY_HOST_POINTER_PROPERTIES_EXT,
    };

    size_t offs;
    size_t req_size;
    int err;

    av_size_mult(ctx->threads_num_buf, ctx->thread_buf_size, &req_size);

    offs = (uintptr_t)ctx->thread_buf % s->hprops.minImportedHostPointerAlignment;
    import_desc.pHostPointer = ctx->thread_buf - offs;
    req_size = FFALIGN(offs + req_size,
                s->hprops.minImportedHostPointerAlignment);

    err = vk->GetMemoryHostPointerPropertiesEXT(s->hwctx->act_dev,
                                                import_desc.handleType,
                                                import_desc.pHostPointer,
                                                &p_props);

    if (dec->thread_buf_ptr) {
        ff_vk_free_buf(&dec->vkctx, &dec->thread_buf_vk);
    }

    err = ff_vk_create_buf(&dec->vkctx, &dec->thread_buf_vk, req_size,
                            &create_desc,
                            &import_desc,
                            VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
                            VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT |
                                VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);
    if (err < 0)
        return err;

    dec->thread_buf_ptr = ctx->thread_buf;
    dec->offset = offs;

    return ff_vk_map_buffer(&dec->vkctx, &dec->thread_buf_vk, &dec->mapped_thread_buf_ptr, 0);
}


static int init_quant_shd(DiracVulkanDecodeContext *s, FFVkSPIRVCompiler *spv,
                          FFVkSPIRVShader *shd, FFVulkanPipeline *pl)
{
    int err = 0;
    uint8_t *spv_data;
    size_t spv_len;
    void *spv_opaque = NULL;
    const int planes = av_pix_fmt_count_planes(s->vkctx.output_format);
    FFVulkanContext *vkctx = &s->vkctx;
    // FFVkSPIRVShader *shd = &s->quant_shd;
    // FFVulkanPipeline *pl = &s->quant_pl;
    FFVulkanDescriptorSetBinding *desc = (FFVulkanDescriptorSetBinding[])
    {
        {
            .name = "quant_in",
            .type = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
            .stages = VK_SHADER_STAGE_COMPUTE_BIT,
            .buf_content = "int inBuffer[];",
            .mem_quali = "readonly",
        },
        {
            .name       = "out_img",
            .type       = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE,
            .stages     = VK_SHADER_STAGE_COMPUTE_BIT,
            .mem_layout = ff_vk_shader_rep_fmt(s->vkctx.output_format),
            .dimensions = 2,
            .elems      = planes,
            .mem_quali = "writeonly",
        },
    };
    // av_log(vkctx, AV_LOG_INFO, "N planes = %i\n", planes);

    RET(ff_vk_shader_init(pl, shd, "dequant", VK_SHADER_STAGE_COMPUTE_BIT, 0));
    ff_vk_shader_set_compute_sizes(shd, 8, 8, 1);

    GLSLC(0, layout(push_constant, std430) uniform pushConstants { );
    GLSLC(1,     int left;                                         );
    GLSLC(1,     int top;                                          );
    GLSLC(1,     int tot_h;                                        );
    GLSLC(1,     int tot_v;                                        );
    GLSLC(1,     int off;                                          );
    GLSLC(1,     int plane;                                        );
    GLSLC(1,     int qs;                                           );
    GLSLC(1,     int qf;                                           );
    GLSLC(0, };                                                    );
    GLSLC(0,                                                       );

    ff_vk_add_push_constant(pl, 0, sizeof(SliceCoeffsPushConst), VK_SHADER_STAGE_COMPUTE_BIT);
    RET(ff_vk_pipeline_descriptor_set_add(vkctx, pl, shd, desc, 2, 0, 0));

    GLSLD(dequant);
    GLSLC(0, void main()                                              );
    GLSLC(0, {                                                        );
    GLSLC(1,     int i = 0;                                           );
    GLSLC(1,     int y = int(gl_GlobalInvocationID.y);                );
    GLSLC(1,     for(; y < tot_v; y += int(gl_NumWorkGroups.y)) {     );
    GLSLC(2,        int x = int(gl_GlobalInvocationID.x);             );
    GLSLC(2,        for(; x < tot_h; x += int(gl_NumWorkGroups.x)) {  );
    GLSLC(3,            i++;                                          );
    GLSLC(3,            dequant(off + i, ivec2(top + y, left + x));   );
    GLSLC(1,        }                                                 );
    GLSLC(1,     }                                                    );
    GLSLC(0, }                                                        );

    RET(spv->compile_shader(spv, vkctx, shd, &spv_data, &spv_len, "main", &spv_opaque));
    RET(ff_vk_shader_create(vkctx, shd, spv_data, spv_len, "main"));
    RET(ff_vk_init_compute_pipeline(vkctx, pl, shd));
    RET(ff_vk_exec_pipeline_register(vkctx, &s->exec_pool, pl));

fail:
    if (spv_opaque)
        spv->free_shader(spv, &spv_opaque);

    return err;
}

static int vulkan_dirac_uninit(AVCodecContext *avctx) {
    DiracContext *d = avctx->priv_data;
    if (d->hwaccel_picture_private) {
        av_free(d->hwaccel_picture_private);
    }

    free_common(avctx);

    return 0;
}

static int vulkan_dirac_init(AVCodecContext *avctx)
{
int err = 0, qf, cxpos = 0, cypos = 0, nb_q = 0;
    VkResult ret;
    DiracVulkanDecodeContext *dec = avctx->internal->hwaccel_priv_data;
    FFVulkanContext *s;
    FFVulkanFunctions *vk;
    FFVkSPIRVCompiler *spv;

    VkSamplerYcbcrConversionCreateInfo yuv_sampler_info = {
        .sType = VK_STRUCTURE_TYPE_SAMPLER_YCBCR_CONVERSION_CREATE_INFO,
        .components = ff_comp_identity_map,
        .ycbcrModel = VK_SAMPLER_YCBCR_MODEL_CONVERSION_RGB_IDENTITY,
        .ycbcrRange = avctx->color_range == AVCOL_RANGE_MPEG, /* Ignored */
    };

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
    vk = &dec->vkctx.vkfn;

    s->frames_ref = av_buffer_ref(avctx->hw_frames_ctx);
    s->frames = (AVHWFramesContext *)s->frames_ref->data;
    s->hwfc = s->frames->hwctx;

    s->device = (AVHWDeviceContext *)s->frames->device_ref->data;
    s->hwctx = s->device->hwctx;

    err = ff_vk_load_props(s);
    if (err < 0)
        goto fail;

    /* Create queue context */
    qf = ff_vk_qf_init(s, &dec->qf, VK_QUEUE_COMPUTE_BIT);

    if (s->query_props[qf].queryResultStatusSupport)
        nb_q = 1;

    err = ff_vk_exec_pool_init(s, &dec->qf, &dec->exec_pool, 2,
                               nb_q, 0, 0,
                               NULL);
    if (err < 0)
        goto fail;

    /* Get sampler */
    av_chroma_location_enum_to_pos(&cxpos, &cypos, avctx->chroma_sample_location);
    yuv_sampler_info.xChromaOffset = cxpos >> 7;
    yuv_sampler_info.yChromaOffset = cypos >> 7;
    yuv_sampler_info.format = s->hwfc->format[0];
    ret = vk->CreateSamplerYcbcrConversion(s->hwctx->act_dev, &yuv_sampler_info,
                                           s->hwctx->alloc, &dec->yuv_sampler);
    if (ret != VK_SUCCESS) {
        err = AVERROR_EXTERNAL;
        goto fail;
    }

    err = ff_vk_init_sampler(&dec->vkctx, &dec->sampler, 1, VK_FILTER_LINEAR);
    if (err < 0) {
        goto fail;
    }

    for (int i = 0; i < MAX_AUTO_THREADS; i++)
        init_quant_shd(dec, spv, &dec->quant_shd[i], &dec->quant_pl[i]);

    av_log(avctx, AV_LOG_VERBOSE, "Vulkan decoder initialization sucessful\n");

    dec->thread_buf_ptr = NULL;

    err = alloc_data_bufs(dec);
    if (err < 0)
        goto fail;

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
    // DiracContext *dctx = avctx->priv_data;

    frames_ctx->sw_format = avctx->pix_fmt;

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

static int vulkan_dirac_prepare_frame(DiracVulkanDecodeContext *dec, AVFrame *pic,
                               DiracVulkanDecodePicture *vkpic)
{
    FFVulkanFunctions *vk = &dec->vkctx.vkfn;
    vkpic->wait_semaphores = vk->WaitSemaphores;

    return 0;
}

static int vulkan_dirac_start_frame(AVCodecContext          *avctx,
                               av_unused const uint8_t *buffer,
                               av_unused uint32_t       size)
{
    int err;
    DiracVulkanDecodeContext *s = avctx->internal->hwaccel_priv_data;
    DiracContext *c = avctx->priv_data;
    DiracVulkanDecodePicture *pic = c->hwaccel_picture_private;
    pic->frame = c->current_picture;

    err = vulkan_dirac_prepare_frame(s, c->current_picture->avframe, pic);
    if (err < 0) {
        return err;
    }

    return 0;
}

static int vulkan_dirac_end_frame(AVCodecContext *avctx) {
    // av_log(avctx, AV_LOG_INFO, "End dirac HW frame\n");
    // DiracVulkanDecodeContext*dec = avctx->internal->hwaccel_priv_data;
    // DiracContext *ctx = avctx->priv_data;
    // DiracVulkanDecodePicture *pic = ctx->hwaccel_picture_private;

    return 0;
}

static int vulkan_dirac_update_thread_context(AVCodecContext *dst, const AVCodecContext *src)
{
    int err;
    DiracVulkanDecodeContext *src_ctx = src->internal->hwaccel_priv_data;
    DiracVulkanDecodeContext *dst_ctx = dst->internal->hwaccel_priv_data;

    dst_ctx->vkctx = src_ctx->vkctx;
    dst_ctx->yuv_sampler = src_ctx->yuv_sampler;
    dst_ctx->sampler = src_ctx->sampler;
    for (int j = 0; j < MAX_AUTO_THREADS; j++) {
        dst_ctx->wavelet_pl[j] = src_ctx->wavelet_pl[j];
        dst_ctx->wavelet_shd[j] = src_ctx->wavelet_shd[j];

        dst_ctx->quant_pl[j] = src_ctx->quant_pl[j];
        dst_ctx->quant_shd[j] = src_ctx->quant_shd[j];
    }
    dst_ctx->qf = src_ctx->qf;

    if (!dst_ctx->exec_pool.cmd_bufs) {
        err = ff_vk_exec_pool_init(&src_ctx->vkctx, &src_ctx->qf, &dst_ctx->exec_pool, 2,
                                    0, 0, 0,
                                    NULL);
        if (err < 0)
            return err;
    }

    return 0;
}

static int subband_coeffs(const DiracContext *s, int x, int y, int p,
                          SliceCoeffs c[MAX_DWT_LEVELS])
{
    int level, coef = 0;
    for (level = 0; level < s->wavelet_depth; level++) {
        SliceCoeffs *o = &c[level];
        const SubBand *b = &s->plane[p].band[level][3]; /* orientation doens't matter */
        o->top   = b->height * y / s->num_y;
        o->left  = b->width  * x / s->num_x;
        o->tot_h = ((b->width  * (x + 1)) / s->num_x) - o->left;
        o->tot_v = ((b->height * (y + 1)) / s->num_y) - o->top;
        o->tot   = o->tot_h*o->tot_v;
        coef    += o->tot * (4 - !!level);
    }
    return coef;
}

static int decode_hq_slice(const DiracContext *s,
                           DiracSlice *slice, uint8_t *tmp_buf,
                           int threadnr)
{
    DiracVulkanDecodeContext *dec = s->avctx->internal->hwaccel_priv_data;
    int i, level, orientation, quant_idx;
    int *qfactor = (int *)dec->quant_bufs[threadnr]->mapped_mem;
    int *qoffset = qfactor + MAX_DWT_LEVELS * 4;
    GetBitContext *gb = &slice->gb;
    SliceCoeffs *coeffs_num = (SliceCoeffs *) dec->slice_coeffs_bufs[threadnr]->mapped_mem;
    DiracVulkanDecodePicture *pic = s->hwaccel_picture_private;
    SliceCoeffsPushConst push_c;
    push_c.off = (s->thread_buf_size * threadnr) >> 2;

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
            qfactor[level * MAX_DWT_LEVELS + orientation] = ff_dirac_qscale_tab[quant];
            qoffset[level * MAX_DWT_LEVELS + orientation] = ff_dirac_qoffset_intra_tab[quant] + 2;
        }
    }

    for (i = 0; i < 3; i++) {
        int coef_num, coef_par, off = 0;
        int64_t length = s->highquality.size_scaler*get_bits(gb, 8);
        int64_t bits_end = get_bits_count(gb) + 8*length;
        const uint8_t *addr = align_get_bits(gb);
        push_c.plane = i;

        if (length*8 > get_bits_left(gb)) {
            av_log(s->avctx, AV_LOG_ERROR, "end too far away\n");
            return AVERROR_INVALIDDATA;
        }

        coef_num = subband_coeffs(s, slice->slice_x, slice->slice_y, i, coeffs_num);

        coef_par = ff_dirac_golomb_read_32bit(addr, length,
                                                tmp_buf, coef_num);

        if (coef_num > coef_par) {
            const int start_b = coef_par * (1 << (s->pshift + 1));
            const int end_b   = coef_num * (1 << (s->pshift + 1));
            memset(&tmp_buf[start_b], 0, end_b - start_b);
        }

        skip_bits_long(gb, bits_end - get_bits_count(gb));
    }

    return 0;
}

static int vulkan_dirac_decode_slice_row(AVCodecContext *avctx,
                                        void  *arg, int jobnr,
                                        int threadnr)
{
    int i, err;
    DiracContext *s = avctx->priv_data;
    DiracSlice *slices = ((DiracSlice *)arg) + s->num_x*jobnr;
    DiracVulkanDecodeContext *dec = avctx->internal->hwaccel_priv_data;
    uint8_t *thread_buf;

    thread_buf = &dec->mapped_thread_buf_ptr[s->thread_buf_size*threadnr];

    for (i = 0; i < s->num_x; i++) {
        err = decode_hq_slice(s, &slices[i], thread_buf, threadnr);
        if (err < 0)
            return err;
    }

    return 0;
}

static int vulkan_dirac_decode_slice(AVCodecContext *avctx,
                               const uint8_t  *data,
                               uint32_t        size)
{
    int err;
    DiracSlice *slices = (DiracSlice *)data;
    DiracContext *s = avctx->priv_data;
    DiracVulkanDecodeContext *dec = avctx->internal->hwaccel_priv_data;

    if (s->thread_buf != dec->thread_buf_ptr) {
        err = alloc_thread_buf(s, dec);
        if (err < 0)
            return 0;
    }

    avctx->execute2(avctx, vulkan_dirac_decode_slice_row, slices, NULL, s->num_y);

    return 0;
}

const FFHWAccel ff_dirac_vulkan_hwaccel = {
    .p.name                = "dirac_vulkan",
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
    // .caps_internal         = HWACCEL_CAP_ASYNC_SAFE,
};
