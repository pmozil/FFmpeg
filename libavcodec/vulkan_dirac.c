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
#include "libavutil/vulkan_loader.h"
#include "vulkan.h"
#include "hwaccel_internal.h"
#include "libavfilter/vulkan_spirv.h"
#include "vulkan_decode.h"

typedef  struct DiracVulkanDecodePicture {
    DiracFrame *pic;
    FFVulkanDecodeContext *ctx;

    FFVulkanPipeline wavelet_pl[7];
    FFVkSPIRVShader wavelet_shd[7];

    FFVulkanPipeline quant_pl[2];
    FFVkSPIRVShader quant_shd[2];
} DiracVulkanDecodePicture;

static const char dequant_16bit[] = {
    C(0, void dequant_16bit(int idx) {                  )
    C(1,     int16_t val = inBuffer[idx];               )
    C(1,     if (val < 0) {                             )
    C(2,         val = -(((-c)*q_fact + q_shift) >> 2); )
    C(1,     } else if (val > 0) {                      )
    C(2,         val = ((c*q_fact + q_shift) >> 2);     )
    C(1,     }                                          )
    C(1,     outBuffer[idx] = val;                      )
    C(0, }                                              )
};

static const char dequant_32bit[] = {
    C(0, void dequant_32bit(int idx) {                  )
    C(1,     int32_t val = inBuffer[idx];               )
    C(1,     if (val < 0) {                             )
    C(2,         val = -(((-c)*q_fact + q_shift) >> 2); )
    C(1,     } else if (val > 0) {                      )
    C(2,         val = ((c*q_fact + q_shift) >> 2);     )
    C(1,     }                                          )
    C(1,     outBuffer[idx] = val;                      )
    C(0, }                                              )
};

static int init_quant_shd(DiracVulkanDecodePicture *s, FFVkSPIRVCompiler *spv, int idx)
{
    int err = 0;
    uint8_t *spv_data;
    size_t spv_len;
    void *spv_opaque = NULL;

    FFVulkanDecodeShared *sh = s->ctx->shared_ctx;
    FFVulkanContext *vkctx = &sh->s;
    FFVkSPIRVShader *shd = &s->quant_shd[idx];
    FFVulkanPipeline *pl = &s->quant_pl[idx];
    FFVulkanDescriptorSetBinding *desc = (FFVulkanDescriptorSetBinding[])
    {
        {
            .name = "quant_in",
            .type = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
            .stages = VK_SHADER_STAGE_COMPUTE_BIT,
            .buf_content = idx ? "int32_t inBuffer[];" : "int16_t inBuffer[];",
        },
        {
            .name = "outputImage",
            .type = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
            .mem_quali = "writeonly",
            .stages = VK_SHADER_STAGE_COMPUTE_BIT,
            .buf_content = idx ? "int32_t outBuffer[];" : "int16_t outBuffer[];",
        },
    };

    RET(ff_vk_shader_init(pl, shd, idx ? "dequant_32bit" : "dequant_16bit", VK_SHADER_STAGE_COMPUTE_BIT, 0));
    ff_vk_shader_set_compute_sizes(shd, 32, 16, 1);
    GLSLC(0, #extension GL_EXT_shader_explicit_arithmetic_types : enable);

    GLSLC(0, layout(push_constant, std430) uniform pushConstants { );
    GLSLC(1,     int q_fact;                                       );
    GLSLC(1,     int q_shift;                                      );
    GLSLC(1,     int tot;                                          );
    GLSLC(0, };                                                    );
    GLSLC(0,                                                       );

    ff_vk_add_push_constant(pl, 0, sizeof(SliceCoeffsPuchConst), VK_SHADER_STAGE_COMPUTE_BIT);
    RET(ff_vk_pipeline_descriptor_set_add(vkctx, pl, shd, desc, 2, 0, 0));


    GLSLC(0, void main()                          );
    GLSLC(0, {                                    );
    GLSLC(1,     for(int i = 0; i < tot; i++) {   );
    if (idx) {
        GLSLC(2,      dequant_32bit(i);           );
    } else {
        GLSLC(2,      dequant_16bit(i);           );
    }
    GLSLC(1, }                                    );

    RET(spv->compile_shader(spv, vkctx, shd, &spv_data, &spv_len, "main", &spv_opaque));
    RET(ff_vk_shader_create(vkctx, shd, spv_data, spv_len, "main"));
    RET(ff_vk_init_compute_pipeline(vkctx, pl, shd));
    RET(ff_vk_exec_pipeline_register(vkctx, &s->ctx->exec_pool, pl));

fail:
    if (spv_opaque)
        spv->free_shader(spv, &spv_opaque);

    return err;
}

static int vulkan_dirac_init(AVCodecContext *avctx)
{
    int err, qf, cxpos = 0, cypos = 0, nb_q = 0;
    VkResult ret;
    FFVulkanDecodeContext *dec = avctx->internal->hwaccel_priv_data;
    FFVulkanDecodeShared *ctx;
    FFVulkanContext *s;
    FFVulkanFunctions *vk;

    VkSamplerYcbcrConversionCreateInfo yuv_sampler_info = {
        .sType = VK_STRUCTURE_TYPE_SAMPLER_YCBCR_CONVERSION_CREATE_INFO,
        .components = ff_comp_identity_map,
        .ycbcrModel = VK_SAMPLER_YCBCR_MODEL_CONVERSION_RGB_IDENTITY,
        .ycbcrRange = avctx->color_range == AVCOL_RANGE_MPEG, /* Ignored */
    };

    av_log(avctx, AV_LOG_INFO, "INIT IN PROGRESS\n");
    err = ff_decode_get_hw_frames_ctx(avctx, AV_HWDEVICE_TYPE_VULKAN);
    if (err < 0)
        return err;

    /* Initialize contexts */
    ctx = dec->shared_ctx;
    s = &ctx->s;
    vk = &ctx->s.vkfn;

    s->frames_ref = av_buffer_ref(avctx->hw_frames_ctx);
    s->frames = (AVHWFramesContext *)s->frames_ref->data;
    s->hwfc = s->frames->hwctx;

    s->device = (AVHWDeviceContext *)s->frames->device_ref->data;
    s->hwctx = s->device->hwctx;

    err = ff_vk_load_props(s);
    if (err < 0)
        goto fail;

    /* Create queue context */
    qf = ff_vk_qf_init(s, &ctx->qf, VK_QUEUE_COMPUTE_BIT);

    if (s->query_props[qf].queryResultStatusSupport)
        nb_q = 1;

    /* Create decode exec context for this specific main thread.
     * 2 async contexts per thread was experimentally determined to be optimal
     * for a majority of streams. */
    err = ff_vk_exec_pool_init(s, &ctx->qf, &dec->exec_pool, 2,
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
                                           s->hwctx->alloc, &ctx->yuv_sampler);
    if (ret != VK_SUCCESS) {
        err = AVERROR_EXTERNAL;
        goto fail;
    }

    ff_vk_decode_flush(avctx);

    av_log(avctx, AV_LOG_VERBOSE, "Vulkan decoder initialization sucessful\n");

    return 0;

fail:
    ff_vk_decode_uninit(avctx);

    return err;
}

static void free_common(FFRefStructOpaque unused, void *obj)
{
    FFVulkanDecodeShared *ctx = obj;
    FFVulkanContext *s = &ctx->s;
    FFVulkanFunctions *vk = &ctx->s.vkfn;

    /* Destroy layered view */
    if (ctx->layered_view)
        vk->DestroyImageView(s->hwctx->act_dev, ctx->layered_view, s->hwctx->alloc);

    /* This also frees all references from this pool */
    av_frame_free(&ctx->layered_frame);
    av_buffer_unref(&ctx->dpb_hwfc_ref);

    /* Destroy parameters */
    if (ctx->empty_session_params)
        vk->DestroyVideoSessionParametersKHR(s->hwctx->act_dev,
                                             ctx->empty_session_params,
                                             s->hwctx->alloc);

    ff_vk_video_common_uninit(s, &ctx->common);

    if (ctx->yuv_sampler)
        vk->DestroySamplerYcbcrConversion(s->hwctx->act_dev, ctx->yuv_sampler,
                                          s->hwctx->alloc);

    ff_vk_uninit(s);
}

static int vulkan_decode_bootstrap(AVCodecContext *avctx, AVBufferRef *frames_ref)
{
    int err;
    FFVulkanDecodeContext *dec = avctx->internal->hwaccel_priv_data;
    AVHWFramesContext *frames = (AVHWFramesContext *)frames_ref->data;
    AVHWDeviceContext *device = (AVHWDeviceContext *)frames->device_ref->data;
    AVVulkanDeviceContext *hwctx = device->hwctx;
    FFVulkanDecodeShared *ctx;

    if (dec->shared_ctx)
        return 0;

    dec->shared_ctx = ff_refstruct_alloc_ext(sizeof(*ctx), 0, NULL,
                                             free_common);
    if (!dec->shared_ctx)
        return AVERROR(ENOMEM);

    ctx = dec->shared_ctx;

    ctx->s.extensions = ff_vk_extensions_to_mask(hwctx->enabled_dev_extensions,
                                                 hwctx->nb_enabled_dev_extensions);

    if (!(ctx->s.extensions & FF_VK_EXT_VIDEO_DECODE_QUEUE)) {
        av_log(avctx, AV_LOG_ERROR, "Device does not support the %s extension!\n",
               VK_KHR_VIDEO_DECODE_QUEUE_EXTENSION_NAME);
        ff_refstruct_unref(&dec->shared_ctx);
        return AVERROR(ENOSYS);
    }

    err = ff_vk_load_functions(device, &ctx->s.vkfn, ctx->s.extensions, 1, 1);
    if (err < 0) {
        ff_refstruct_unref(&dec->shared_ctx);
        return err;
    }

    return 0;
}

static int vulkan_dirac_frame_params(AVCodecContext *avctx, AVBufferRef *hw_frames_ctx)
{
    int err;
    VkFormat vkfmt;
    AVHWFramesContext *frames_ctx = (AVHWFramesContext*)hw_frames_ctx->data;
    AVVulkanFramesContext *hwfc = frames_ctx->hwctx;
    DiracContext *s = avctx->priv_data;

    frames_ctx->sw_format = avctx->pix_fmt;
    // frames_ctx->sw_format = AV_PIX_FMT_NONE;

    err = vulkan_decode_bootstrap(avctx, hw_frames_ctx);
    if (err < 0)
        return err;

    frames_ctx->width  = s->seq.width;
    frames_ctx->height = s->seq.height;
    frames_ctx->format = AV_PIX_FMT_VULKAN;

    hwfc->format[0]    = vkfmt;
    hwfc->tiling       = VK_IMAGE_TILING_OPTIMAL;
    hwfc->usage        = VK_IMAGE_USAGE_TRANSFER_SRC_BIT         |
                         VK_IMAGE_USAGE_SAMPLED_BIT              |
                         VK_IMAGE_USAGE_VIDEO_DECODE_DST_BIT_KHR;

    return err;
}

const FFHWAccel ff_dirac_vulkan_hwaccel = {
    .p.name                = "dirac_vulkan",
    .p.type                = AVMEDIA_TYPE_VIDEO,
    .p.id                  = AV_CODEC_ID_DIRAC,
    .p.pix_fmt             = AV_PIX_FMT_VULKAN,
    // .start_frame           = &vk_av1_start_frame,
    // .decode_slice          = &vk_av1_decode_slice,
    // .end_frame             = &vk_av1_end_frame,
    // .free_frame_priv       = &vk_av1_free_frame_priv,
    .frame_priv_data_size  = sizeof(DiracVulkanDecodePicture),
    .init                  = &vulkan_dirac_init,
    .update_thread_context = &ff_vk_update_thread_context,
    .decode_params         = &ff_vk_params_invalidate,
    .flush                 = &ff_vk_decode_flush,
    .uninit                = &ff_vk_decode_uninit,
    .frame_params          = &vulkan_dirac_frame_params,
    .priv_data_size        = sizeof(FFVulkanDecodeContext),

    /* NOTE: Threading is intentionally disabled here. Due to the design of Vulkan,
     * where frames are opaque to users, and mostly opaque for driver developers,
     * there's an issue with current hardware accelerator implementations of AV1,
     * where they require an internal index. With regular hwaccel APIs, this index
     * is given to users as an opaque handle directly. With Vulkan, due to increased
     * flexibility, this index cannot be present anywhere.
     * The current implementation tracks the index for the driver and submits it
     * as necessary information. Due to needing to modify the decoding context,
     * which is not thread-safe, on frame free, threading is disabled. */
    .caps_internal         = HWACCEL_CAP_ASYNC_SAFE,
    // .caps_internal         = HWACCEL_CAP_ASYNC_SAFE | HWACCEL_CAP_THREAD_SAFE,
};
