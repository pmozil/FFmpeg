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

typedef  struct DiracVulkanDecodeContext {
    FFVulkanContext vkctx;
    FFVulkanFunctions vkfn;
    VkSamplerYcbcrConversion yuv_sampler;

    FFVulkanPipeline wavelet_pl[7];
    FFVkSPIRVShader wavelet_shd[7];

    FFVulkanPipeline quant_pl[2];
    FFVkSPIRVShader quant_shd[2];

    FFVkQueueFamilyCtx qf;
    FFVkExecPool exec_pool;
} DiracVulkanDecodeContext;

typedef  struct DiracVulkanDecodePicture {
    DiracFrame *pic;
    DiracVulkanDecodeContext *ctx;

    VkImageView img_view_ref;
    VkImageView img_view_out;

    VkImageAspectFlags              img_aspect;

    VkSemaphore                     sem;
    uint64_t                        sem_value;

    PFN_vkWaitSemaphores            wait_semaphores;
    PFN_vkDestroyImageView          destroy_image_view;
} DiracVulkanDecodePicture;

static const char dequant_16bit[] = {
    C(0, void dequant_16bit(int idx, int16_t qf, int16_t qs) {     )
    C(1,     int16_t val = inBuffer[idx];                          )
    C(1,     if (val < 0) {                                        )
    C(2,         val = -(((-val)*qf + qs) >> int16_t(2));          )
    C(1,     } else if (val > 0) {                                 )
    C(2,         val = ((val*qf + qs) >> int16_t(2));              )
    C(1,     }                                                     )
    C(1,     outBuffer[idx] = val;                                 )
    C(0, }                                                         )
};

static const char dequant_32bit[] = {
    C(0, void dequant_32bit(int idx, int32_t qf, int32_t qs) {     )
    C(1,     int32_t val = inBuffer[idx];                          )
    C(1,     if (val < 0) {                                        )
    C(2,         val = -(((-val)*qf + qs) >> int32_t(2));          )
    C(1,     } else if (val > 0) {                                 )
    C(2,         val = ((val*qf + qs) >> int32_t(2));              )
    C(1,     }                                                     )
    C(1,     outBuffer[idx] = val;                                 )
    C(0, }                                                         )
};

static int init_quant_shd(DiracVulkanDecodeContext *s, FFVkSPIRVCompiler *spv, int idx)
{
    int err = 0;
    uint8_t *spv_data;
    size_t spv_len;
    void *spv_opaque = NULL;

    FFVulkanContext *vkctx = &s->vkctx;
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

    if (idx)
    {
        GLSLD(dequant_32bit);
    } else
    {
        GLSLD(dequant_16bit);
    }

    GLSLC(0, void main()                          );
    GLSLC(0, {                                    );
    GLSLC(1,     for(int i = 0; i < tot; i++) {   );
    if (idx) {
        GLSLC(2,      dequant_32bit(i, int32_t(q_fact), int32_t(q_shift)););
    } else {
        GLSLC(2,      dequant_16bit(i, int16_t(q_fact), int16_t(q_shift)););
    }
    GLSLC(1,     }                                );
    GLSLC(0, }                                    );

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

    spv = ff_vk_spirv_init();
    if (!spv) {
        av_log(avctx, AV_LOG_ERROR, "Unable to initialize SPIR-V compiler!\n");
        return AVERROR_EXTERNAL;
    }

    VkSamplerYcbcrConversionCreateInfo yuv_sampler_info = {
        .sType = VK_STRUCTURE_TYPE_SAMPLER_YCBCR_CONVERSION_CREATE_INFO,
        .components = ff_comp_identity_map,
        .ycbcrModel = VK_SAMPLER_YCBCR_MODEL_CONVERSION_RGB_IDENTITY,
        .ycbcrRange = avctx->color_range == AVCOL_RANGE_MPEG, /* Ignored */
    };

    err = ff_decode_get_hw_frames_ctx(avctx, AV_HWDEVICE_TYPE_VULKAN);
    if (err < 0)
        goto fail;

    /* Initialize contexts */
    s = &dec->vkctx;
    vk = &dec->vkfn;

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

    /* Create decode exec context for this specific main thread.
     * 2 async contexts per thread was experimentally determined to be optimal
     * for a majority of streams. */
    err = ff_vk_exec_pool_init(s, &dec->qf, &dec->exec_pool, 2,
                               nb_q, 0, 0,
                               NULL);
    if (err < 0)
        goto fail;

    /* Get sampler */
    // av_chroma_location_enum_to_pos(&cxpos, &cypos, avctx->chroma_sample_location);
    // yuv_sampler_info.xChromaOffset = cxpos >> 7;
    // yuv_sampler_info.yChromaOffset = cypos >> 7;
    // yuv_sampler_info.format = s->hwfc->format[0];
    // ret = vk->CreateSamplerYcbcrConversion(s->hwctx->act_dev, &yuv_sampler_info,
    //                                        s->hwctx->alloc, &dec->yuv_sampler);
    // if (ret != VK_SUCCESS) {
    //     err = AVERROR_EXTERNAL;
    //     goto fail;
    // }

    init_quant_shd(dec, spv, 0);
    init_quant_shd(dec, spv, 1);

    av_log(avctx, AV_LOG_VERBOSE, "Vulkan decoder initialization sucessful\n");

fail:
    if (spv)
    {
        spv->uninit(&spv);
    }
    vulkan_dirac_uninit(avctx);

    return err;
}

static void free_common(AVCodecContext *avctx)
{
    DiracVulkanDecodeContext *ctx = avctx->internal->hwaccel_priv_data;
    FFVulkanContext *s = &ctx->vkctx;
    FFVulkanFunctions *vk = &ctx->vkfn;

    if (ctx->yuv_sampler)
        vk->DestroySamplerYcbcrConversion(s->hwctx->act_dev, ctx->yuv_sampler,
                                          s->hwctx->alloc);

    ff_vk_uninit(s);
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
    VkFormat vkfmt;
    AVHWFramesContext *frames_ctx = (AVHWFramesContext*)hw_frames_ctx->data;
    AVVulkanFramesContext *hwfc = frames_ctx->hwctx;
    DiracContext *dctx = avctx->priv_data;

    frames_ctx->sw_format = dctx->seq.pix_fmt;

    err = vulkan_decode_bootstrap(avctx, hw_frames_ctx);
    if (err < 0)
        return err;

    frames_ctx->width  = avctx->coded_width;
    frames_ctx->height = avctx->coded_height;
    frames_ctx->format = AV_PIX_FMT_VULKAN;

    hwfc->format[0]    = vkfmt;
    hwfc->tiling       = VK_IMAGE_TILING_OPTIMAL;
    hwfc->usage        = VK_IMAGE_USAGE_TRANSFER_SRC_BIT         |
                         VK_IMAGE_USAGE_SAMPLED_BIT              |
                         VK_IMAGE_USAGE_VIDEO_DECODE_DST_BIT_KHR;

    return err;
}

//
// static void vulkan_dirac_free_frame_priv(FFRefStructOpaque _hwctx, void *data)
// {
//     AVHWDeviceContext *hwctx = _hwctx.nc;
//     DiracVulkanDecodePicture *dp = data;
//
//     /* Free frame resources */
//     av_free(dp);
//     // ff_vk_decode_free_frame(hwctx, &hp->vp);
// }

static int vulkan_dirac_create_view(DiracVulkanDecodeContext *dec, VkImageView *dst_view,
                                 VkImageAspectFlags *aspect, AVVkFrame *src,
                                 VkFormat vkf, int is_current)
{
    VkResult ret;
    FFVulkanFunctions *vk = &dec->vkctx.vkfn;
    VkImageAspectFlags aspect_mask = ff_vk_aspect_bits_from_vkfmt(vkf);

    VkSamplerYcbcrConversionInfo yuv_sampler_info = {
        .sType = VK_STRUCTURE_TYPE_SAMPLER_YCBCR_CONVERSION_INFO,
        .conversion = dec->yuv_sampler,
    };
    VkImageViewCreateInfo img_view_create_info = {
        .sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO,
        .pNext = &yuv_sampler_info,
        .viewType = VK_IMAGE_VIEW_TYPE_2D_ARRAY,
        .format = vkf,
        .image = src->img[0],
        .components = (VkComponentMapping) {
            .r = VK_COMPONENT_SWIZZLE_IDENTITY,
            .g = VK_COMPONENT_SWIZZLE_IDENTITY,
            .b = VK_COMPONENT_SWIZZLE_IDENTITY,
            .a = VK_COMPONENT_SWIZZLE_IDENTITY,
        },
        .subresourceRange = (VkImageSubresourceRange) {
            .aspectMask     = VK_IMAGE_ASPECT_COLOR_BIT,
            .baseArrayLayer = 0,
            .layerCount     = VK_REMAINING_ARRAY_LAYERS,
            .levelCount     = 1,
        },
    };

    ret = vk->CreateImageView(dec->vkctx.hwctx->act_dev, &img_view_create_info,
                              dec->vkctx.hwctx->alloc, dst_view);
    if (ret != VK_SUCCESS)
        return AVERROR_EXTERNAL;

    *aspect = aspect_mask;

    return 0;
}

static int vulkan_dirac_prepare_frame(DiracVulkanDecodeContext *dec, AVFrame *pic,
                               DiracVulkanDecodePicture *vkpic)
{
    int err;
    FFVulkanFunctions *vk = &dec->vkctx.vkfn;

    vkpic->img_view_ref  = VK_NULL_HANDLE;
    vkpic->img_view_out  = VK_NULL_HANDLE;

    vkpic->destroy_image_view = vk->DestroyImageView;
    vkpic->wait_semaphores = vk->WaitSemaphores;

    AVHWFramesContext *frames = (AVHWFramesContext *)pic->hw_frames_ctx->data;
    AVVulkanFramesContext *hwfc = frames->hwctx;

    err = vulkan_dirac_create_view(dec, &vkpic->img_view_out,
                                &vkpic->img_aspect,
                                (AVVkFrame *)pic->data[0],
                                hwfc->format[0], 1);
    if (err < 0)
        return err;

    return 0;
}

static int vulkan_dirac_start_frame(AVCodecContext          *avctx,
                               av_unused const uint8_t *buffer,
                               av_unused uint32_t       size)
{
    av_log(avctx, AV_LOG_INFO, "Start dirac HW frame\n");

    return 0;
}

static int vulkan_dirac_end_frame(AVCodecContext *avctx) {
    av_log(avctx, AV_LOG_INFO, "End dirac HW frame\n");

    return 0;
}

const FFHWAccel ff_dirac_vulkan_hwaccel = {
    .p.name                = "dirac_vulkan",
    .p.type                = AVMEDIA_TYPE_VIDEO,
    .p.id                  = AV_CODEC_ID_DIRAC,
    .p.pix_fmt             = AV_PIX_FMT_VULKAN,
    .start_frame           = &vulkan_dirac_start_frame,
    .end_frame             = &vulkan_dirac_end_frame,
    // .decode_slice          = &vk_h264_decode_slice,
    // .free_frame_priv       = &vulkan_dirac_free_frame_priv,
    .uninit                = &vulkan_dirac_uninit,
    .init                  = &vulkan_dirac_init,
    .frame_params          = &vulkan_dirac_frame_params,
    .frame_priv_data_size  = sizeof(DiracVulkanDecodePicture),
    .update_thread_context = &ff_vk_update_thread_context,
    .decode_params         = &ff_vk_params_invalidate,
    .flush                 = &ff_vk_decode_flush,
    .priv_data_size        = sizeof(DiracVulkanDecodeContext),
    .caps_internal         = HWACCEL_CAP_ASYNC_SAFE | HWACCEL_CAP_THREAD_SAFE,
};
