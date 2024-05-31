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
    VkSamplerYcbcrConversion yuv_sampler;
    VkSampler sampler;

    FFVulkanPipeline wavelet_pl[7];
    FFVkSPIRVShader wavelet_shd[7];

    FFVulkanPipeline quant_pl;
    FFVkSPIRVShader quant_shd;

    FFVkQueueFamilyCtx qf;
    FFVkExecPool exec_pool;

    FFVkQueueFamilyCtx upload_qf;
    FFVkExecPool upload_exec_pool;
} DiracVulkanDecodeContext;

typedef  struct DiracVulkanDecodePicture {
    VkImageView views[AV_NUM_DATA_POINTERS];
    VkImageAspectFlags              img_aspect;

    VkSemaphore                     sem;
    uint64_t                        sem_value;

    PFN_vkWaitSemaphores            wait_semaphores;
    PFN_vkDestroyImageView          destroy_image_view;

    FFVkExecContext *exec;
    SliceCoeffsPushConst push_c;

    AVBufferPool *buf_pool;
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
    C(1,     vec4 vals = imageLoad(out_img[plane], pos);           )
    C(1,     switch (plane) {                                      )
    C(2,         case 0:                                           )
    C(3,             vals.x = float(val);                          )
    C(3,             break;                                        )
    C(2,         case 1:                                           )
    C(3,             vals.y = float(val);                          )
    C(3,             break;                                        )
    C(2,         case 2:                                           )
    C(3,             vals.z = float(val);                          )
    C(3,             break;                                        )
    C(2,         default:                                          )
    C(3,             break;                                        )
    C(1,     }                                                     )
    C(1,     imageStore(out_img[plane], pos, vals);                )
    C(0, }                                                         )
};

static void free_common(AVCodecContext *avctx)
{
    DiracVulkanDecodeContext *ctx = avctx->internal->hwaccel_priv_data;
    FFVulkanContext *s = &ctx->vkctx;

    // ff_vk_exec_pool_free(&ctx->vkctx, &ctx->exec_pool);
    // ff_vk_exec_pool_free(&ctx->vkctx, &ctx->upload_exec_pool);
    // FFVulkanFunctions *vk = &ctx->vkctx.vkfn;

    // if (ctx->yuv_sampler)
    //     vk->DestroySamplerYcbcrConversion(s->hwctx->act_dev, ctx->yuv_sampler,
    //                                       s->hwctx->alloc);
    // if (ctx->sampler)
    //     vk->DestroySampler(s->hwctx->act_dev, ctx->sampler, s->hwctx->alloc);

    ff_vk_uninit(s);
}

static int init_quant_shd(DiracVulkanDecodeContext *s, FFVkSPIRVCompiler *spv)
{
    int err = 0;
    uint8_t *spv_data;
    size_t spv_len;
    void *spv_opaque = NULL;
    const int planes = av_pix_fmt_count_planes(s->vkctx.output_format);
    FFVulkanContext *vkctx = &s->vkctx;
    FFVkSPIRVShader *shd = &s->quant_shd;
    FFVulkanPipeline *pl = &s->quant_pl;
    FFVulkanDescriptorSetBinding *desc = (FFVulkanDescriptorSetBinding[])
    {
        {
            .name = "quant_in",
            .type = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
            .stages = VK_SHADER_STAGE_COMPUTE_BIT,
            .buf_content = "int inBuffer[];",
        },
        {
            .name       = "out_img",
            .type       = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE,
            .mem_layout = ff_vk_shader_rep_fmt(s->vkctx.output_format),
            .dimensions = 2,
            .elems      = planes,
            .stages     = VK_SHADER_STAGE_COMPUTE_BIT,
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
    DiracVulkanDecodeContext *dec = avctx->internal->hwaccel_priv_data;
    FFVulkanContext *s = &dec->vkctx;
    DiracContext *d = avctx->priv_data;
    if (d->hwaccel_picture_private) {
        av_free(d->hwaccel_picture_private);
    }

    /* Wait on and free execution pool */
    // if (dec->exec_pool.cmd_bufs) {
    //     ff_vk_exec_pool_free(&dec->vkctx, &dec->exec_pool);
    // }

    ff_vk_shader_free(s, &dec->quant_shd);
    // ff_vk_pipeline_free(s, &dec->quant_pl);

    // for (int i = 0; i < 7; i++) {
    //     ff_vk_pipeline_free(&dec->vkctx, &dec->wavelet_pl[i]);
    //     ff_vk_shader_free(&dec->vkctx, &dec->wavelet_shd[i]);
    // }

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

    nb_q = 0;
    qf = ff_vk_qf_init(s, &dec->upload_qf, VK_QUEUE_TRANSFER_BIT);

    if (s->query_props[qf].queryResultStatusSupport)
        nb_q = 1;

    err = ff_vk_exec_pool_init(s, &dec->upload_qf, &dec->upload_exec_pool, 2,
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

    init_quant_shd(dec, spv);

    av_log(avctx, AV_LOG_VERBOSE, "Vulkan decoder initialization sucessful\n");

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
    VkFormat vkfmt;
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

    hwfc->format[0]    = vkfmt;
    hwfc->tiling       = VK_IMAGE_TILING_OPTIMAL;
    hwfc->usage        = VK_IMAGE_USAGE_SAMPLED_BIT |
                         VK_IMAGE_USAGE_STORAGE_BIT |
                         VK_IMAGE_USAGE_TRANSFER_SRC_BIT |
                         VK_IMAGE_USAGE_TRANSFER_DST_BIT;

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
    int err;
    FFVulkanFunctions *vk = &dec->vkctx.vkfn;
    vkpic->wait_semaphores = vk->WaitSemaphores;
    vkpic->exec = ff_vk_exec_get(&dec->exec_pool);

    err = ff_vk_create_imageviews(&dec->vkctx, vkpic->exec, vkpic->views, vkpic->frame->avframe);
    if (err < 0) {
        return err;
    }

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
    DiracVulkanDecodeContext*dec = avctx->internal->hwaccel_priv_data;
    DiracContext *ctx = avctx->priv_data;
    DiracVulkanDecodePicture *pic = ctx->hwaccel_picture_private;

    av_buffer_pool_uninit(&pic->buf_pool);

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
    for (int i = 0; i < 2; i++) {
        dst_ctx->wavelet_pl[i] = src_ctx->wavelet_pl[i];
        dst_ctx->wavelet_shd[i] = src_ctx->wavelet_shd[i];
    }
    dst_ctx->quant_pl = src_ctx->quant_pl;
    dst_ctx->quant_shd = src_ctx->quant_shd;
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
//
// static int vulkan_dirac_decode_slice(AVCodecContext *avctx,
//                                const uint8_t  *data,
//                                uint32_t        size)
// {
//     int i, err;
//     DiracContext *s = avctx->priv_data;
//     uint8_t *thread_buf = &s->thread_buf[s->thread_buf_size*size];
//     DiracSlice *slices = (DiracSlice *)data;
//     for (i = 0; i < s->num_x; i++) {
//         err = decode_hq_slice(s, &slices[i], thread_buf);
//         if (err < 0)
//             return err;
//     }
//
//     return 0;
// }

const FFHWAccel ff_dirac_vulkan_hwaccel = {
    .p.name                = "dirac_vulkan",
    .p.type                = AVMEDIA_TYPE_VIDEO,
    .p.id                  = AV_CODEC_ID_DIRAC,
    .p.pix_fmt             = AV_PIX_FMT_VULKAN,
    .start_frame           = &vulkan_dirac_start_frame,
    .end_frame             = &vulkan_dirac_end_frame,
    // .decode_slice          = &vulkan_dirac_decode_slice,
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
