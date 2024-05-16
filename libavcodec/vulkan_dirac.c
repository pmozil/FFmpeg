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
#include "hwaccel_internal.h"
#include "libavfilter/vulkan_spirv.h"
#include "vulkan_decode.h"

typedef  struct DiracVulkanDecodeContext {
    DiracContext *dirac_ctx;

    FFVulkanContext vkctx;
    FFVkExecPool e;
    FFVkQueueFamilyCtx qf;
    VkSampler sampler;

    FFVulkanPipeline wavelet_pl[7];
    FFVkSPIRVShader wavelet_shd[7];

    FFVulkanPipeline quant_pl[2];
    FFVkSPIRVShader quant_shd[2];
} DiracVulkanDecodeContext;

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

int init_quant_shd(DiracVulkanDecodeContext *s, FFVkSPIRVCompiler *spv, int idx)
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
    RET(ff_vk_exec_pipeline_register(vkctx, &s->e, pl));

fail:
    if (spv_opaque)
        spv->free_shader(spv, &spv_opaque);

    return err;
}

int vulkan_dirac_uninit(AVCodecContext *avctx)
{
    DiracVulkanDecodeContext *s = avctx->internal->hwaccel_priv_data;
    FFVulkanContext *vkctx = &s->vkctx;
    FFVulkanFunctions *vk = &vkctx->vkfn;

    ff_vk_exec_pool_free(vkctx, &s->e);

    for (int i = 0; i < 2; i++)
    {
        ff_vk_pipeline_free(vkctx, &s->quant_pl[i]);
        ff_vk_shader_free(vkctx, &s->quant_shd[i]);
    }

    for (int i = 0; i < 7; i++)
    {
        ff_vk_pipeline_free(vkctx, &s->wavelet_pl[i]);
        ff_vk_shader_free(vkctx, &s->wavelet_shd[i]);
    }

    if (s->sampler)
      vk->DestroySampler(vkctx->hwctx->act_dev, s->sampler, vkctx->hwctx->alloc);

    ff_vk_uninit(&s->vkctx);
}

int vulkan_dirac_init(AVCodecContext *avctx)
{
    int err;
    DiracVulkanDecodeContext *ctx = avctx->internal->hwaccel_priv_data;
    FFVulkanContext *s = &ctx->vkctx;
    FFVulkanFunctions *vk = &s->vkfn;
    FFVkSPIRVCompiler *spv = ff_vk_spirv_init();
    if (!spv) {
        av_log(ctx, AV_LOG_ERROR, "Unable to initialize SPIR-V compiler!\n");
        return AVERROR_EXTERNAL;
    }


    s->frames_ref = av_buffer_ref(avctx->hw_frames_ctx);
    s->frames = (AVHWFramesContext *)s->frames_ref->data;
    s->hwfc = s->frames->hwctx;

    s->device = (AVHWDeviceContext *)s->frames->device_ref->data;
    s->hwctx = s->device->hwctx;

    RET(ff_vk_load_props(s));
    ff_vk_qf_init(s, &ctx->qf, VK_QUEUE_COMPUTE_BIT);
    RET(ff_vk_exec_pool_init(s, &ctx->qf, &ctx->e, ctx->qf.nb_queues * 4, 0, 0, 0,
                           NULL));
    RET(ff_vk_init_sampler(s, &ctx->sampler, 1, VK_FILTER_LINEAR));

    // init_quant_shd

    spv->uninit(&spv);
    return 0;
fail:
    ff_vk_decode_uninit(avctx);
    return err;
}

const FFHWAccel ff_av1_vulkan_hwaccel = {
    .p.name                = "av1_vulkan",
    .p.type                = AVMEDIA_TYPE_VIDEO,
    .p.id                  = AV_CODEC_ID_AV1,
    .p.pix_fmt             = AV_PIX_FMT_VULKAN,
    // .start_frame           = &vk_av1_start_frame,
    // .decode_slice          = &vk_av1_decode_slice,
    // .end_frame             = &vk_av1_end_frame,
    // .free_frame_priv       = &vk_av1_free_frame_priv,
    .init                  = &vulkan_dirac_init,
    // .update_thread_context = &ff_vk_update_thread_context,
    // .decode_params         = &ff_vk_params_invalidate,
    // .flush                 = &ff_vk_decode_flush,
    .uninit                = &vulkan_dirac_uninit,
    // .frame_params          = &ff_vk_frame_params,
    // .frame_priv_data_size  = sizeof(AV1VulkanDecodePicture),
    .priv_data_size        = sizeof(DiracVulkanDecodeContext),

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
};
