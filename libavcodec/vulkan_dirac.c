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

    FFVulkanPipeline wavelet_pl;
    FFVkSPIRVShader wavelet_shd;

    FFVulkanPipeline quant_pl;
    FFVkSPIRVShader quant_shd;

    FFVkQueueFamilyCtx qf;
    FFVkExecPool exec_pool;

    uint8_t *quant_val_buf_ptr;
    int quant_val_buf_size;
    int thread_buf_size;
    FFVkBuffer quant_val_buf_host;

    int n_slice_bufs;
    uint8_t *slice_buf_ptr;
    int slice_buf_size;
    FFVkBuffer slice_buf_host;

    uint8_t *quant_buf_ptr;
    int quant_buf_size;
    FFVkBuffer quant_buf_host;
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
    C(0, void dequant(int plane, int idx, ivec2 pos, int qf, int qs) {        )
    C(1,     int val = inBuffer[idx];                                         )
    C(1,     if (val < 0) {                                                   )
    C(2,         val = -(((-val)*qf + qs) >> 2);                              )
    C(1,     } else if (val > 0) {                                            )
    C(2,         val = ((val*qf + qs) >> 2);                                  )
    C(1,     }                                                                )
    C(1,     imageStore(out_img[plane], pos, vec4(255) / 255.0);              )
    // C(1,     imageStore(out_img[plane], pos, vec4(1.0));              )
    C(0, }                                                                    )
};

static const char proc_slice[] = {
    C(0, void proc_slice(int slice_idx) {                                                         )
    C(1,     const Slice s = slices[slice_idx];                                                   )
    C(1,     const int base_idx = slice_idx << 4;                                                 )
    C(1,     int offs = s.off;                                                                    )
    C(1,     int i = 0;                                                                           )
    C(1,     for(int plane = 0; plane < 3; plane++) {                                             )
    C(2,        for(int level = 0; level < wavelet_depth; level++) {                              )
    C(3,        const SliceCoeffs c = s.coeffs[DWT_LEVELS * plane + level];                       )
    C(3,        int orient = level == 0 ? 0 : 1;                                                  )
    C(3,        for(; orient < 4; orient++) {                                                     )
    C(4,            int y = int(gl_GlobalInvocationID.y);                                         )
    C(4,            int qf = quantMatrix[base_idx + level * DWT_LEVELS + orient];                 )
    C(4,            int qs = quantMatrix[base_idx + (level + 4) * DWT_LEVELS + orient];           )
    C(4,            for(; y < c.tot_v; y += int(gl_NumWorkGroups.y)) {                            )
    C(5,               int x = int(gl_GlobalInvocationID.x);                                      )
    C(4,               for(; x < c.tot_h; x += int(gl_NumWorkGroups.x)) {                         )
    C(5,                   offs++;                                                                )
    C(5,                   dequant(plane, offs, ivec2(c.left + x, c.top + y), qf, qs);            )
    C(4,               }                                                                          )
    C(4,               }                                                                          )
    C(3,            }                                                                             )
    C(2,        }                                                                                 )
    C(1,    }                                                                                     )
    C(0, }                                                                                        )
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
    //     ff_vk_pipeline_free(&dec->vkctx, &dec->wavelet_pl[i]);
    //     ff_vk_shader_free(&dec->vkctx, &dec->wavelet_shd[i]);
    // }

    //
    // if (dec->yuv_sampler)
    //     vk->DestroySamplerYcbcrConversion(s->hwctx->act_dev, dec->yuv_sampler,
    //                                       s->hwctx->alloc);
    // if (dec->sampler)
    //     vk->DestroySampler(s->hwctx->act_dev, dec->sampler, s->hwctx->alloc);

    if (dec->quant_val_buf_ptr) {
        ff_vk_free_buf(&dec->vkctx, &dec->quant_val_buf_host);
        av_free(dec->quant_val_buf_ptr);
    }

    if (dec->slice_buf_ptr) {
        ff_vk_free_buf(&dec->vkctx, &dec->slice_buf_host);
        av_free(dec->slice_buf_ptr);
    }

    if (dec->quant_buf_ptr) {
        ff_vk_free_buf(&dec->vkctx, &dec->quant_buf_host);
        av_free(dec->quant_buf_ptr);
    }

    // ff_vk_uninit(s);
}

static inline int alloc_host_mapped_buf(DiracVulkanDecodeContext *dec, size_t req_size,
                                 uint8_t *mem, FFVkBuffer *buf) {
    int err;
    size_t offs;

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

    offs = (uintptr_t)mem % s->hprops.minImportedHostPointerAlignment;
    import_desc.pHostPointer = mem - offs;
    req_size = FFALIGN(offs + req_size,
                s->hprops.minImportedHostPointerAlignment);

    err = vk->GetMemoryHostPointerPropertiesEXT(s->hwctx->act_dev,
                                                import_desc.handleType,
                                                import_desc.pHostPointer,
                                                &p_props);

    err = ff_vk_create_buf(s, buf, req_size,
                            &create_desc,
                            &import_desc,
                            VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT |
                                VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
                            VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT |
                                VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);
    if (err < 0)
        return err;

    return 0;
}

static int alloc_slices_buf(DiracContext *ctx, DiracVulkanDecodeContext *dec) {
    int err, length = ctx->num_y * ctx->num_x;

    dec->n_slice_bufs = length;

    if (dec->slice_buf_ptr) {
        ff_vk_free_buf(&dec->vkctx, &dec->slice_buf_host);
    }

    dec->slice_buf_size = sizeof(DiracSliceVkBuf) * length;
    dec->slice_buf_ptr = av_realloc(dec->slice_buf_ptr, dec->slice_buf_size);
    if (!dec->slice_buf_ptr)
        return AVERROR(ENOMEM);

    err = alloc_host_mapped_buf(dec, dec->slice_buf_size,
                                dec->slice_buf_ptr, &dec->slice_buf_host);
    if (err < 0)
        return err;

    err = ff_vk_set_descriptor_buffer(&dec->vkctx, &dec->quant_pl,
                                    NULL, 1, 1, 0,
                                    dec->quant_buf_host.address,
                                    dec->quant_buf_host.size,
                                    VK_FORMAT_UNDEFINED);
    if (err < 0)
        return err;

    return 0;
}

static int alloc_dequant_buf(DiracContext *ctx, DiracVulkanDecodeContext *dec) {
    int err, length = ctx->num_y * ctx->num_x;

    if (dec->quant_buf_ptr) {
        ff_vk_free_buf(&dec->vkctx, &dec->quant_buf_host);
    }

    dec->n_slice_bufs = length;

    dec->quant_buf_size = sizeof(int32_t) * MAX_DWT_LEVELS * 8 * length;
    dec->quant_buf_ptr = av_realloc(dec->quant_buf_ptr, dec->quant_buf_size);
    if (!dec->quant_buf_ptr)
        return AVERROR(ENOMEM);

    err = alloc_host_mapped_buf(dec, dec->quant_buf_size,
                                dec->quant_buf_ptr, &dec->quant_buf_host);
    if (err < 0)
        return err;

    err = ff_vk_set_descriptor_buffer(&dec->vkctx, &dec->quant_pl,
                                    NULL, 1, 2, 0,
                                    dec->slice_buf_host.address,
                                    dec->slice_buf_host.size,
                                    VK_FORMAT_UNDEFINED);
    if (err < 0)
        return err;

    return 0;
}

static int alloc_quant_buf(DiracContext *ctx, DiracVulkanDecodeContext *dec) {
    int err, length = ctx->num_y * ctx->num_x;

    if (ctx->thread_buf_size < 0)
        return 0;

    if (dec->quant_val_buf_ptr) {
        ff_vk_free_buf(&dec->vkctx, &dec->quant_val_buf_host);
    }

    dec->thread_buf_size = ctx->thread_buf_size;

    dec->quant_val_buf_size = ctx->thread_buf_size * 3 * length;
    dec->quant_val_buf_ptr = av_realloc(dec->quant_val_buf_ptr, dec->quant_val_buf_size);
    if (!dec->quant_val_buf_ptr)
        return AVERROR(ENOMEM);

    err = alloc_host_mapped_buf(dec, dec->quant_val_buf_size,
                                dec->quant_val_buf_ptr, &dec->quant_val_buf_host);
    if (err < 0)
        return err;

    err = ff_vk_set_descriptor_buffer(&dec->vkctx, &dec->quant_pl,
                                    NULL, 1, 0, 0,
                                    dec->quant_val_buf_host.address,
                                    dec->quant_val_buf_host.size,
                                    VK_FORMAT_UNDEFINED);
    if (err < 0)
        return err;

    return 0;
}

static int init_quant_shd(DiracVulkanDecodeContext *s, FFVkSPIRVCompiler *spv)
{
    int err = 0;
    uint8_t *spv_data;
    size_t spv_len;
    void *spv_opaque = NULL;
    const int planes = av_pix_fmt_count_planes(s->vkctx.output_format);
    FFVulkanContext *vkctx = &s->vkctx;
    FFVulkanDescriptorSetBinding *desc;
    FFVkSPIRVShader *shd = &s->quant_shd;
    FFVulkanPipeline *pl = &s->quant_pl;
    FFVkExecPool *exec = &s->exec_pool;

    RET(ff_vk_shader_init(pl, shd, "dequant", VK_SHADER_STAGE_COMPUTE_BIT, 0));
    ff_vk_shader_set_compute_sizes(shd, 16, 16, 4);

    shd = &s->quant_shd;


    desc = (FFVulkanDescriptorSetBinding[])
    {
        {
          .name = "out_img",
          .type = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE,
          .mem_layout = ff_vk_shader_rep_fmt(s->vkctx.output_format),
          .mem_quali = "writeonly",
          .dimensions = 2,
          .elems = planes,
          .stages = VK_SHADER_STAGE_COMPUTE_BIT,
        },
    };
    RET(ff_vk_pipeline_descriptor_set_add(vkctx, pl, shd, desc, 1, 0, 0));

    GLSLC(0, struct SliceCoeffs { );
    GLSLC(1,     int left;        );
    GLSLC(1,     int top;         );
    GLSLC(1,     int tot_h;       );
    GLSLC(1,     int tot_v;       );
    GLSLC(1,     int tot;         );
    GLSLC(0, };                   );

    GLSLC(0, struct Slice {                                  );
    GLSLC(1,     int idx;                                    );
    GLSLC(1,     int off;                                    );
    GLSLF(1,     SliceCoeffs coeffs[%i];, MAX_DWT_LEVELS * 3 );
    GLSLC(0, };                                              );

    desc = (FFVulkanDescriptorSetBinding[])
    {
        {
            .name = "quant_in_buf",
            .type = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
            .stages = VK_SHADER_STAGE_COMPUTE_BIT,
            .buf_content = "int inBuffer[];",
            .mem_quali = "readonly",
        },
        {
            .name = "quant_vals_buf",
            .type = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
            .stages = VK_SHADER_STAGE_COMPUTE_BIT,
            .buf_content = "int quantMatrix[];",
            .mem_quali = "readonly",
        },
        {
            .name = "slices_buf",
            .type = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
            .stages = VK_SHADER_STAGE_COMPUTE_BIT,
            .buf_content = "Slice slices[];",
            .mem_quali = "readonly",
        },
    };
    RET(ff_vk_pipeline_descriptor_set_add(vkctx, pl, shd, desc, 3, 1, 0));

    ff_vk_add_push_constant(pl, 0, sizeof(SliceCoeffsPushConst), VK_SHADER_STAGE_COMPUTE_BIT);

    GLSLC(0, layout(push_constant, std430) uniform pushConstants {  );
    GLSLC(1,     int wavelet_depth;                                 );
    GLSLC(1,     int slices_num;                                    );
    GLSLC(0, };                                                     );
    GLSLC(0,                                                        );

    GLSLF(0, #define DWT_LEVELS %i, MAX_DWT_LEVELS                  );

    GLSLD(dequant);
    GLSLD(proc_slice);
    GLSLC(0, void main()                                                        );
    GLSLC(0, {                                                                  );
    GLSLC(1,    int idx = int(gl_GlobalInvocationID.z);                         );
    GLSLC(1,    for (; idx < 10; idx += int(gl_NumWorkGroups.z)) {              );
    GLSLC(2,        proc_slice(idx);                                            );
    GLSLC(1,    }                                                               );
    GLSLC(0, }                                                                  );
    RET(spv->compile_shader(spv, vkctx, shd, &spv_data, &spv_len, "main", &spv_opaque));
    RET(ff_vk_shader_create(vkctx, shd, spv_data, spv_len, "main"));
    RET(ff_vk_init_compute_pipeline(vkctx, pl, shd));
    RET(ff_vk_exec_pipeline_register(vkctx, exec, pl));

    av_log(&s->vkctx, AV_LOG_INFO, "N desc sets: %i\n", pl->nb_descriptor_sets);
    av_log(&s->vkctx, AV_LOG_INFO, "Bindings: %i\n", pl->desc_set[0].nb_bindings);

fail:
    if (spv_opaque)
        spv->free_shader(spv, &spv_opaque);

    return err;
}

static int vulkan_dirac_uninit(AVCodecContext *avctx) {
    // DiracContext *d = avctx->priv_data;
    // if (d->hwaccel_picture_private) {
    //     av_freep(d->hwaccel_picture_private);
    // }

    free_common(avctx);

    return 0;
}

static int vulkan_dirac_init(AVCodecContext *avctx)
{
    int err = 0;
    // VkResult ret;
    DiracVulkanDecodeContext *dec = avctx->internal->hwaccel_priv_data;
    FFVulkanContext *s;
    // FFVulkanFunctions *vk;
    FFVkSPIRVCompiler *spv;
    //
    // VkSamplerYcbcrConversionCreateInfo yuv_sampler_info = {
    //     .sType = VK_STRUCTURE_TYPE_SAMPLER_YCBCR_CONVERSION_CREATE_INFO,
    //     .components = ff_comp_identity_map,
    //     .ycbcrModel = VK_SAMPLER_YCBCR_MODEL_CONVERSION_RGB_IDENTITY,
    //     .ycbcrRange = avctx->color_range == AVCOL_RANGE_MPEG, /* Ignored */
    // };

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
    // vk = &dec->vkctx.vkfn;

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

    err = ff_vk_exec_pool_init(s, &dec->qf, &dec->exec_pool, 4, 0, 0, 0, NULL);

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

    err = ff_vk_init_sampler(&dec->vkctx, &dec->sampler, 1, VK_FILTER_LINEAR);
    if (err < 0) {
        goto fail;
    }

    av_log(avctx, AV_LOG_VERBOSE, "Vulkan decoder initialization sucessful\n");

    init_quant_shd(dec, spv);

    dec->quant_val_buf_ptr = NULL;
    dec->slice_buf_ptr = NULL;
    dec->quant_buf_ptr = NULL;
    dec->thread_buf_size = 0;
    dec->n_slice_bufs = 0;

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

    if (s->quant_buf_ptr == NULL ||
        s->slice_buf_ptr == NULL ||
        s->quant_val_buf_ptr == NULL ||
        c->num_x * c->num_y != s->n_slice_bufs) {
        err = alloc_dequant_buf(c, s);
        if (err < 0)
            return err;
        err = alloc_slices_buf(c, s);
        if (err < 0)
            return err;
        err = alloc_quant_buf(c, s);
        if (err < 0)
            return err;
    }

    if (s->thread_buf_size != c->thread_buf_size) {
        err = alloc_quant_buf(c, s);
        if (err < 0)
            return 0;
    }

    return 0;
}

static int vulkan_dirac_end_frame(AVCodecContext *avctx) {
    int err;
    DiracVulkanDecodeContext*dec = avctx->internal->hwaccel_priv_data;
    FFVulkanFunctions *vk = &dec->vkctx.vkfn;
    DiracContext *ctx = avctx->priv_data;
    DiracVulkanDecodePicture *pic = ctx->hwaccel_picture_private;
    VkImageView views[AV_NUM_DATA_POINTERS];
    VkImageMemoryBarrier2 img_bar[37];
    int nb_img_bar = 0;
    // VkImageMemoryBarrier2 buf_bar[37];
    // int nb_buf_bar = 0;

    FFVkExecContext *exec = ff_vk_exec_get(&dec->exec_pool);
    ff_vk_exec_start(&dec->vkctx, exec);

    ff_vk_update_push_exec(&dec->vkctx, exec, &dec->quant_pl,
                           VK_SHADER_STAGE_COMPUTE_BIT,
                           0, sizeof(SliceCoeffsPushConst), &(SliceCoeffsPushConst) {
                            .wavelet_depth = ctx->wavelet_depth,
                            .slices_num = ctx->num_x * ctx->num_y,
                           });

    err = ff_vk_exec_add_dep_frame(&dec->vkctx, exec, pic->frame->avframe,
                                 VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT,
                                 VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT);
    if (err < 0)
        goto fail;
    err = ff_vk_create_imageviews(&dec->vkctx, exec, views, pic->frame->avframe);
    if (err < 0)
        goto fail;
    ff_vk_update_descriptor_img_array(&dec->vkctx, &dec->quant_pl,
                                      exec, pic->frame->avframe, views, 0, 0,
                                      VK_IMAGE_LAYOUT_GENERAL,
                                      dec->sampler);

    ff_vk_frame_barrier(&dec->vkctx, exec, pic->frame->avframe,
                        img_bar, &nb_img_bar,
                        VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT,
                        VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT,
                        VK_ACCESS_SHADER_WRITE_BIT,
                        VK_IMAGE_LAYOUT_GENERAL,
                        VK_QUEUE_FAMILY_IGNORED);

    ff_vk_exec_bind_pipeline(&dec->vkctx, exec, &dec->quant_pl);

    vk->CmdPipelineBarrier2(exec->buf, &(VkDependencyInfo) {
            .sType = VK_STRUCTURE_TYPE_DEPENDENCY_INFO,
            .pImageMemoryBarriers = img_bar,
            .imageMemoryBarrierCount = nb_img_bar,
        });

    vk->CmdDispatch(exec->buf, 25, 25, ctx->num_y * ctx->num_x);

    return ff_vk_exec_submit(&dec->vkctx, exec);
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

static inline int decode_hq_slice(const DiracContext *s, int jobnr)
{
    int i, level, orientation, quant_idx;
    DiracVulkanDecodeContext *dec = s->avctx->internal->hwaccel_priv_data;
    int *qfactor = ((int *) dec->quant_buf_ptr) + jobnr * 8 * MAX_DWT_LEVELS;
    int *qoffset = qfactor + 4 * MAX_DWT_LEVELS;

    // int qfactor[MAX_DWT_LEVELS][4], qoffset[MAX_DWT_LEVELS][4];
    // uint8_t *tmp_buf = s->thread_buf;
    uint8_t *quant_val_base = (uint8_t *)dec->quant_val_buf_ptr;
    uint8_t *tmp_buf = &quant_val_base[s->thread_buf_size * 3 * jobnr];
    DiracSlice *slice = &s->slice_params_buf[jobnr];
    GetBitContext *gb = &slice->gb;
    DiracSliceVkBuf *slice_vk = ((DiracSliceVkBuf *)dec->slice_buf_ptr) + jobnr;
    SliceCoeffs *coeffs_num = slice_vk->slices;

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

    /* Luma + 2 Chroma planes */
    for (i = 0; i < 3; i++) {
        int coef_num, coef_par;
        int64_t length = s->highquality.size_scaler*get_bits(gb, 8);
        int64_t bits_end = get_bits_count(gb) + 8*length;
        const uint8_t *addr = align_get_bits(gb);

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

        coeffs_num += MAX_DWT_LEVELS;
        tmp_buf += length;
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
    int err;
    DiracContext *s = avctx->priv_data;
    DiracVulkanDecodeContext *dec = avctx->internal->hwaccel_priv_data;

    if (dec->thread_buf_size < s->thread_buf_size) {
        err = alloc_quant_buf(s, dec);
        if (err < 0)
            return 0;
    }

    avctx->execute2(avctx, decode_hq_slice_row, NULL, NULL, s->num_y);

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
