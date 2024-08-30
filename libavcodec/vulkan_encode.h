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

#ifndef AVCODEC_VULKAN_ENCODE_H
#define AVCODEC_VULKAN_ENCODE_H

#include "codec_id.h"
#include "internal.h"

#include "encode.h"
#include "hwconfig.h"

#include "vulkan_video.h"
#include "hw_base_encode.h"

typedef struct FFVulkanEncodeDescriptor {
    enum AVCodecID                   codec_id;
    FFVulkanExtensions               encode_extension;
    VkVideoCodecOperationFlagBitsKHR encode_op;

    VkExtensionProperties ext_props;
} FFVulkanEncodeDescriptor;

typedef struct FFVulkanEncodePicture {
    FFHWBaseEncodePicture  base;
    VkVideoPictureResourceInfoKHR dpb_res;
    VkVideoReferenceSlotInfoKHR dpb_slot;

    struct {
        VkImageView        view;
        VkImageAspectFlags aspect;
    } in;

    struct {
        VkImageView        view;
        VkImageAspectFlags aspect;
    } dpb;

    void                  *codec_layer;
    void                  *codec_rc_layer;

    FFVkExecContext       *exec;
    AVBufferRef           *pkt_buf;
} FFVulkanEncodePicture;

/**
 * Callback for writing stream-level headers.
 */
typedef int (*vkenc_cb_write_stream_headers)(AVCodecContext *avctx,
                                             uint8_t *data, size_t *data_len);

/**
 * Callback for initializing codec-specific picture headers.
 */
typedef int (*vkenc_cb_init_pic_headers)(AVCodecContext *avctx,
                                         FFVulkanEncodePicture *pic);

/**
 * Callback for writing alignment data.
 * Align is the value to align offset to.
 */
typedef int (*vkenc_cb_write_filler)(AVCodecContext *avctx, uint32_t filler,
                                     uint8_t *data, size_t *data_len);

/**
 * Callback for writing any extra units requested. data_len must be set
 * to the available size, and its value will be overwritten by the #bytes written
 * to the output buffer.
 */
typedef int (*vkenc_cb_write_extra_headers)(AVCodecContext *avctx,
                                            FFVulkanEncodePicture *pic,
                                            uint8_t *data, size_t *data_len);

typedef struct FFVulkanCodec {
    /**
     * Codec feature flags.
     */
    int flags;
/* Codec output packet without timestamp delay, which means the
 * output packet has same PTS and DTS. */
#define VK_ENC_FLAG_NO_DELAY 1 << 6

    /**
     * Size of the codec-specific picture struct.
     */
    size_t picture_priv_data_size;

    int (*init_profile)(AVCodecContext *avctx, VkVideoProfileInfoKHR *profile,
                        void *pnext);

    int (*init_seq_params)(AVCodecContext *avctx);

    int (*create_session_params)(AVCodecContext *avctx);

    int (*setup_rc)(AVCodecContext *avctx, FFHWBaseEncodePicture *pic,
                    VkVideoEncodeRateControlInfoKHR *rc_info,
                    VkVideoEncodeRateControlLayerInfoKHR *rc_layer);

    int (*setup_session_params)(AVCodecContext *avctx, FFHWBaseEncodePicture *pic,
                                VkVideoEncodeSessionParametersGetInfoKHR *params_info);

    int (*setup_pic_params)(AVCodecContext *avctx, FFHWBaseEncodePicture *pic,
                            VkVideoEncodeInfoKHR *encode_info);
} FFVulkanCodec;

typedef struct FFVkEncodeCommonOptions {
    int quality;
    int profile;
    int level;
    int async_depth;
    VkVideoEncodeUsageFlagBitsKHR usage;
    VkVideoEncodeContentFlagBitsKHR content;
    VkVideoEncodeTuningModeKHR tune;
    VkVideoEncodeRateControlModeFlagBitsKHR rc_mode;
} FFVkEncodeCommonOptions;

typedef struct FFVulkanEncodeContext {
    FFVulkanContext s;
    FFVkVideoCommon common;
    FFHWBaseEncodeContext base;
    const FFVulkanCodec *codec;

    AVBufferPool *buf_pool;

    VkFormat pic_format;

    FFVkEncodeCommonOptions opts;

    VkVideoProfileInfoKHR profile;
    VkVideoProfileListInfoKHR profile_list;
    VkVideoCapabilitiesKHR caps;
    VkVideoEncodeQualityLevelPropertiesKHR quality_props;
    VkVideoEncodeCapabilitiesKHR enc_caps;
    VkVideoEncodeUsageInfoKHR usage_info;

    FFVkQueueFamilyCtx qf_enc;
    FFVkExecPool enc_pool;
} FFVulkanEncodeContext;

#define VULKAN_ENCODE_COMMON_OPTIONS \
    { "quality", "Set encode quality (trades off against speed, higher is faster)", OFFSET(common.opts.quality), AV_OPT_TYPE_INT, { .i64 = 0 }, 0, INT_MAX, FLAGS }, \
    { "rc_mode", "Select rate control type", OFFSET(common.opts.rc_mode), AV_OPT_TYPE_INT, { .i64 = VK_VIDEO_ENCODE_RATE_CONTROL_MODE_DEFAULT_KHR }, 0, INT_MAX, FLAGS, "rc_mode" }, \
        { "default", NULL, 0, AV_OPT_TYPE_CONST, { .i64 = VK_VIDEO_ENCODE_RATE_CONTROL_MODE_DEFAULT_KHR      }, INT_MIN, INT_MAX, FLAGS, "rc_mode" }, \
        { "cqp",     NULL, 0, AV_OPT_TYPE_CONST, { .i64 = VK_VIDEO_ENCODE_RATE_CONTROL_MODE_DISABLED_BIT_KHR }, INT_MIN, INT_MAX, FLAGS, "rc_mode" }, \
        { "cbr",     NULL, 0, AV_OPT_TYPE_CONST, { .i64 = VK_VIDEO_ENCODE_RATE_CONTROL_MODE_CBR_BIT_KHR      }, INT_MIN, INT_MAX, FLAGS, "rc_mode" }, \
        { "vbr",     NULL, 0, AV_OPT_TYPE_CONST, { .i64 = VK_VIDEO_ENCODE_RATE_CONTROL_MODE_VBR_BIT_KHR      }, INT_MIN, INT_MAX, FLAGS, "rc_mode" }, \
    { "tune", "Select tuning type", OFFSET(common.opts.tune), AV_OPT_TYPE_INT, { .i64 = VK_VIDEO_ENCODE_TUNING_MODE_DEFAULT_KHR }, 0, INT_MAX, FLAGS, "tune" }, \
        { "default",  NULL, 0, AV_OPT_TYPE_CONST, { .i64 = VK_VIDEO_ENCODE_TUNING_MODE_DEFAULT_KHR           }, INT_MIN, INT_MAX, FLAGS, "tune" }, \
        { "hq",       NULL, 0, AV_OPT_TYPE_CONST, { .i64 = VK_VIDEO_ENCODE_TUNING_MODE_HIGH_QUALITY_KHR      }, INT_MIN, INT_MAX, FLAGS, "tune" }, \
        { "ll",       NULL, 0, AV_OPT_TYPE_CONST, { .i64 = VK_VIDEO_ENCODE_TUNING_MODE_LOW_LATENCY_KHR       }, INT_MIN, INT_MAX, FLAGS, "tune" }, \
        { "ull",      NULL, 0, AV_OPT_TYPE_CONST, { .i64 = VK_VIDEO_ENCODE_TUNING_MODE_ULTRA_LOW_LATENCY_KHR }, INT_MIN, INT_MAX, FLAGS, "tune" }, \
        { "lossless", NULL, 0, AV_OPT_TYPE_CONST, { .i64 = VK_VIDEO_ENCODE_TUNING_MODE_LOSSLESS_KHR          }, INT_MIN, INT_MAX, FLAGS, "tune" }, \
    { "usage", "Select usage type", OFFSET(common.opts.usage), AV_OPT_TYPE_FLAGS, { .i64 = VK_VIDEO_DECODE_USAGE_DEFAULT_KHR }, 0, INT_MAX, FLAGS, "usage" }, \
        { "default",    NULL, 0, AV_OPT_TYPE_CONST, { .i64 = VK_VIDEO_DECODE_USAGE_DEFAULT_KHR          }, INT_MIN, INT_MAX, FLAGS, "usage" }, \
        { "transcode",  NULL, 0, AV_OPT_TYPE_CONST, { .i64 = VK_VIDEO_ENCODE_USAGE_TRANSCODING_BIT_KHR  }, INT_MIN, INT_MAX, FLAGS, "usage" }, \
        { "stream",     NULL, 0, AV_OPT_TYPE_CONST, { .i64 = VK_VIDEO_ENCODE_USAGE_STREAMING_BIT_KHR    }, INT_MIN, INT_MAX, FLAGS, "usage" }, \
        { "record",     NULL, 0, AV_OPT_TYPE_CONST, { .i64 = VK_VIDEO_ENCODE_USAGE_RECORDING_BIT_KHR    }, INT_MIN, INT_MAX, FLAGS, "usage" }, \
        { "conference", NULL, 0, AV_OPT_TYPE_CONST, { .i64 = VK_VIDEO_ENCODE_USAGE_CONFERENCING_BIT_KHR }, INT_MIN, INT_MAX, FLAGS, "usage" }, \
    { "content", "Select content type", OFFSET(common.opts.content), AV_OPT_TYPE_FLAGS, { .i64 = VK_VIDEO_ENCODE_CONTENT_DEFAULT_KHR }, 0, INT_MAX, FLAGS, "content" }, \
        { "default",  NULL, 0, AV_OPT_TYPE_CONST, { .i64 = VK_VIDEO_ENCODE_CONTENT_DEFAULT_KHR      }, INT_MIN, INT_MAX, FLAGS, "content" }, \
        { "camera",   NULL, 0, AV_OPT_TYPE_CONST, { .i64 = VK_VIDEO_ENCODE_CONTENT_CAMERA_BIT_KHR   }, INT_MIN, INT_MAX, FLAGS, "content" }, \
        { "desktop",  NULL, 0, AV_OPT_TYPE_CONST, { .i64 = VK_VIDEO_ENCODE_CONTENT_DESKTOP_BIT_KHR  }, INT_MIN, INT_MAX, FLAGS, "content" }, \
        { "rendered", NULL, 0, AV_OPT_TYPE_CONST, { .i64 = VK_VIDEO_ENCODE_CONTENT_RENDERED_BIT_KHR }, INT_MIN, INT_MAX, FLAGS, "content" }

/**
 * Initialize encoder.
 */
av_cold int ff_vulkan_encode_init(AVCodecContext *avctx, FFVulkanEncodeContext *ctx,
                                  void *caps, const FFVulkanEncodeDescriptor *vk_desc,
                                  const FFVulkanCodec *codec, void *quality_pnext);

/**
 * Encode.
 */
int ff_vulkan_encode_receive_packet(AVCodecContext *avctx, AVPacket *pkt);

/**
 * Uninitialize encoder.
 */
void ff_vulkan_encode_uninit(FFVulkanEncodeContext *ctx);

/**
 * Paperwork.
 */
extern const AVCodecHWConfigInternal *const ff_vulkan_encode_hw_configs[];

#endif /* AVCODEC_VULKAN_ENCODE_H */
