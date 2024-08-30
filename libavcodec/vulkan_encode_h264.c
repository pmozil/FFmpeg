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

#include "libavutil/opt.h"
#include "libavutil/mem.h"

#include "cbs.h"
#include "cbs_h264.h"
#include "h264_levels.h"
#include "h2645data.h"
#include "codec_internal.h"
#include "version.h"
#include "hw_base_encode_h264.h"

#include "vulkan_encode.h"

enum UnitElems {
    UNIT_AUD        = 1 << 0,
    UNIT_TIMING     = 1 << 1,
    UNIT_IDENTIFIER = 1 << 2,
    UNIT_RECOVERY   = 1 << 3,
};

/* Random (version 4) ISO 11578 UUID. */
static const uint8_t vulkan_encode_h264_sei_identifier_uuid[16] = {
    0x03, 0xfd, 0xf2, 0x0a, 0x5d, 0x4c, 0x05, 0x48,
    0x20, 0x98, 0xca, 0x6b, 0x0c, 0x95, 0x30, 0x1c,
};

typedef struct VulkanEncodeH264Context {
    FFVulkanEncodeContext common;
    FFHWBaseEncodeH264 units;
    FFHWBaseEncodeH264Opts unit_opts;

    enum UnitElems unit_elems;

    VkVideoSessionParametersKHR session_params;
    VkVideoEncodeH264SessionParametersGetInfoKHR session_params_info;

    StdVideoH264SequenceParameterSet    vksps;
    StdVideoH264ScalingLists            vksps_scaling;
    StdVideoH264HrdParameters           vksps_vui_header;
    StdVideoH264SequenceParameterSetVui vksps_vui;

    StdVideoH264PictureParameterSet     vkpps;
    StdVideoH264ScalingLists            vkpps_scaling;

    VkVideoEncodeH264ProfileInfoKHR profile;

    VkVideoEncodeH264CapabilitiesKHR caps;
    VkVideoEncodeH264QualityLevelPropertiesKHR quality_props;
} VulkanEncodeH264Context;

const FFVulkanEncodeDescriptor ff_vk_enc_h264_desc = {
    .codec_id         = AV_CODEC_ID_H264,
    .encode_extension = FF_VK_EXT_VIDEO_ENCODE_H264,
    .encode_op        = VK_VIDEO_CODEC_OPERATION_ENCODE_H264_BIT_KHR,
    .ext_props = {
        .extensionName = VK_STD_VULKAN_VIDEO_CODEC_H264_ENCODE_EXTENSION_NAME,
        .specVersion   = VK_STD_VULKAN_VIDEO_CODEC_H264_ENCODE_SPEC_VERSION,
    },
};

typedef struct VulkanEncodeH264Picture {
    int frame_num;
    int64_t last_idr_frame;
    uint16_t idr_pic_id;
    int primary_pic_type;
    int slice_type;
    int pic_order_cnt;

    VkVideoEncodeH264RateControlInfoKHR vkrc_info;
    VkVideoEncodeH264RateControlLayerInfoKHR vkrc_layer_info;

    StdVideoEncodeH264WeightTable slice_wt;
    StdVideoEncodeH264SliceHeader slice_hdr;
    VkVideoEncodeH264NaluSliceInfoKHR vkslice;

    StdVideoEncodeH264PictureInfo   h264pic_info;
    VkVideoEncodeH264PictureInfoKHR vkh264pic_info;

    StdVideoEncodeH264ReferenceInfo h264dpb_info;
    VkVideoEncodeH264DpbSlotInfoKHR vkh264dpb_info;

    StdVideoEncodeH264RefListModEntry mods[MAX_REFERENCE_LIST_NUM][H264_MAX_RPLM_COUNT];
    StdVideoEncodeH264RefPicMarkingEntry mmco[H264_MAX_RPLM_COUNT];
    StdVideoEncodeH264ReferenceListsInfo ref_list_info;
} VulkanEncodeH264Picture;









#if 0
static int vulkan_encode_h264_add_nal(AVCodecContext *avctx,
                                      CodedBitstreamFragment *au,
                                      void *nal_unit)
{
    H264RawNALUnitHeader *header = nal_unit;

    int err = ff_cbs_insert_unit_content(au, -1,
                                         header->nal_unit_type, nal_unit, NULL);
    if (err < 0)
        av_log(avctx, AV_LOG_ERROR, "Failed to add NAL unit: "
               "type = %d.\n", header->nal_unit_type);

    return err;
}

static int vulkan_encode_h264_write_access_unit(AVCodecContext *avctx,
                                                uint8_t *data, size_t *data_len,
                                                CodedBitstreamFragment *au)
{
    VulkanEncodeH264Context *enc = avctx->priv_data;

    int err = ff_cbs_write_fragment_data(enc->cbc, au);
    if (err < 0) {
        av_log(avctx, AV_LOG_ERROR, "Failed to write packed header.\n");
        return err;
    }

    if (*data_len < au->data_size) {
        av_log(avctx, AV_LOG_ERROR, "Access unit too large: %zu < %zu.\n",
               *data_len, au->data_size);
        return AVERROR(ENOSPC);
    }

    memcpy(data, au->data, au->data_size);
    *data_len = au->data_size;

    return 0;
}

static int vulkan_encode_h264_write_sequence_header(AVCodecContext *avctx,
                                                    uint8_t *data, size_t *data_len)
{
    int err;
    VulkanEncodeH264Context *enc = avctx->priv_data;
    CodedBitstreamFragment   *au = &enc->current_access_unit;

    if (enc->write_units & UNIT_AUD) {
        err = vulkan_encode_h264_add_nal(avctx, au, &enc->raw_aud);
        if (err < 0)
            goto fail;
    }

    err = vulkan_encode_h264_add_nal(avctx, au, &enc->raw_sps);
    if (err < 0)
        goto fail;

    err = vulkan_encode_h264_add_nal(avctx, au, &enc->raw_pps);
    if (err < 0)
        goto fail;

    err = vulkan_encode_h264_write_access_unit(avctx, data, data_len, au);
fail:
    ff_cbs_fragment_reset(au);
    return err;
}

static int vulkan_encode_h264_write_filler(AVCodecContext *avctx, uint32_t filler,
                                           uint8_t *data, size_t *data_len)
{
    int err;
    VulkanEncodeH264Context *enc = avctx->priv_data;
    CodedBitstreamFragment   *au = &enc->current_access_unit;

    H264RawFiller raw_filler = {
        .nal_unit_header = {
            .nal_unit_type = H264_NAL_FILLER_DATA,
        },
        .filler_size = filler,
    };

    err = vulkan_encode_h264_add_nal(avctx, au, &raw_filler);
    if (err < 0)
        goto fail;

    err = vulkan_encode_h264_write_access_unit(avctx, data, data_len, au);
fail:
    ff_cbs_fragment_reset(au);
    return err;
}

static av_cold int vulkan_encode_h264_create_session(AVCodecContext *avctx)
{
    VkResult ret;
    VulkanEncodeH264Context *enc = avctx->priv_data;
    FFVulkanEncodeContext *ctx = &enc->common;
    FFVulkanContext *s = &ctx->s;
    FFVulkanFunctions *vk = &enc->vkenc.s.vkfn;

    VkVideoEncodeH264SessionParametersAddInfoKHR h264_params_info;
    VkVideoEncodeH264SessionParametersCreateInfoKHR h264_params;
    VkVideoSessionParametersCreateInfoKHR session_params_create;

    h264_params_info = (VkVideoEncodeH264SessionParametersAddInfoKHR) {
        .sType = VK_STRUCTURE_TYPE_VIDEO_ENCODE_H264_SESSION_PARAMETERS_ADD_INFO_KHR,
        .pStdSPSs = &enc->vksps,
        .stdSPSCount = 1,
        .pStdPPSs = &enc->vkpps,
        .stdPPSCount = 1,
    };
    h264_params = (VkVideoEncodeH264SessionParametersCreateInfoKHR) {
        .sType = VK_STRUCTURE_TYPE_VIDEO_ENCODE_H264_SESSION_PARAMETERS_CREATE_INFO_KHR,
        .maxStdSPSCount = 1,
        .maxStdPPSCount = 1,
        .pParametersAddInfo = &h264_params_info,
    };
    session_params_create = (VkVideoSessionParametersCreateInfoKHR) {
        .sType = VK_STRUCTURE_TYPE_VIDEO_SESSION_PARAMETERS_CREATE_INFO_KHR,
        .pNext = &h264_params,
        .videoSession = ctx->common.session,
        .videoSessionParametersTemplate = NULL,
    };

    /* Create session parameters */
    ret = vk->CreateVideoSessionParametersKHR(s->hwctx->act_dev, &session_params_create,
                                              s->hwctx->alloc,
                                              &enc->common.common.session_params);
    if (ret != VK_SUCCESS) {
        av_log(avctx, AV_LOG_ERROR, "Unable to create Vulkan video session parameters: %s!\n",
               ff_vk_ret2str(ret));
        return AVERROR_EXTERNAL;
    }

    return 0;
}

static int vulkan_encode_h264_init_pic_headers(AVCodecContext *avctx,
                                               FFVulkanEncodePicture *pic)
{
    VulkanEncodeH264Context    *enc = avctx->priv_data;
    VulkanEncodeH264Picture   *hpic = pic->priv_data;
    FFVulkanEncodePicture     *prev = pic->prev;
    VulkanEncodeH264Picture  *hprev = prev ? prev->priv_data : NULL;

    int qp = pic->qp;
    int cpb_delay;
    int dpb_delay;
    int primary_pic_type;
    int slice_type;

    if (pic->type == FF_VK_FRAME_KEY) {
        av_assert0(pic->display_order == pic->encode_order);

        hpic->frame_num      = 0;
        hpic->last_idr_frame = pic->display_order;
        hpic->idr_pic_id     = hprev ? hprev->idr_pic_id + 1 : 0;

        primary_pic_type = 0;
        slice_type       = 7; // SPEC: add slice types above 5
    } else {
        av_assert0(prev);

        hpic->frame_num      = hprev->frame_num + prev->is_reference;
        hpic->last_idr_frame = hprev->last_idr_frame;
        hpic->idr_pic_id     = hprev->idr_pic_id;

        /* SPEC: missing StdVideoH264PictureType entries */
        if (pic->type == FF_VK_FRAME_I) {
            slice_type       = 7;
            primary_pic_type = 0;
        } else if (pic->type == FF_VK_FRAME_P) {
            slice_type       = 5;
            primary_pic_type = 1;
        } else {
            slice_type       = 6;
            primary_pic_type = 2;
        }
    }

    hpic->pic_order_cnt = pic->display_order - hpic->last_idr_frame;
    if (enc->raw_sps.pic_order_cnt_type == 2)
        hpic->pic_order_cnt *= 2;

    dpb_delay     = pic->display_order - pic->encode_order + enc->max_b_depth;
    cpb_delay     = pic->encode_order - hpic->last_idr_frame;

    enc->write_units = 0x0;

    if (pic->display_order == 0 && enc->insert_units & UNIT_IDENTIFIER)
        enc->write_units |= UNIT_IDENTIFIER;

    if (enc->insert_units & UNIT_AUD) {
        enc->raw_aud = (H264RawAUD) {
            .nal_unit_header = {
                .nal_unit_type = H264_NAL_AUD,
            },
            .primary_pic_type = primary_pic_type,
        };
        enc->write_units |= UNIT_AUD;
    }
    if (enc->insert_units & UNIT_TIMING) {
        enc->sei_pic_timing = (H264RawSEIPicTiming) {
            .cpb_removal_delay = 2 * cpb_delay,
            .dpb_output_delay  = 2 * dpb_delay,
        };
        enc->write_units |= UNIT_TIMING;
    }
    if (enc->insert_units & UNIT_RECOVERY && pic->type == FF_VK_FRAME_I) {
        enc->sei_recovery_point = (H264RawSEIRecoveryPoint) {
            .recovery_frame_cnt = 0,
            .exact_match_flag   = 1,
            .broken_link_flag   = enc->b_per_p > 0,
        };
        enc->write_units |= UNIT_RECOVERY;
    }

    hpic->slice_wt = (StdVideoEncodeH264WeightTable) {
        .flags = (StdVideoEncodeH264WeightTableFlags) {
            .luma_weight_l0_flag = 0,
            .chroma_weight_l0_flag = 0,
            .luma_weight_l1_flag = 0,
            .chroma_weight_l1_flag = 0,
        },
        .luma_log2_weight_denom = 0,
        .chroma_log2_weight_denom = 0,
        .luma_weight_l0 = { 0 },
        .luma_offset_l0 = { 0 },
        .chroma_weight_l0 = { { 0 } },
        .chroma_offset_l0 = { { 0 } },
        .luma_weight_l1 = { 0 },
        .luma_offset_l1 = { 0 },
        .chroma_weight_l1 = { { 0 } },
        .chroma_offset_l1 = { { 0 } },
    };

    hpic->slice_hdr = (StdVideoEncodeH264SliceHeader) {
        .flags = (StdVideoEncodeH264SliceHeaderFlags) {
            .direct_spatial_mv_pred_flag = 0,
            .num_ref_idx_active_override_flag = 0,
        },
        .first_mb_in_slice = 0,
        .slice_type = slice_type,
        .cabac_init_idc = 0,
        .disable_deblocking_filter_idc = 1,
        .slice_alpha_c0_offset_div2 = 0,
        .slice_beta_offset_div2 = 0,
        .pWeightTable = &hpic->slice_wt,
    };

    hpic->vkslice = (VkVideoEncodeH264NaluSliceInfoKHR) {
        .sType = VK_STRUCTURE_TYPE_VIDEO_ENCODE_H264_NALU_SLICE_INFO_KHR,
        .pNext = NULL,
        .pStdSliceHeader = &hpic->slice_hdr,
    };

    hpic->h264pic_info = (StdVideoEncodeH264PictureInfo) {
        .flags = (StdVideoEncodeH264PictureInfoFlags) {
            .IdrPicFlag = pic->type == FF_VK_FRAME_KEY,
            .is_reference = pic->is_reference,
            .no_output_of_prior_pics_flag = 0,
            .long_term_reference_flag = 0,
            .adaptive_ref_pic_marking_mode_flag = 0,
            /* Reserved */
        },
        .seq_parameter_set_id = enc->raw_sps.seq_parameter_set_id,
        .pic_parameter_set_id = enc->raw_pps.pic_parameter_set_id,
        .idr_pic_id = hpic->idr_pic_id,
        .primary_pic_type = pic->type == FF_VK_FRAME_P ? STD_VIDEO_H264_PICTURE_TYPE_P :
                            pic->type == FF_VK_FRAME_B ? STD_VIDEO_H264_PICTURE_TYPE_B :
                            pic->type == FF_VK_FRAME_I ? STD_VIDEO_H264_PICTURE_TYPE_I :
                                                         STD_VIDEO_H264_PICTURE_TYPE_IDR,
        .frame_num = hpic->frame_num,
        .PicOrderCnt = hpic->pic_order_cnt,
        .temporal_id = 0, /* ? */
        .pRefLists = &hpic->ref_list_info,
    };

    hpic->ref_list_info = (StdVideoEncodeH264ReferenceListsInfo) {
        .flags                    = (StdVideoEncodeH264ReferenceListsInfoFlags) {
            .ref_pic_list_modification_flag_l0 = 0,
            .ref_pic_list_modification_flag_l1 = 0,
        },
        .pRefList0ModOperations   = hpic->l0mods,
        .refList0ModOpCount       = 0,
        .pRefList1ModOperations   = hpic->l1mods,
        .refList1ModOpCount       = 0,
        .pRefPicMarkingOperations = hpic->marks,
        .refPicMarkingOpCount     = 0,
    };

    for (int i = 0; i < pic->nb_refs; i++) {
        FFVulkanEncodePicture *ref = pic->refs[i];
        VulkanEncodeH264Picture *href = ref->priv_data;

        hpic->l0ref_info[0] = (StdVideoEncodeH264ReferenceInfo) {
            .flags = (StdVideoEncodeH264ReferenceInfoFlags) {
                .used_for_long_term_reference = 0,
            },
            .FrameNum = href->frame_num,
            .PicOrderCnt = href->pic_order_cnt,
            .long_term_pic_num = 0,
            .long_term_frame_idx = 0,
        };

        hpic->l0refs[i] = (VkVideoEncodeH264DpbSlotInfoKHR) {
            .sType = VK_STRUCTURE_TYPE_VIDEO_ENCODE_H264_DPB_SLOT_INFO_KHR,
            .pStdReferenceInfo = &hpic->l0ref_info[i],
        };

        pic->ref_data[i] = &hpic->l0refs[i];
    }

    hpic->vkh264pic_info = (VkVideoEncodeH264PictureInfoKHR) {
        .sType = VK_STRUCTURE_TYPE_VIDEO_ENCODE_H264_PICTURE_INFO_KHR,
        .pNext = NULL,
        .naluSliceEntryCount = 1,
        .pNaluSliceEntries = &hpic->vkslice,
        .pStdPictureInfo = &hpic->h264pic_info,
    };

    hpic->vkrc_info = (VkVideoEncodeH264RateControlInfoKHR) {
        .sType = VK_STRUCTURE_TYPE_VIDEO_ENCODE_H264_RATE_CONTROL_INFO_KHR,
        .temporalLayerCount = 1,
    };

    hpic->vkrc_layer_info = (VkVideoEncodeH264RateControlLayerInfoKHR) {
        .sType = VK_STRUCTURE_TYPE_VIDEO_ENCODE_H264_RATE_CONTROL_LAYER_INFO_KHR,
        .minQp = (VkVideoEncodeH264QpKHR){ qp, qp, qp },
        .maxQp = (VkVideoEncodeH264QpKHR){ qp, qp, qp },
        .useMinQp = 1,
        .useMaxQp = 1,
    };

    pic->codec_info     = &hpic->vkh264pic_info;
    pic->codec_layer    = &hpic->vkrc_info;
    pic->codec_rc_layer = &hpic->vkrc_layer_info;

    return 0;
}

static int vulkan_encode_h264_write_extra_headers(AVCodecContext *avctx,
                                                  FFVulkanEncodePicture *pic,
                                                  uint8_t *data, size_t *data_len)
{
    int err;
    VulkanEncodeH264Context *enc = avctx->priv_data;
    CodedBitstreamFragment   *au = &enc->current_access_unit;

    if (enc->write_units) {
        if (enc->write_units & UNIT_AUD) {
            err = vulkan_encode_h264_add_nal(avctx, au, &enc->raw_aud);
            if (err < 0)
                goto fail;
        }

        if (enc->write_units & UNIT_IDENTIFIER) {
            err = ff_cbs_sei_add_message(enc->cbc, au, 1,
                                         SEI_TYPE_USER_DATA_UNREGISTERED,
                                         &enc->sei_identifier, NULL);
            if (err < 0)
                goto fail;
        }
        if (enc->write_units & UNIT_TIMING) {
            if (pic->type == FF_VK_FRAME_KEY) {
                err = ff_cbs_sei_add_message(enc->cbc, au, 1,
                                             SEI_TYPE_BUFFERING_PERIOD,
                                             &enc->sei_buffering_period, NULL);
                if (err < 0)
                    goto fail;
            }
            err = ff_cbs_sei_add_message(enc->cbc, au, 1,
                                         SEI_TYPE_PIC_TIMING,
                                         &enc->sei_pic_timing, NULL);
            if (err < 0)
                goto fail;
        }
        if (enc->write_units & UNIT_RECOVERY) {
            err = ff_cbs_sei_add_message(enc->cbc, au, 1,
                                         SEI_TYPE_RECOVERY_POINT,
                                         &enc->sei_recovery_point, NULL);
            if (err < 0)
                goto fail;
        }

        err = vulkan_encode_h264_write_access_unit(avctx, data, data_len, au);
        if (err < 0)
            goto fail;

        ff_cbs_fragment_reset(au);

        return 0;
    }

fail:
    ff_cbs_fragment_reset(au);
    return err;
}

static const FFVulkanEncoder encoder = {
    .pic_priv_data_size = sizeof(VulkanEncodeH264Picture),
    .write_stream_headers = vulkan_encode_h264_write_sequence_header,
    .init_pic_headers = vulkan_encode_h264_init_pic_headers,
    .write_filler = vulkan_encode_h264_write_filler,
    .filler_header_size = 6,
    .write_extra_headers = vulkan_encode_h264_write_extra_headers,
};

static av_cold int vulkan_encode_h264_init(AVCodecContext *avctx)
{
    int err;
    VulkanEncodeH264Context *enc = avctx->priv_data;

    enc->profile = (VkVideoEncodeH264ProfileInfoKHR) {
        .sType = VK_STRUCTURE_TYPE_VIDEO_ENCODE_H264_PROFILE_INFO_KHR,
        .stdProfileIdc = avctx->profile,
    };

    enc->caps = (VkVideoEncodeH264CapabilitiesKHR) {
        .sType = VK_STRUCTURE_TYPE_VIDEO_ENCODE_H264_CAPABILITIES_KHR,
    };

    err = ff_cbs_init(&enc->cbc, AV_CODEC_ID_H264, avctx);
    if (err < 0)
        return err;

    enc->mb_width  = FFALIGN(avctx->width,  16) / 16;
    enc->mb_height = FFALIGN(avctx->height, 16) / 16;

    enc->bit_rate    = avctx->bit_rate;
    enc->gop_size    = 3; /* avctx->gop_size; */
    enc->b_per_p     = 0; /* avctx->max_b_frames; */
    enc->max_b_depth = 0; /* FFMIN(enc->desired_b_depth,
                             av_log2(enc->b_per_p) + 1); */

    enc->vkenc.gop_size = enc->gop_size;
    enc->vkenc.bitrate =enc->bit_rate;

    err = ff_vulkan_encode_init(avctx, &enc->vkenc, &enc->profile, &enc->caps,
                                &encoder, &ff_vk_enc_h264_desc,
                                enc->b_per_p, enc->max_b_depth);
    if (err < 0)
        return err;

    if (enc->insert_units & UNIT_IDENTIFIER) {
        int len;

        memcpy(enc->sei_identifier.uuid_iso_iec_11578,
               vulkan_encode_h264_sei_identifier_uuid,
               sizeof(enc->sei_identifier.uuid_iso_iec_11578));

        len = snprintf(NULL, 0,
                       "%s / Vulkan video %i.%i.%i / %s %i.%i.%i / %s",
                       LIBAVCODEC_IDENT,
                       CODEC_VER(ff_vk_enc_h264_desc.ext_props.specVersion),
                       enc->vkenc.s.driver_props.driverName,
                       CODEC_VER(enc->vkenc.s.props.properties.driverVersion),
                       enc->vkenc.s.props.properties.deviceName);

        if (len >= 0) {
            enc->sei_identifier_string = av_malloc(len + 1);
            if (!enc->sei_identifier_string)
                return AVERROR(ENOMEM);

            len = snprintf(enc->sei_identifier_string, len + 1,
                           "%s / Vulkan video %i.%i.%i / %s %i.%i.%i / %s",
                           LIBAVCODEC_IDENT,
                           CODEC_VER(ff_vk_enc_h264_desc.ext_props.specVersion),
                           enc->vkenc.s.driver_props.driverName,
                           CODEC_VER(enc->vkenc.s.props.properties.driverVersion),
                           enc->vkenc.s.props.properties.deviceName);

            enc->sei_identifier.data        = enc->sei_identifier_string;
            enc->sei_identifier.data_length = len + 1;
        }
    }

    err = vulkan_encode_h264_init_seq_params(avctx);
    if (err < 0)
        return err;

    err = vulkan_encode_h264_create_session(avctx);
    if (err < 0)
        return err;

    if (avctx->flags & AV_CODEC_FLAG_GLOBAL_HEADER) {
        uint8_t data[4096];
        size_t data_len = sizeof(data);

        err = vulkan_encode_h264_write_sequence_header(avctx, data, &data_len);
        if (err < 0) {
            av_log(avctx, AV_LOG_ERROR, "Failed to write sequence header "
                   "for extradata: %d.\n", err);
            return err;
        } else {
            avctx->extradata_size = data_len;
            avctx->extradata = av_mallocz(avctx->extradata_size +
                                          AV_INPUT_BUFFER_PADDING_SIZE);
            if (!avctx->extradata) {
                err = AVERROR(ENOMEM);
                return err;
            }
            memcpy(avctx->extradata, data, avctx->extradata_size);
        }
    }

    return 0;
}

static av_cold int vulkan_encode_h264_close(AVCodecContext *avctx)
{
    VulkanEncodeH264Context *enc = avctx->priv_data;
    ff_vulkan_encode_uninit(&enc->vkenc);
    return 0;
}

static void vulkan_encode_h264_flush(AVCodecContext *avctx)
{

}

#define OFFSET(x) offsetof(VulkanEncodeH264Context, x)
#define FLAGS (AV_OPT_FLAG_VIDEO_PARAM | AV_OPT_FLAG_ENCODING_PARAM)
static const AVOption vulkan_encode_h264_options[] = {
    { "profile", "Select profile", OFFSET(vkenc.opts.profile), AV_OPT_TYPE_INT, { .i64 = FF_PROFILE_H264_MAIN }, 0, INT_MAX, FLAGS, "profile" },
        { "baseline", NULL, 0, AV_OPT_TYPE_CONST, { .i64 = FF_PROFILE_H264_BASELINE            }, INT_MIN, INT_MAX, FLAGS, "profile" },
        { "main",     NULL, 0, AV_OPT_TYPE_CONST, { .i64 = FF_PROFILE_H264_MAIN                }, INT_MIN, INT_MAX, FLAGS, "profile" },
        { "high",     NULL, 0, AV_OPT_TYPE_CONST, { .i64 = FF_PROFILE_H264_HIGH                }, INT_MIN, INT_MAX, FLAGS, "profile" },
        { "high444p", NULL, 0, AV_OPT_TYPE_CONST, { .i64 = FF_PROFILE_H264_HIGH_444_PREDICTIVE }, INT_MIN, INT_MAX, FLAGS, "profile" },

    { "b_depth", "Maximum B-frame reference depth", OFFSET(desired_b_depth), AV_OPT_TYPE_INT, { .i64 = 1 }, 1, INT_MAX, FLAGS },

    FF_VK_ENCODE_COMMON_OPTS

    { NULL },
};
#endif





static int vk_enc_h264_setup_rc(AVCodecContext *avctx, FFHWBaseEncodePicture *pic,
                                VkVideoEncodeRateControlInfoKHR *rc_info,
                                VkVideoEncodeRateControlLayerInfoKHR *rc_layer)
{
    FFVulkanEncodeContext *ctx = avctx->priv_data;
    VulkanEncodeH264Picture *hp = pic->codec_priv;

    hp->vkrc_info = (VkVideoEncodeH264RateControlInfoKHR) {
        .sType = VK_STRUCTURE_TYPE_VIDEO_ENCODE_H264_RATE_CONTROL_INFO_KHR,
        .flags = VK_VIDEO_ENCODE_H264_RATE_CONTROL_REFERENCE_PATTERN_FLAT_BIT_KHR |
                 VK_VIDEO_ENCODE_H264_RATE_CONTROL_REGULAR_GOP_BIT_KHR,
        .idrPeriod = ctx->base.gop_size,
        .gopFrameCount = ctx->base.gop_size,
        .consecutiveBFrameCount = ctx->base.b_per_p + 1, // TODO: not sure
        .temporalLayerCount = 0,
    };
    rc_info->pNext = &hp->vkrc_info;

    if (rc_info->rateControlMode > VK_VIDEO_ENCODE_RATE_CONTROL_MODE_DISABLED_BIT_KHR) {
        hp->vkrc_layer_info = (VkVideoEncodeH264RateControlLayerInfoKHR) {
            .sType = VK_STRUCTURE_TYPE_VIDEO_ENCODE_H264_RATE_CONTROL_LAYER_INFO_KHR,

            .useMinQp = avctx->qmin > 0,
            .minQp.qpI = avctx->qmin > 0 ? avctx->qmin : 0,
            .minQp.qpP = avctx->qmin > 0 ? avctx->qmin : 0,
            .minQp.qpB = avctx->qmin > 0 ? avctx->qmin : 0,

            .useMaxQp = avctx->qmax > 0,
            .maxQp.qpI = avctx->qmax > 0 ? avctx->qmax : 0,
            .maxQp.qpP = avctx->qmax > 0 ? avctx->qmax : 0,
            .maxQp.qpB = avctx->qmax > 0 ? avctx->qmax : 0,

            .useMaxFrameSize = 0,
        };
        rc_layer->pNext = &hp->vkrc_layer_info;
        hp->vkrc_info.temporalLayerCount = 1;
    }

    return 0;
}

static void vk_enc_h264_update_pic_info(AVCodecContext *avctx,
                                        FFHWBaseEncodePicture *pic)
{
    int pic_order_cnt_type;
    FFVulkanEncodeContext     *ctx = avctx->priv_data;
    VulkanEncodeH264Picture    *hp = pic->codec_priv;
    FFHWBaseEncodePicture    *prev = pic->prev;
    VulkanEncodeH264Picture *hprev = prev ? prev->codec_priv : NULL;

    if (pic->type == FF_HW_PICTURE_TYPE_IDR) {
        av_assert0(pic->display_order == pic->encode_order);

        hp->frame_num      = 0;
        hp->last_idr_frame = pic->display_order;
        hp->idr_pic_id     = hprev ? hprev->idr_pic_id + 1 : 0;

        hp->primary_pic_type = 0;
        hp->slice_type       = 7;
    } else {
        av_assert0(prev);

        hp->frame_num = hprev->frame_num + prev->is_reference;

        hp->last_idr_frame = hprev->last_idr_frame;
        hp->idr_pic_id     = hprev->idr_pic_id;

        if (pic->type == FF_HW_PICTURE_TYPE_I) {
            hp->slice_type       = 7;
            hp->primary_pic_type = 0;
        } else if (pic->type == FF_HW_PICTURE_TYPE_P) {
            hp->slice_type       = 5;
            hp->primary_pic_type = 1;
        } else {
            hp->slice_type       = 6;
            hp->primary_pic_type = 2;
        }
    }

    hp->pic_order_cnt = pic->display_order - hp->last_idr_frame;
    pic_order_cnt_type = ctx->base.max_b_depth ? 0 : 2;
    if (pic_order_cnt_type == 2)
        hp->pic_order_cnt *= 2;
}

static void setup_slices(AVCodecContext *avctx,
                         FFHWBaseEncodePicture *pic)
{
    VulkanEncodeH264Picture *hp = pic->codec_priv;

    hp->slice_wt = (StdVideoEncodeH264WeightTable) {
        .flags = (StdVideoEncodeH264WeightTableFlags) {
            .luma_weight_l0_flag = 0,
            .chroma_weight_l0_flag = 0,
            .luma_weight_l1_flag = 0,
            .chroma_weight_l1_flag = 0,
        },
        .luma_log2_weight_denom = 0,
        .chroma_log2_weight_denom = 0,
        .luma_weight_l0 = { 0 },
        .luma_offset_l0 = { 0 },
        .chroma_weight_l0 = { { 0 } },
        .chroma_offset_l0 = { { 0 } },
        .luma_weight_l1 = { 0 },
        .luma_offset_l1 = { 0 },
        .chroma_weight_l1 = { { 0 } },
        .chroma_offset_l1 = { { 0 } },
    };

    hp->slice_hdr = (StdVideoEncodeH264SliceHeader) {
        .flags = (StdVideoEncodeH264SliceHeaderFlags) {
            .direct_spatial_mv_pred_flag = 1,
            .num_ref_idx_active_override_flag = 0,
        },
        .first_mb_in_slice = 1,
        .slice_type = hp->slice_type,
        .cabac_init_idc = 0,
        .disable_deblocking_filter_idc = 0,
        .slice_alpha_c0_offset_div2 = 0,
        .slice_beta_offset_div2 = 0,
        .pWeightTable = &hp->slice_wt,
    };

    hp->vkslice = (VkVideoEncodeH264NaluSliceInfoKHR) {
        .sType = VK_STRUCTURE_TYPE_VIDEO_ENCODE_H264_NALU_SLICE_INFO_KHR,
        .pNext = NULL,
        .pStdSliceHeader = &hp->slice_hdr,
    };

    hp->vkh264pic_info.pNaluSliceEntries = &hp->vkslice;
    hp->vkh264pic_info.naluSliceEntryCount = 1;
}

static void vk_enc_h264_default_ref_pic_list(AVCodecContext *avctx,
                                             FFHWBaseEncodePicture *pic,
                                             FFHWBaseEncodePicture **rpl0,
                                             FFHWBaseEncodePicture **rpl1,
                                             int *rpl_size)
{
    FFHWBaseEncodePicture *prev;
    VulkanEncodeH264Picture *hp, *hn, *hc;
    int i, j, n = 0;

    prev = pic->prev;
    av_assert0(prev);
    hp = pic->codec_priv;

    for (i = 0; i < pic->prev->nb_dpb_pics; i++) {
        hn = prev->dpb[i]->codec_priv;
        av_assert0(hn->frame_num < hp->frame_num);

        if (pic->type == FF_HW_PICTURE_TYPE_P) {
            for (j = n; j > 0; j--) {
                hc = rpl0[j - 1]->codec_priv;
                av_assert0(hc->frame_num != hn->frame_num);
                if (hc->frame_num > hn->frame_num)
                    break;
                rpl0[j] = rpl0[j - 1];
            }
            rpl0[j] = prev->dpb[i];

        } else if (pic->type == FF_HW_PICTURE_TYPE_B) {
            for (j = n; j > 0; j--) {
                hc = rpl0[j - 1]->codec_priv;
                av_assert0(hc->pic_order_cnt != hp->pic_order_cnt);
                if (hc->pic_order_cnt < hp->pic_order_cnt) {
                    if (hn->pic_order_cnt > hp->pic_order_cnt ||
                        hn->pic_order_cnt < hc->pic_order_cnt)
                        break;
                } else {
                    if (hn->pic_order_cnt > hc->pic_order_cnt)
                        break;
                }
                rpl0[j] = rpl0[j - 1];
            }
            rpl0[j] = prev->dpb[i];

            for (j = n; j > 0; j--) {
                hc = rpl1[j - 1]->codec_priv;
                av_assert0(hc->pic_order_cnt != hp->pic_order_cnt);
                if (hc->pic_order_cnt > hp->pic_order_cnt) {
                    if (hn->pic_order_cnt < hp->pic_order_cnt ||
                        hn->pic_order_cnt > hc->pic_order_cnt)
                        break;
                } else {
                    if (hn->pic_order_cnt < hc->pic_order_cnt)
                        break;
                }
                rpl1[j] = rpl1[j - 1];
            }
            rpl1[j] = prev->dpb[i];
        }

        ++n;
    }

    if (pic->type == FF_HW_PICTURE_TYPE_B) {
        for (i = 0; i < n; i++) {
            if (rpl0[i] != rpl1[i])
                break;
        }
        if (i == n)
            FFSWAP(FFHWBaseEncodePicture *, rpl1[0], rpl1[1]);
    }

    if (pic->type == FF_HW_PICTURE_TYPE_P ||
        pic->type == FF_HW_PICTURE_TYPE_B) {
        av_log(avctx, AV_LOG_DEBUG, "Default RefPicList0 for fn=%d/poc=%d:",
               hp->frame_num, hp->pic_order_cnt);
        for (i = 0; i < n; i++) {
            hn = rpl0[i]->codec_priv;
            av_log(avctx, AV_LOG_DEBUG, "  fn=%d/poc=%d",
                   hn->frame_num, hn->pic_order_cnt);
        }
        av_log(avctx, AV_LOG_DEBUG, "\n");
    }
    if (pic->type == FF_HW_PICTURE_TYPE_B) {
        av_log(avctx, AV_LOG_DEBUG, "Default RefPicList1 for fn=%d/poc=%d:",
               hp->frame_num, hp->pic_order_cnt);
        for (i = 0; i < n; i++) {
            hn = rpl1[i]->codec_priv;
            av_log(avctx, AV_LOG_DEBUG, "  fn=%d/poc=%d",
                   hn->frame_num, hn->pic_order_cnt);
        }
        av_log(avctx, AV_LOG_DEBUG, "\n");
    }

    *rpl_size = n;
}

static void setup_refs(AVCodecContext *avctx,
                       FFHWBaseEncodePicture *pic,
                       VkVideoEncodeInfoKHR *encode_info)
{
    int idx, n, i, j;
    VulkanEncodeH264Context *enc = avctx->priv_data;
    VulkanEncodeH264Picture *hp = pic->codec_priv;
    FFHWBaseEncodePicture *prev = pic->prev;
    FFHWBaseEncodePicture *def_l0[MAX_DPB_SIZE], *def_l1[MAX_DPB_SIZE];
    VulkanEncodeH264Picture *href;

    hp->ref_list_info = (StdVideoEncodeH264ReferenceListsInfo) {
        .flags = (StdVideoEncodeH264ReferenceListsInfoFlags) {
            .ref_pic_list_modification_flag_l0 = 0,
            .ref_pic_list_modification_flag_l1 = 0,
            /* Reserved */
        },
        .num_ref_idx_l0_active_minus1 = pic->nb_refs[0] - 1,
        .num_ref_idx_l1_active_minus1 = pic->nb_refs[1] - 1,
//        .RefPicList0 = WAT,
//        .RefPicList1 = WAT,
        /* Reserved */
        .pRefList0ModOperations = NULL, /* All set below */
        .refList0ModOpCount = 0,
        .pRefList1ModOperations = NULL,
        .refList1ModOpCount = 0,
        .pRefPicMarkingOperations = NULL,
        .refPicMarkingOpCount = 0,
    };

    for (i = 0; i < STD_VIDEO_H264_MAX_NUM_LIST_REF; i++)
        hp->ref_list_info.RefPicList0[i] = hp->ref_list_info.RefPicList1[i] = -1;

    idx = 0;
    for (int j = 0; j < pic->nb_refs[0]; j++) {
        FFHWBaseEncodePicture *ref = pic->refs[0][j];
        FFVulkanEncodePicture *rvp = ref->priv;
        VkVideoReferenceSlotInfoKHR *slot_info;
        slot_info = (VkVideoReferenceSlotInfoKHR *)&encode_info->pReferenceSlots[idx];

//        hp->ref_list_info.RefPicList0[j]

        idx++;
    }

    hp->h264pic_info.pRefLists = &hp->ref_list_info;

    if (pic->is_reference && pic->type != FF_HW_PICTURE_TYPE_IDR) {
        FFHWBaseEncodePicture *discard_list[MAX_DPB_SIZE];
        int discard = 0, keep = 0;

        // Discard everything which is in the DPB of the previous frame but
        // not in the DPB of this one.
        for (i = 0; i < prev->nb_dpb_pics; i++) {
            for (j = 0; j < pic->nb_dpb_pics; j++) {
                if (prev->dpb[i] == pic->dpb[j])
                    break;
            }
            if (j == pic->nb_dpb_pics) {
                discard_list[discard] = prev->dpb[i];
                ++discard;
            } else {
                ++keep;
            }
        }
        av_assert0(keep <= enc->units.dpb_frames);

        if (discard == 0) {
            hp->h264pic_info.flags.adaptive_ref_pic_marking_mode_flag = 0;
        } else {
            hp->h264pic_info.flags.adaptive_ref_pic_marking_mode_flag = 1;
            for (i = 0; i < discard; i++) {
                VulkanEncodeH264Picture *old = discard_list[i]->codec_priv;
                av_assert0(old->frame_num < hp->frame_num);
                hp->mmco[i] = (StdVideoEncodeH264RefPicMarkingEntry) {
                    .memory_management_control_operation = 1,
                    .difference_of_pic_nums_minus1 = hp->frame_num - old->frame_num - 1,
                };
            }
            hp->mmco[i] = (StdVideoEncodeH264RefPicMarkingEntry) {
                .memory_management_control_operation = 0,
            };
            hp->ref_list_info.pRefPicMarkingOperations = hp->mmco;
            hp->ref_list_info.refPicMarkingOpCount = i;
        }
    }

    if (pic->type == FF_HW_PICTURE_TYPE_I || pic->type == FF_HW_PICTURE_TYPE_IDR)
        return;

    // If the intended references are not the first entries of RefPicListN
    // by default, use ref-pic-list-modification to move them there.
    vk_enc_h264_default_ref_pic_list(avctx, pic,
                                     def_l0, def_l1, &n);

    if (pic->type == FF_HW_PICTURE_TYPE_P) {
        int need_rplm = 0;
        for (i = 0; i < pic->nb_refs[0]; i++) {
            av_assert0(pic->refs[0][i]);
            if (pic->refs[0][i] != (FFHWBaseEncodePicture *)def_l0[i])
                need_rplm = 1;
        }

        hp->ref_list_info.flags.ref_pic_list_modification_flag_l0 = need_rplm;
        if (need_rplm) {
            int pic_num = hp->frame_num;
            for (i = 0; i < pic->nb_refs[0]; i++) {
                href = pic->refs[0][i]->codec_priv;
                av_assert0(href->frame_num != pic_num);
                if (href->frame_num < pic_num) {
                    hp->mods[0][i] = (StdVideoEncodeH264RefListModEntry) {
                        .modification_of_pic_nums_idc = 0,
                        .abs_diff_pic_num_minus1 = pic_num - href->frame_num - 1,
                    };
                } else {
                    hp->mods[0][i] = (StdVideoEncodeH264RefListModEntry) {
                        .modification_of_pic_nums_idc = 1,
                        .abs_diff_pic_num_minus1 = href->frame_num - pic_num - 1,
                    };
                }
                pic_num = href->frame_num;
            }
            hp->mods[0][i] = (StdVideoEncodeH264RefListModEntry) {
                .modification_of_pic_nums_idc = 3,
            };
        }
    } else {
        int need_rplm_l0 = 0, need_rplm_l1 = 0;
        int n0 = 0, n1 = 0;
        for (i = 0; i < pic->nb_refs[0]; i++) {
            av_assert0(pic->refs[0][i]);
            href = pic->refs[0][i]->codec_priv;
            av_assert0(href->pic_order_cnt < hp->pic_order_cnt);
            if (pic->refs[0][i] != (FFHWBaseEncodePicture *)def_l0[n0])
                need_rplm_l0 = 1;
            ++n0;
        }

        for (int i = 0; i < pic->nb_refs[1]; i++) {
            av_assert0(pic->refs[1][i]);
            href = pic->refs[1][i]->codec_priv;
            av_assert0(href->pic_order_cnt > hp->pic_order_cnt);
            if (pic->refs[1][i] != (FFHWBaseEncodePicture *)def_l1[n1])
                need_rplm_l1 = 1;
            ++n1;
        }

        hp->ref_list_info.flags.ref_pic_list_modification_flag_l0 = need_rplm_l0;
        if (need_rplm_l0) {
            int pic_num = hp->frame_num;
            for (i = j = 0; i < pic->nb_refs[0]; i++) {
                href = pic->refs[0][i]->codec_priv;
                av_assert0(href->frame_num != pic_num);
                if (href->frame_num < pic_num) {
                    hp->mods[0][j] = (StdVideoEncodeH264RefListModEntry) {
                        .modification_of_pic_nums_idc = 0,
                        .abs_diff_pic_num_minus1 = pic_num - href->frame_num - 1,
                    };
                } else {
                    hp->mods[0][j] = (StdVideoEncodeH264RefListModEntry) {
                        .modification_of_pic_nums_idc = 1,
                        .abs_diff_pic_num_minus1 = href->frame_num - pic_num - 1,
                    };
                }
                pic_num = href->frame_num;
                ++j;
            }
            hp->mods[0][j] = (StdVideoEncodeH264RefListModEntry) {
                .modification_of_pic_nums_idc = 3,
            };
            hp->ref_list_info.pRefList0ModOperations = hp->mods[0];
            hp->ref_list_info.refList0ModOpCount = j;
        }

        hp->ref_list_info.flags.ref_pic_list_modification_flag_l1 = need_rplm_l1;
        if (need_rplm_l1) {
            int pic_num = hp->frame_num;
            for (i = j = 0; i < pic->nb_refs[1]; i++) {
                href = pic->refs[1][i]->codec_priv;
                av_assert0(href->frame_num != pic_num);
                if (href->frame_num < pic_num) {
                    hp->mods[1][j] = (StdVideoEncodeH264RefListModEntry) {
                        .modification_of_pic_nums_idc = 0,
                        .abs_diff_pic_num_minus1 = pic_num - href->frame_num - 1,
                    };
                } else {
                    hp->mods[1][j] = (StdVideoEncodeH264RefListModEntry) {
                        .modification_of_pic_nums_idc = 1,
                        .abs_diff_pic_num_minus1 = href->frame_num - pic_num - 1,
                    };
                }
                pic_num = href->frame_num;
                ++j;
            }
            hp->mods[1][j] = (StdVideoEncodeH264RefListModEntry) {
                .modification_of_pic_nums_idc = 3,
            };
            hp->ref_list_info.pRefList1ModOperations = hp->mods[1];
            hp->ref_list_info.refList1ModOpCount = j;
        }
    }
}

static int vk_enc_h264_setup_pic_params(AVCodecContext *avctx,
                                        FFHWBaseEncodePicture *pic,
                                        VkVideoEncodeInfoKHR *encode_info)
{
    FFVulkanEncodePicture *vp = pic->priv;
    VulkanEncodeH264Picture *hp = pic->codec_priv;
    VkVideoReferenceSlotInfoKHR *ref_slot;

    vk_enc_h264_update_pic_info(avctx, pic);

    hp->vkh264pic_info = (VkVideoEncodeH264PictureInfoKHR) {
        .sType = VK_STRUCTURE_TYPE_VIDEO_ENCODE_H264_PICTURE_INFO_KHR,
        .pNext = NULL,
        .pNaluSliceEntries = NULL, // Filled in during setup_slices()
        .naluSliceEntryCount = 0, // Filled in during setup_slices()
        .pStdPictureInfo = &hp->h264pic_info,
    };

    hp->h264pic_info = (StdVideoEncodeH264PictureInfo) {
        .flags = (StdVideoEncodeH264PictureInfoFlags) {
            .IdrPicFlag = pic->type == FF_HW_PICTURE_TYPE_IDR,
            .is_reference = pic->is_reference,
            .no_output_of_prior_pics_flag = 0,
            .long_term_reference_flag = 0,
            .adaptive_ref_pic_marking_mode_flag = 0, // Filled in during setup_refs()
            /* Reserved */
        },
        .seq_parameter_set_id = 0,
        .pic_parameter_set_id = 0,
        .idr_pic_id = hp->idr_pic_id,
        .primary_pic_type = pic->type == FF_HW_PICTURE_TYPE_P ? STD_VIDEO_H264_PICTURE_TYPE_P :
                            pic->type == FF_HW_PICTURE_TYPE_B ? STD_VIDEO_H264_PICTURE_TYPE_B :
                            pic->type == FF_HW_PICTURE_TYPE_I ? STD_VIDEO_H264_PICTURE_TYPE_I :
                                                                STD_VIDEO_H264_PICTURE_TYPE_IDR,
        .frame_num = hp->frame_num,
        .PicOrderCnt = hp->pic_order_cnt,
        .temporal_id = 0, /* ? */
        /* Reserved */
        .pRefLists = NULL, // Filled in during setup_refs
    };
    encode_info->pNext = &hp->vkh264pic_info;

    hp->h264dpb_info = (StdVideoEncodeH264ReferenceInfo) {
        .flags = (StdVideoEncodeH264ReferenceInfoFlags) {
            .used_for_long_term_reference = 0,
            /* Reserved */
        },
        .primary_pic_type = hp->h264pic_info.primary_pic_type,
        .FrameNum = hp->h264pic_info.frame_num,
        .PicOrderCnt = hp->h264pic_info.PicOrderCnt,
        .long_term_pic_num = 0,
        .long_term_frame_idx = 0,
        .temporal_id = hp->h264pic_info.temporal_id,
    };
    hp->vkh264dpb_info = (VkVideoEncodeH264DpbSlotInfoKHR) {
        .sType = VK_STRUCTURE_TYPE_VIDEO_ENCODE_H264_DPB_SLOT_INFO_KHR,
        .pStdReferenceInfo = &hp->h264dpb_info,
    };

    vp->dpb_slot.pNext = &hp->vkh264dpb_info;

    ref_slot = (VkVideoReferenceSlotInfoKHR *)encode_info->pSetupReferenceSlot;
    ref_slot->slotIndex = 0;
    ref_slot->pNext = &hp->vkh264dpb_info;

    setup_slices(avctx, pic);

    setup_refs(avctx, pic, encode_info);

    return 0;
}

static av_cold int vk_enc_h264_init_sequence_params(AVCodecContext *avctx)
{
    int err;
    VulkanEncodeH264Context *enc = avctx->priv_data;
    FFVulkanEncodeContext *ctx = &enc->common;
    FFHWBaseEncodeContext *base_ctx = &ctx->base;

    H264RawSPS                          *sps = &enc->units.raw_sps;
    H264RawHRD                          *hrd = &sps->vui.nal_hrd_parameters;
    StdVideoH264ScalingLists            *vksps_scaling = &enc->vksps_scaling;
    StdVideoH264HrdParameters           *vksps_vui_header = &enc->vksps_vui_header;
    StdVideoH264SequenceParameterSetVui *vksps_vui = &enc->vksps_vui;
    StdVideoH264SequenceParameterSet    *vksps = &enc->vksps;

    H264RawPPS                          *pps = &enc->units.raw_pps;
    StdVideoH264ScalingLists            *vkpps_scaling = &enc->vkpps_scaling;
    StdVideoH264PictureParameterSet     *vkpps = &enc->vkpps;

    enc->unit_opts.bit_rate = avctx->bit_rate;

    err = ff_hw_base_encode_init_params_h264(base_ctx, avctx,
                                             &enc->units, &enc->unit_opts);
    if (err < 0)
        return err;

    *vksps_scaling = (StdVideoH264ScalingLists) {
        .scaling_list_present_mask = sps->seq_scaling_matrix_present_flag,
        .use_default_scaling_matrix_mask = 1,
    };

    *vksps_vui_header = (StdVideoH264HrdParameters) {
        .cpb_cnt_minus1 = hrd->cpb_cnt_minus1,
        .bit_rate_scale = hrd->bit_rate_scale,
        .initial_cpb_removal_delay_length_minus1 = hrd->initial_cpb_removal_delay_length_minus1,
        .cpb_removal_delay_length_minus1 = hrd->cpb_removal_delay_length_minus1,
        .dpb_output_delay_length_minus1 = hrd->dpb_output_delay_length_minus1,
        .time_offset_length = hrd->time_offset_length,
    };

    for (int i = 0; i < H264_MAX_CPB_CNT; i++) {
        vksps_vui_header->bit_rate_value_minus1[i] = hrd->bit_rate_value_minus1[i];
        vksps_vui_header->cpb_size_value_minus1[i] = hrd->cpb_size_value_minus1[i];
        vksps_vui_header->cbr_flag[i] = hrd->cbr_flag[i];
    }

    *vksps_vui = (StdVideoH264SequenceParameterSetVui) {
        .aspect_ratio_idc = sps->vui.aspect_ratio_idc,
        .sar_width = sps->vui.sar_width,
        .sar_height = sps->vui.sar_height,
        .video_format = sps->vui.video_format,
        .colour_primaries = sps->vui.colour_primaries,
        .transfer_characteristics = sps->vui.transfer_characteristics,
        .matrix_coefficients = sps->vui.matrix_coefficients,
        .num_units_in_tick = sps->vui.num_units_in_tick,
        .time_scale = sps->vui.time_scale,
        .pHrdParameters = vksps_vui_header,
        .max_num_reorder_frames = sps->vui.max_num_reorder_frames,
        .max_dec_frame_buffering = sps->vui.max_dec_frame_buffering,
        .flags = (StdVideoH264SpsVuiFlags) {
            .aspect_ratio_info_present_flag = sps->vui.aspect_ratio_info_present_flag,
            .overscan_info_present_flag = sps->vui.overscan_info_present_flag,
            .overscan_appropriate_flag = sps->vui.overscan_appropriate_flag,
            .video_signal_type_present_flag = sps->vui.video_signal_type_present_flag,
            .video_full_range_flag = sps->vui.video_full_range_flag,
            .color_description_present_flag = sps->vui.colour_description_present_flag,
            .chroma_loc_info_present_flag = sps->vui.chroma_loc_info_present_flag,
            .timing_info_present_flag = sps->vui.timing_info_present_flag,
            .fixed_frame_rate_flag = sps->vui.fixed_frame_rate_flag,
            .bitstream_restriction_flag = sps->vui.bitstream_restriction_flag,
            .nal_hrd_parameters_present_flag = sps->vui.nal_hrd_parameters_present_flag,
            .vcl_hrd_parameters_present_flag = sps->vui.vcl_hrd_parameters_present_flag,
        },
    };

    *vksps = (StdVideoH264SequenceParameterSet) {
        .profile_idc = sps->profile_idc,
        .level_idc = sps->level_idc,
        .seq_parameter_set_id = sps->seq_parameter_set_id,
        .chroma_format_idc = sps->chroma_format_idc,
        .bit_depth_luma_minus8 = sps->bit_depth_luma_minus8,
        .bit_depth_chroma_minus8 = sps->bit_depth_chroma_minus8,
        .log2_max_frame_num_minus4 = sps->log2_max_frame_num_minus4,
        .pic_order_cnt_type = sps->pic_order_cnt_type,
        .log2_max_pic_order_cnt_lsb_minus4 = sps->log2_max_pic_order_cnt_lsb_minus4,
        .offset_for_non_ref_pic = sps->offset_for_non_ref_pic,
        .offset_for_top_to_bottom_field = sps->offset_for_top_to_bottom_field,
        .num_ref_frames_in_pic_order_cnt_cycle = sps->num_ref_frames_in_pic_order_cnt_cycle,
        .max_num_ref_frames = sps->max_num_ref_frames,
        .pic_width_in_mbs_minus1 = sps->pic_width_in_mbs_minus1,
        .pic_height_in_map_units_minus1 = sps->pic_height_in_map_units_minus1,
        .frame_crop_left_offset = sps->frame_crop_left_offset,
        .frame_crop_right_offset = sps->frame_crop_right_offset,
        .frame_crop_top_offset = sps->frame_crop_top_offset,
        .frame_crop_bottom_offset = sps->frame_crop_bottom_offset,
        .flags = (StdVideoH264SpsFlags) {
            .constraint_set0_flag = sps->constraint_set0_flag,
            .constraint_set1_flag = sps->constraint_set1_flag,
            .constraint_set2_flag = sps->constraint_set2_flag,
            .constraint_set3_flag = sps->constraint_set3_flag,
            .constraint_set4_flag = sps->constraint_set4_flag,
            .constraint_set5_flag = sps->constraint_set5_flag,
            .direct_8x8_inference_flag = sps->direct_8x8_inference_flag,
            .mb_adaptive_frame_field_flag = sps->mb_adaptive_frame_field_flag,
            .frame_mbs_only_flag = sps->frame_mbs_only_flag,
            .delta_pic_order_always_zero_flag = sps->delta_pic_order_always_zero_flag,
            .separate_colour_plane_flag = sps->separate_colour_plane_flag,
            .gaps_in_frame_num_value_allowed_flag = sps->gaps_in_frame_num_allowed_flag,
            .qpprime_y_zero_transform_bypass_flag = sps->qpprime_y_zero_transform_bypass_flag,
            .frame_cropping_flag = sps->frame_cropping_flag,
            .seq_scaling_matrix_present_flag = sps->seq_scaling_matrix_present_flag,
            .vui_parameters_present_flag = sps->vui_parameters_present_flag,
        },
        .pOffsetForRefFrame = sps->offset_for_ref_frame,
        .pSequenceParameterSetVui = vksps_vui,
    };

    *vkpps_scaling = (StdVideoH264ScalingLists) {
        .scaling_list_present_mask = pps->pic_scaling_matrix_present_flag,
        .use_default_scaling_matrix_mask = 1,
    };

    *vkpps = (StdVideoH264PictureParameterSet) {
        .seq_parameter_set_id = pps->seq_parameter_set_id,
        .pic_parameter_set_id = pps->pic_parameter_set_id,
        .num_ref_idx_l0_default_active_minus1 = pps->num_ref_idx_l0_default_active_minus1,
        .num_ref_idx_l1_default_active_minus1 = pps->num_ref_idx_l1_default_active_minus1,
        .weighted_bipred_idc = pps->weighted_bipred_idc,
        .pic_init_qp_minus26 = pps->pic_init_qp_minus26,
        .pic_init_qs_minus26 = pps->pic_init_qs_minus26,
        .chroma_qp_index_offset = pps->chroma_qp_index_offset,
        .second_chroma_qp_index_offset = pps->second_chroma_qp_index_offset,
        .flags = (StdVideoH264PpsFlags) {
            .transform_8x8_mode_flag = pps->transform_8x8_mode_flag,
            .redundant_pic_cnt_present_flag = pps->redundant_pic_cnt_present_flag,
            .constrained_intra_pred_flag = pps->constrained_intra_pred_flag,
            .deblocking_filter_control_present_flag = pps->deblocking_filter_control_present_flag,
            .weighted_pred_flag = pps->weighted_pred_flag,
            .bottom_field_pic_order_in_frame_present_flag = pps->bottom_field_pic_order_in_frame_present_flag,
            .entropy_coding_mode_flag = pps->entropy_coding_mode_flag,
            .pic_scaling_matrix_present_flag = pps->pic_scaling_matrix_present_flag,
        },
    };

    return 0;
}

static int vk_enc_h264_create_session_params(AVCodecContext *avctx)
{
    VkResult ret;
    VulkanEncodeH264Context *enc = avctx->priv_data;
    FFVulkanEncodeContext *ctx = &enc->common;
    FFVulkanContext *s = &ctx->s;
    FFVulkanFunctions *vk = &ctx->s.vkfn;

    VkVideoEncodeH264SessionParametersAddInfoKHR h264_params_info;
    VkVideoEncodeH264SessionParametersCreateInfoKHR h264_params;
    VkVideoSessionParametersCreateInfoKHR session_params_create;

    h264_params_info = (VkVideoEncodeH264SessionParametersAddInfoKHR) {
        .sType = VK_STRUCTURE_TYPE_VIDEO_ENCODE_H264_SESSION_PARAMETERS_ADD_INFO_KHR,
        .pStdSPSs = &enc->vksps,
        .stdSPSCount = 1,
        .pStdPPSs = &enc->vkpps,
        .stdPPSCount = 1,
    };
    h264_params = (VkVideoEncodeH264SessionParametersCreateInfoKHR) {
        .sType = VK_STRUCTURE_TYPE_VIDEO_ENCODE_H264_SESSION_PARAMETERS_CREATE_INFO_KHR,
        .maxStdSPSCount = 1,
        .maxStdPPSCount = 1,
        .pParametersAddInfo = &h264_params_info,
    };
    session_params_create = (VkVideoSessionParametersCreateInfoKHR) {
        .sType = VK_STRUCTURE_TYPE_VIDEO_SESSION_PARAMETERS_CREATE_INFO_KHR,
        .pNext = &h264_params,
        .videoSession = ctx->common.session,
        .videoSessionParametersTemplate = NULL,
    };

    /* Create session parameters */
    ret = vk->CreateVideoSessionParametersKHR(s->hwctx->act_dev, &session_params_create,
                                              s->hwctx->alloc, &enc->session_params);
    if (ret != VK_SUCCESS) {
        av_log(avctx, AV_LOG_ERROR, "Unable to create Vulkan video session parameters: %s!\n",
               ff_vk_ret2str(ret));
        return AVERROR_EXTERNAL;
    }

    enc->session_params_info = (VkVideoEncodeH264SessionParametersGetInfoKHR) {
        .sType = VK_STRUCTURE_TYPE_VIDEO_ENCODE_H264_SESSION_PARAMETERS_GET_INFO_KHR,
        .writeStdSPS = 1,
        .writeStdPPS = 1,
        .stdSPSId = 0,
        .stdPPSId = 0,
    };

    return 0;
}

static int vk_enc_h264_setup_session_params(AVCodecContext *avctx,
                                            FFHWBaseEncodePicture *pic,
                                            VkVideoEncodeSessionParametersGetInfoKHR *params_info)
{
    VulkanEncodeH264Context *enc = avctx->priv_data;
    params_info->videoSessionParameters = enc->session_params;
    params_info->pNext = &enc->session_params_info;
    return 0;
}

static int vk_enc_h264_init_profile(AVCodecContext *avctx,
                                    VkVideoProfileInfoKHR *profile, void *pnext)
{
    VkResult ret;
    VulkanEncodeH264Context *enc = avctx->priv_data;
    FFVulkanEncodeContext *ctx = &enc->common;
    FFVulkanContext *s = &ctx->s;
    FFVulkanFunctions *vk = &ctx->s.vkfn;
    VkVideoEncodeH264CapabilitiesKHR h264_caps = {
        .sType = VK_STRUCTURE_TYPE_VIDEO_ENCODE_H264_CAPABILITIES_KHR,
    };
    VkVideoEncodeCapabilitiesKHR enc_caps = {
        .sType = VK_STRUCTURE_TYPE_VIDEO_ENCODE_CAPABILITIES_KHR,
        .pNext = &h264_caps,
    };
    VkVideoCapabilitiesKHR caps = {
        .sType = VK_STRUCTURE_TYPE_VIDEO_CAPABILITIES_KHR,
        .pNext = &enc_caps,
    };

    /* In order of preference */
    int last_supported = AV_PROFILE_UNKNOWN;
    static const int known_profiles[] = {
        AV_PROFILE_H264_CONSTRAINED_BASELINE,
        AV_PROFILE_H264_MAIN,
        AV_PROFILE_H264_HIGH,
        AV_PROFILE_H264_HIGH_10,
    };

    enc->profile = (VkVideoEncodeH264ProfileInfoKHR) {
        .sType = VK_STRUCTURE_TYPE_VIDEO_ENCODE_H264_PROFILE_INFO_KHR,
        .pNext = pnext,
        .stdProfileIdc = ff_vk_h264_profile_to_vk(avctx->profile),
    };
    profile->pNext = &enc->profile;

    /* User has explicitly specified a profile. */
    if (avctx->profile != AV_PROFILE_UNKNOWN)
        return 0;

    av_log(avctx, AV_LOG_DEBUG, "Supported profiles:\n");
    for (int i = 0; i < FF_ARRAY_ELEMS(known_profiles); i++) {
        enc->profile.stdProfileIdc = ff_vk_h264_profile_to_vk(known_profiles[i]);
        ret = vk->GetPhysicalDeviceVideoCapabilitiesKHR(s->hwctx->phys_dev,
                                                        profile,
                                                        &caps);
        if (ret == VK_SUCCESS) {
            av_log(avctx, AV_LOG_DEBUG, "    %s\n",
                   avcodec_profile_name(avctx->codec_id, known_profiles[i]));
            last_supported = known_profiles[i];
        }
    }

    if (last_supported == AV_PROFILE_UNKNOWN) {
        av_log(avctx, AV_LOG_ERROR, "No supported profiles for given format\n");
        return AVERROR(ENOTSUP);
    }

    enc->profile.stdProfileIdc = ff_vk_h264_profile_to_vk(last_supported);
    av_log(avctx, AV_LOG_VERBOSE, "Using profile %s\n",
           avcodec_profile_name(avctx->codec_id, last_supported));
    avctx->profile = last_supported;

    return 0;
}

static const FFVulkanCodec enc_cb = {
    .flags = FF_HW_FLAG_B_PICTURES |
             FF_HW_FLAG_B_PICTURE_REFERENCES |
             FF_HW_FLAG_NON_IDR_KEY_PICTURES |
             FF_HW_FLAG_SLICE_CONTROL,
    .picture_priv_data_size = sizeof(VulkanEncodeH264Picture),

    .init_profile = vk_enc_h264_init_profile,
    .setup_rc = vk_enc_h264_setup_rc,
    .setup_session_params = vk_enc_h264_setup_session_params,
    .setup_pic_params = vk_enc_h264_setup_pic_params,
};

static av_cold int vulkan_encode_h264_init(AVCodecContext *avctx)
{
    int err, ref_l0, ref_l1;
    VulkanEncodeH264Context *enc = avctx->priv_data;
    FFVulkanEncodeContext *ctx = &enc->common;
    FFHWBaseEncodeContext *base_ctx = &ctx->base;
    int flags;

    if (avctx->profile == AV_PROFILE_UNKNOWN)
        avctx->profile = enc->common.opts.profile;

    enc->caps = (VkVideoEncodeH264CapabilitiesKHR) {
        .sType = VK_STRUCTURE_TYPE_VIDEO_ENCODE_H264_CAPABILITIES_KHR,
    };

    enc->quality_props = (VkVideoEncodeH264QualityLevelPropertiesKHR) {
        .sType = VK_STRUCTURE_TYPE_VIDEO_ENCODE_H264_QUALITY_LEVEL_PROPERTIES_KHR,
    };

    err = ff_vulkan_encode_init(avctx, &enc->common, &enc->caps,
                                &ff_vk_enc_h264_desc, &enc_cb,
                                &enc->quality_props);
    if (err < 0)
        return err;

    err = vk_enc_h264_init_sequence_params(avctx);
    if (err < 0)
        return err;

    err = vk_enc_h264_create_session_params(avctx);
    if (err < 0)
        return err;

    flags = ctx->codec->flags;
    if (!enc->caps.maxPPictureL0ReferenceCount &&
        !enc->caps.maxBPictureL0ReferenceCount &&
        !enc->caps.maxL1ReferenceCount) {
        /* Intra-only */
        flags |= FF_HW_FLAG_INTRA_ONLY;
        ref_l0 = ref_l1 = 0;
    } else if (!enc->caps.maxPPictureL0ReferenceCount) {
        /* No P-frames? How. */
        base_ctx->p_to_gpb = 1;
        ref_l0 = enc->caps.maxBPictureL0ReferenceCount;
        ref_l1 = enc->caps.maxL1ReferenceCount;
    } else if (!enc->caps.maxBPictureL0ReferenceCount &&
               !enc->caps.maxL1ReferenceCount) {
        /* No B-frames */
        flags &= ~(FF_HW_FLAG_B_PICTURES | FF_HW_FLAG_B_PICTURE_REFERENCES);
        ref_l0 = enc->caps.maxPPictureL0ReferenceCount;
        ref_l1 = 0;
    } else {
        /* P and B frames */
        ref_l0 = FFMIN(enc->caps.maxPPictureL0ReferenceCount,
                       enc->caps.maxBPictureL0ReferenceCount);
        ref_l1 = enc->caps.maxL1ReferenceCount;
    }

    err = ff_hw_base_init_gop_structure(base_ctx, avctx, ref_l0, ref_l1,
                                        flags, 0);
    if (err < 0)
        return err;

    av_log(avctx, AV_LOG_VERBOSE, "H264 encoder capabilities:\n");
    av_log(avctx, AV_LOG_VERBOSE, "    Standard capability flags:\n");
    av_log(avctx, AV_LOG_VERBOSE, "        separate_color_plane: %i\n",
           !!(enc->caps.stdSyntaxFlags & VK_VIDEO_ENCODE_H264_STD_SEPARATE_COLOR_PLANE_FLAG_SET_BIT_KHR));
    av_log(avctx, AV_LOG_VERBOSE, "        qprime_y_zero_transform_bypass: %i\n",
           !!(enc->caps.stdSyntaxFlags & VK_VIDEO_ENCODE_H264_STD_QPPRIME_Y_ZERO_TRANSFORM_BYPASS_FLAG_SET_BIT_KHR));
    av_log(avctx, AV_LOG_VERBOSE, "        scaling_lists: %i\n",
           !!(enc->caps.stdSyntaxFlags & VK_VIDEO_ENCODE_H264_STD_SCALING_MATRIX_PRESENT_FLAG_SET_BIT_KHR));
    av_log(avctx, AV_LOG_VERBOSE, "        chroma_qp_index_offset: %i\n",
           !!(enc->caps.stdSyntaxFlags & VK_VIDEO_ENCODE_H264_STD_CHROMA_QP_INDEX_OFFSET_BIT_KHR));
    av_log(avctx, AV_LOG_VERBOSE, "        second_chroma_qp_index_offset: %i\n",
           !!(enc->caps.stdSyntaxFlags & VK_VIDEO_ENCODE_H264_STD_SECOND_CHROMA_QP_INDEX_OFFSET_BIT_KHR));
    av_log(avctx, AV_LOG_VERBOSE, "        pic_init_qp: %i\n",
           !!(enc->caps.stdSyntaxFlags & VK_VIDEO_ENCODE_H264_STD_PIC_INIT_QP_MINUS26_BIT_KHR));
    av_log(avctx, AV_LOG_VERBOSE, "        weighted:%s%s%s\n",
           enc->caps.stdSyntaxFlags & VK_VIDEO_ENCODE_H264_STD_WEIGHTED_PRED_FLAG_SET_BIT_KHR ?
               " pred" : "",
           enc->caps.stdSyntaxFlags & VK_VIDEO_ENCODE_H264_STD_WEIGHTED_BIPRED_IDC_EXPLICIT_BIT_KHR ?
               " bipred_explicit" : "",
           enc->caps.stdSyntaxFlags & VK_VIDEO_ENCODE_H264_STD_WEIGHTED_BIPRED_IDC_IMPLICIT_BIT_KHR ?
               " bipred_implicit" : "");
    av_log(avctx, AV_LOG_VERBOSE, "        8x8_transforms: %i\n",
           !!(enc->caps.stdSyntaxFlags & VK_VIDEO_ENCODE_H264_STD_TRANSFORM_8X8_MODE_FLAG_SET_BIT_KHR));
    av_log(avctx, AV_LOG_VERBOSE, "        disable_direct_spatial_mv_pred: %i\n",
           !!(enc->caps.stdSyntaxFlags & VK_VIDEO_ENCODE_H264_STD_DIRECT_SPATIAL_MV_PRED_FLAG_UNSET_BIT_KHR));
    av_log(avctx, AV_LOG_VERBOSE, "        coder:%s%s\n",
           enc->caps.stdSyntaxFlags & VK_VIDEO_ENCODE_H264_STD_ENTROPY_CODING_MODE_FLAG_UNSET_BIT_KHR ?
               " cabac" : "",
           enc->caps.stdSyntaxFlags & VK_VIDEO_ENCODE_H264_STD_ENTROPY_CODING_MODE_FLAG_SET_BIT_KHR ?
               " cavlc" : "");
    av_log(avctx, AV_LOG_VERBOSE, "        direct_8x8_inference: %i\n",
           !!(enc->caps.stdSyntaxFlags & VK_VIDEO_ENCODE_H264_STD_DIRECT_8X8_INFERENCE_FLAG_UNSET_BIT_KHR));
    av_log(avctx, AV_LOG_VERBOSE, "        constrained_intra_pred: %i\n",
           !!(enc->caps.stdSyntaxFlags & VK_VIDEO_ENCODE_H264_STD_CONSTRAINED_INTRA_PRED_FLAG_SET_BIT_KHR));
    av_log(avctx, AV_LOG_VERBOSE, "        deblock:%s%s%s\n",
           enc->caps.stdSyntaxFlags & VK_VIDEO_ENCODE_H264_STD_DEBLOCKING_FILTER_DISABLED_BIT_KHR ?
               " filter_disabling" : "",
           enc->caps.stdSyntaxFlags & VK_VIDEO_ENCODE_H264_STD_DEBLOCKING_FILTER_ENABLED_BIT_KHR ?
               " filter_enabling" : "",
           enc->caps.stdSyntaxFlags & VK_VIDEO_ENCODE_H264_STD_DEBLOCKING_FILTER_PARTIAL_BIT_KHR ?
               " filter_partial" : "");

    av_log(avctx, AV_LOG_VERBOSE, "    Capability flags:\n");
    av_log(avctx, AV_LOG_VERBOSE, "        hdr_compliance: %i\n",
           !!(enc->caps.flags & VK_VIDEO_ENCODE_H264_CAPABILITY_HRD_COMPLIANCE_BIT_KHR));
    av_log(avctx, AV_LOG_VERBOSE, "        pred_weight_table_generated: %i\n",
           !!(enc->caps.flags & VK_VIDEO_ENCODE_H264_CAPABILITY_PREDICTION_WEIGHT_TABLE_GENERATED_BIT_KHR));
    av_log(avctx, AV_LOG_VERBOSE, "        row_unaligned_slice: %i\n",
           !!(enc->caps.flags & VK_VIDEO_ENCODE_H264_CAPABILITY_ROW_UNALIGNED_SLICE_BIT_KHR));
    av_log(avctx, AV_LOG_VERBOSE, "        different_slice_type: %i\n",
           !!(enc->caps.flags & VK_VIDEO_ENCODE_H264_CAPABILITY_DIFFERENT_SLICE_TYPE_BIT_KHR));
    av_log(avctx, AV_LOG_VERBOSE, "        b_frame_in_l0_list: %i\n",
           !!(enc->caps.flags & VK_VIDEO_ENCODE_H264_CAPABILITY_B_FRAME_IN_L0_LIST_BIT_KHR));
    av_log(avctx, AV_LOG_VERBOSE, "        b_frame_in_l1_list: %i\n",
           !!(enc->caps.flags & VK_VIDEO_ENCODE_H264_CAPABILITY_B_FRAME_IN_L1_LIST_BIT_KHR));
    av_log(avctx, AV_LOG_VERBOSE, "        per_pict_type_min_max_qp: %i\n",
           !!(enc->caps.flags & VK_VIDEO_ENCODE_H264_CAPABILITY_PER_PICTURE_TYPE_MIN_MAX_QP_BIT_KHR));
    av_log(avctx, AV_LOG_VERBOSE, "        per_slice_constant_qp: %i\n",
           !!(enc->caps.flags & VK_VIDEO_ENCODE_H264_CAPABILITY_PER_SLICE_CONSTANT_QP_BIT_KHR));
    av_log(avctx, AV_LOG_VERBOSE, "        generate_prefix_nalu: %i\n",
           !!(enc->caps.flags & VK_VIDEO_ENCODE_H264_CAPABILITY_GENERATE_PREFIX_NALU_BIT_KHR));

    av_log(avctx, AV_LOG_VERBOSE, "    Capabilities:\n");
    av_log(avctx, AV_LOG_VERBOSE, "        maxLevelIdc: %i\n",
           enc->caps.maxLevelIdc);
    av_log(avctx, AV_LOG_VERBOSE, "        maxSliceCount: %i\n",
           enc->caps.maxSliceCount);
    av_log(avctx, AV_LOG_VERBOSE, "    max(P/B)PictureL0ReferenceCount: %i P's; %i B's\n",
           enc->caps.maxPPictureL0ReferenceCount,
           enc->caps.maxBPictureL0ReferenceCount);
    av_log(avctx, AV_LOG_VERBOSE, "    maxL1ReferenceCount: %i\n",
           enc->caps.maxL1ReferenceCount);
    av_log(avctx, AV_LOG_VERBOSE, "    maxTemporalLayerCount: %i\n",
           enc->caps.maxTemporalLayerCount);
    av_log(avctx, AV_LOG_VERBOSE, "    expectDyadicTemporalLayerPattern: %i\n",
           enc->caps.expectDyadicTemporalLayerPattern);
    av_log(avctx, AV_LOG_VERBOSE, "    min/max Qp: [%i, %i]\n",
           enc->caps.maxQp, enc->caps.minQp);
    av_log(avctx, AV_LOG_VERBOSE, "    prefersGopRemainingFrames: %i\n",
           enc->caps.prefersGopRemainingFrames);
    av_log(avctx, AV_LOG_VERBOSE, "    requiresGopRemainingFrames: %i\n",
           enc->caps.requiresGopRemainingFrames);

    return 0;
}

static av_cold int vulkan_encode_h264_close(AVCodecContext *avctx)
{
    VulkanEncodeH264Context *enc = avctx->priv_data;
    FFVulkanEncodeContext *ctx = &enc->common;
    FFVulkanContext *s = &ctx->s;
    FFVulkanFunctions *vk = &ctx->s.vkfn;

    if (enc->session_params)
        vk->DestroyVideoSessionParametersKHR(s->hwctx->act_dev,
                                             enc->session_params,
                                             s->hwctx->alloc);

    ff_vulkan_encode_uninit(&enc->common);
    return 0;
}

static void vulkan_encode_h264_flush(AVCodecContext *avctx)
{

}

#define OFFSET(x) offsetof(VulkanEncodeH264Context, x)
#define FLAGS (AV_OPT_FLAG_VIDEO_PARAM | AV_OPT_FLAG_ENCODING_PARAM)
static const AVOption vulkan_encode_h264_options[] = {
    HW_BASE_ENCODE_COMMON_OPTIONS,
    VULKAN_ENCODE_COMMON_OPTIONS,

    { "profile", "Set profile (profile_idc and constraint_set*_flag)",
      OFFSET(common.opts.profile), AV_OPT_TYPE_INT,
      { .i64 = AV_PROFILE_UNKNOWN }, AV_PROFILE_UNKNOWN, 0xffff, FLAGS, .unit = "profile" },

#define PROFILE(name, value)  name, NULL, 0, AV_OPT_TYPE_CONST, \
      { .i64 = value }, 0, 0, FLAGS, .unit = "profile"
    { PROFILE("constrained_baseline", AV_PROFILE_H264_CONSTRAINED_BASELINE) },
    { PROFILE("main",                 AV_PROFILE_H264_MAIN) },
    { PROFILE("high",                 AV_PROFILE_H264_HIGH) },
    { PROFILE("high444p",             AV_PROFILE_H264_HIGH_10) },
#undef PROFILE

    { "level", "Set level (level_idc)",
      OFFSET(common.opts.level), AV_OPT_TYPE_INT,
      { .i64 = AV_LEVEL_UNKNOWN }, AV_LEVEL_UNKNOWN, 0xff, FLAGS, .unit = "level" },

#define LEVEL(name, value) name, NULL, 0, AV_OPT_TYPE_CONST, \
      { .i64 = value }, 0, 0, FLAGS, .unit = "level"
    { LEVEL("1",   10) },
    { LEVEL("1.1", 11) },
    { LEVEL("1.2", 12) },
    { LEVEL("1.3", 13) },
    { LEVEL("2",   20) },
    { LEVEL("2.1", 21) },
    { LEVEL("2.2", 22) },
    { LEVEL("3",   30) },
    { LEVEL("3.1", 31) },
    { LEVEL("3.2", 32) },
    { LEVEL("4",   40) },
    { LEVEL("4.1", 41) },
    { LEVEL("4.2", 42) },
    { LEVEL("5",   50) },
    { LEVEL("5.1", 51) },
    { LEVEL("5.2", 52) },
    { LEVEL("6",   60) },
    { LEVEL("6.1", 61) },
    { LEVEL("6.2", 62) },
#undef LEVEL

    { "coder", "Entropy coder type", OFFSET(unit_opts.cabac), AV_OPT_TYPE_INT, { .i64 = 1 }, 0, 1, FLAGS, "coder" },
        { "cabac", NULL, 0, AV_OPT_TYPE_CONST, { .i64 = 1 }, INT_MIN, INT_MAX, FLAGS, "coder" },
        { "vlc",   NULL, 0, AV_OPT_TYPE_CONST, { .i64 = 0 }, INT_MIN, INT_MAX, FLAGS, "coder" },

    { "units", "Set units to include", OFFSET(unit_elems), AV_OPT_TYPE_FLAGS, { .i64 = UNIT_IDENTIFIER | UNIT_AUD | UNIT_RECOVERY }, 0, INT_MAX, FLAGS, "units" },
        { "identifier", "Include encoder version identifier", 0, AV_OPT_TYPE_CONST, { .i64 = UNIT_IDENTIFIER }, INT_MIN, INT_MAX, FLAGS, "units" },
        { "aud",        "Include AUD units", 0, AV_OPT_TYPE_CONST, { .i64 = UNIT_AUD }, INT_MIN, INT_MAX, FLAGS, "units" },
        { "timing",     "Include timing parameters (buffering_period and pic_timing)", 0, AV_OPT_TYPE_CONST, { .i64 = UNIT_TIMING }, INT_MIN, INT_MAX, FLAGS, "units" },
        { "recovery",   "Include recovery points where appropriate", 0, AV_OPT_TYPE_CONST, { .i64 = UNIT_RECOVERY }, INT_MIN, INT_MAX, FLAGS, "units" },

    { NULL },
};

static const FFCodecDefault vulkan_encode_h264_defaults[] = {
    { "b",              "0"   },
    { "bf",             "2"   },
    { "g",              "120" },
    { "i_qfactor",      "1"   },
    { "i_qoffset",      "0"   },
    { "b_qfactor",      "1"   },
    { "b_qoffset",      "0"   },
    { "qmin",           "-1"  },
    { "qmax",           "-1"  },
    { NULL },
};

static const AVClass vulkan_encode_h264_class = {
    .class_name = "h264_vulkan",
    .item_name  = av_default_item_name,
    .option     = vulkan_encode_h264_options,
    .version    = LIBAVUTIL_VERSION_INT,
};

const FFCodec ff_h264_vulkan_encoder = {
    .p.name         = "h264_vulkan",
    CODEC_LONG_NAME("H.264/AVC (Vulkan)"),
    .p.type         = AVMEDIA_TYPE_VIDEO,
    .p.id           = AV_CODEC_ID_H264,
    .priv_data_size = sizeof(VulkanEncodeH264Context),
    .init           = &vulkan_encode_h264_init,
    FF_CODEC_RECEIVE_PACKET_CB(&ff_vulkan_encode_receive_packet),
    .flush          = &vulkan_encode_h264_flush,
    .close          = &vulkan_encode_h264_close,
    .p.priv_class   = &vulkan_encode_h264_class,
    .p.capabilities = AV_CODEC_CAP_DELAY |
                      AV_CODEC_CAP_HARDWARE |
                      AV_CODEC_CAP_DR1 |
                      AV_CODEC_CAP_ENCODER_FLUSH |
                      AV_CODEC_CAP_ENCODER_REORDERED_OPAQUE,
    .caps_internal  = FF_CODEC_CAP_INIT_CLEANUP,
    .defaults       = vulkan_encode_h264_defaults,
    .p.pix_fmts = (const enum AVPixelFormat[]) {
        AV_PIX_FMT_VULKAN,
        AV_PIX_FMT_NONE,
    },
    .hw_configs     = ff_vulkan_encode_hw_configs,
    .p.wrapper_name = "vulkan",
};
