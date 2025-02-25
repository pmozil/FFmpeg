GEN_CLEANSUFFIXES = *.o *.c *.d

clean::
	$(RM) $(GEN_CLEANSUFFIXES:%=libavcodec/vulkan/%)

OBJS-$(CONFIG_FFV1_VULKAN_ENCODER)  +=  vulkan/common.o \
					vulkan/rangecoder.o vulkan/ffv1_vlc.o \
					vulkan/ffv1_common.o vulkan/ffv1_reset.o \
					vulkan/ffv1_enc_common.o \
					vulkan/ffv1_enc_rct.o vulkan/ffv1_enc_setup.o \
					vulkan/ffv1_enc_vlc.o vulkan/ffv1_enc_ac.o \
					vulkan/ffv1_enc.o vulkan/ffv1_enc_rgb.o

OBJS-$(CONFIG_FFV1_VULKAN_HWACCEL)  +=  vulkan/common.o \
					vulkan/rangecoder.o vulkan/ffv1_vlc.o \
					vulkan/ffv1_common.o vulkan/ffv1_reset.o \
					vulkan/ffv1_dec_setup.o vulkan/ffv1_dec.o

OBJS-$(CONFIG_FFV1_VULKAN_HWACCEL)  +=  vulkan/common.o \
					vulkan/rangecoder.o vulkan/ffv1_vlc.o \
					vulkan/ffv1_common.o vulkan/ffv1_reset.o \
					vulkan/ffv1_dec_setup.o vulkan/ffv1_dec.o

OBJS-$(CONFIG_DIRAC_VULKAN_HWACCEL) += vulkan/common.o \
					vulkan/dirac/dirac_cpy_to_image_8bit.o \
					vulkan/dirac/dirac_cpy_to_image_10bit.o \
					vulkan/dirac/dirac_cpy_to_image_12bit.o \
					vulkan/dirac/dirac_daub97_vert.o \
					vulkan/dirac/dirac_daub97_horiz.o \
					vulkan/dirac/dirac_dd97_vert.o \
					vulkan/dirac/dirac_dd97_horiz.o \
					vulkan/dirac/dirac_fidelity_vert.o \
					vulkan/dirac/dirac_fidelity_horiz.o \
					vulkan/dirac/dirac_dd137_vert.o \
					vulkan/dirac/dirac_dd137_horiz.o \
					vulkan/dirac/dirac_legall_vert.o \
					vulkan/dirac/dirac_legall_horiz.o \
					vulkan/dirac/dirac_haar0_horiz.o \
					vulkan/dirac/dirac_haar1_horiz.o \
					vulkan/dirac/dirac_haar_vert.o \
					vulkan/dirac/dirac_dequant.o \
					vulkan/dirac/dirac_structs.o

VULKAN = $(subst $(SRC_PATH)/,,$(wildcard $(SRC_PATH)/libavcodec/vulkan/*.comp))
.SECONDARY: $(VULKAN:.comp=.c)
libavcodec/vulkan/%.c: TAG = VULKAN
libavcodec/vulkan/%.c: $(SRC_PATH)/libavcodec/vulkan/%.comp
	$(M)$(SRC_PATH)/tools/source2c $< $@
