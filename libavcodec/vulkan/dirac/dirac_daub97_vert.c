// Generated from libavcodec/vulkan/dirac/dirac_daub97_vert.comp
const char *ff_source_dirac_daub97_vert_comp =
"layout(push_constant, std430) uniform pushConstants {\n"
"    ivec2 plane_sizes[3];\n"
"    int plane_offs[3];\n"
"    int plane_strides[3];\n"
"    int dw[3];\n"
"    int wavelet_depth;\n"
"};\n"
"\n"
"int getIdx(int plane, int x, int y) {\n"
"  return plane_offs[plane] + plane_strides[plane] * y + x;\n"
"}\n"
"\n"
"void main() {\n"
"    int off_x = int(gl_WorkGroupSize.x * gl_NumWorkGroups.x);\n"
"    int pic_z = int(gl_GlobalInvocationID.z);\n"
"\n"
"    uint h = int(plane_sizes[pic_z].y);\n"
"    uint w = int(plane_sizes[pic_z].x);\n"
"    int x = int(gl_GlobalInvocationID.x);\n"
"\n"
"    for (; x < w; x += off_x) {\n"
"        for (int y = 0; y < h; y += 2) {\n"
"            int32_t v0 = inBuf[getIdx(pic_z, x, int(clamp(y - 1, 0, h)))];\n"
"            int32_t v1 = inBuf[getIdx(pic_z, x, y + 1)];\n"
"            inBuf[getIdx(pic_z, x, y)] -= (1817 * (v0 + v1 + 2048)) >> 12;\n"
"        }\n"
"        for (int y = 0; y < h; y += 2) {\n"
"            int32_t v0 = inBuf[getIdx(pic_z, x, y)];\n"
"            int32_t v1 = inBuf[getIdx(pic_z, x, int(clamp(y + 2, 0, h - 2)))];\n"
"            inBuf[getIdx(pic_z, x, y + 1)] -= (3616 * (v0 + v1 + 2048)) >> 12;\n"
"        }\n"
"        for (int y = 0; y < h; y += 2) {\n"
"            int32_t v0 = inBuf[getIdx(pic_z, x, int(clamp(y - 1, 0, h)))];\n"
"            int32_t v1 = inBuf[getIdx(pic_z, x, y + 1)];\n"
"            int32_t v2 = inBuf[getIdx(pic_z, x, y)];\n"
"            outBuf[getIdx(pic_z, x, y)] = v2 + (217 * (v0 + v1 + 2048)) >> 12;\n"
"        }\n"
"        for (int y = 0; y < h; y += 2) {\n"
"            int32_t v0 = inBuf[getIdx(pic_z, x, y)];\n"
"            int32_t v1 = inBuf[getIdx(pic_z, x, int(clamp(y + 2, 0, h - 2)))];\n"
"            int32_t v2 = inBuf[getIdx(pic_z, x, y + 1)];\n"
"            outBuf[getIdx(pic_z, x, y + 1)] = v2 + (6497 * (v0 + v1 + 2048)) >> 12;\n"
"        }\n"
"    }\n"
"}\n"
;
