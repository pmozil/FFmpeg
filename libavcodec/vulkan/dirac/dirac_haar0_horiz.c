// Generated from libavcodec/vulkan/dirac/dirac_haar0_horiz.comp
const char *ff_source_dirac_haar0_horiz_comp =
"/*\n"
" * This file is part of FFmpeg.\n"
" *\n"
" * FFmpeg is free software; you can redistribute it and/or\n"
" * modify it under the terms of the GNU Lesser General Public\n"
" * License as published by the Free Software Foundation; either\n"
" * version 2.1 of the License, or (at your option) any later version.\n"
" *\n"
" * FFmpeg is distributed in the hope that it will be useful,\n"
" * but WITHOUT ANY WARRANTY; without even the implied warranty of\n"
" * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU\n"
" * Lesser General Public License for more details.\n"
" *\n"
" * You should have received a copy of the GNU Lesser General Public\n"
" * License along with FFmpeg; if not, write to the Free Software\n"
" * Foundation, Inc., 51 Franklin Street, Fifth Floor, Boston, MA 02110-1301 USA\n"
" */\n"
"\n"
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
"void idwt_horiz(int plane, int x, int y) {\n"
"    int offs0 = plane_offs[plane] + plane_strides[plane] * y + x;\n"
"    int offs1 = offs0 + plane_sizes[plane].x / 2;\n"
"    int outIdx = plane_offs[plane] + plane_strides[plane] * y + x * 2;\n"
"    int32_t val_orig0 = inBuf[offs0];\n"
"    int32_t val_orig1 = inBuf[offs1];\n"
"    int32_t val_new0 = val_orig0 - ((val_orig1 + 1) >> 1);\n"
"    int32_t val_new1 = val_orig1 + val_new0;\n"
"    outBuf[outIdx] = val_new0;\n"
"    outBuf[outIdx + 1] = val_new1;\n"
"}\n"
"\n"
"void main() {\n"
"    int off_y = int(gl_WorkGroupSize.y * gl_NumWorkGroups.y);\n"
"    int off_x = int(gl_WorkGroupSize.x * gl_NumWorkGroups.x);\n"
"    int pic_z = int(gl_GlobalInvocationID.z);\n"
"\n"
"    uint h = int(plane_sizes[pic_z].y);\n"
"    uint w = int(plane_sizes[pic_z].x);\n"
"\n"
"    int y = int(gl_GlobalInvocationID.y);\n"
"    for (; y < h; y += off_y) {\n"
"        int x = int(gl_GlobalInvocationID.x);\n"
"        for (; 2 * x < w; x += off_x) {\n"
"            idwt_horiz(pic_z, x, y);\n"
"        }\n"
"    }\n"
"}\n"
;
