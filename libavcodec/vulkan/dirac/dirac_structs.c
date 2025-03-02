// Generated from libavcodec/vulkan/dirac/dirac_structs.comp
const char *ff_source_dirac_structs_comp =
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
"struct Slice {\n"
"    int left;\n"
"    int top;\n"
"    int tot_h;\n"
"    int tot_v;\n"
"    int tot;\n"
"    int offs;\n"
"    int pad0;\n"
"    int pad1;\n"
"};\n"
"\n"
"struct SubbandOffset {\n"
"    int base_off;\n"
"    int stride;\n"
"    int pad0;\n"
"    int pad1;\n"
"};\n"
"\n"
"struct LUTState {\n"
"    int16_t   val0;\n"
"    int16_t   val1;\n"
"    int16_t   val2;\n"
"    int16_t   val3;\n"
"    int16_t   val4;\n"
"    uint8_t   val0_bits;\n"
"    int8_t    sign;\n"
"    int8_t    num;\n"
"    uint8_t   val;\n"
"    uint16_t  state;\n"
"};\n"
"\n"
"#define DWT_LEVELS 5\n"
;
