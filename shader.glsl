#version 460
#define IS_WITHIN(v1, v2) ((v1.x < v2.x) && (v1.y < v2.y))

#extension GL_EXT_buffer_reference : require
#extension GL_EXT_buffer_reference2 : require
layout (local_size_x = 30, local_size_y = 32, local_size_z = 1) in;

layout (set = 0, binding = 0, rgba8) uniform writeonly image2D out_img[3];

struct SliceCoeffs {
    int left;
    int top;
    int tot_h;
    int tot_v;
    int tot;
};
struct Slice {
    int idx;
    int offs[15];
    SliceCoeffs coeffs[15];
};
layout (set = 1, binding = 0) readonly buffer quant_in_buf {
    int inBuffer[];
};
layout (set = 1, binding = 1) readonly buffer quant_vals_buf {
    int quantMatrix[];
};
layout (set = 1, binding = 2) readonly buffer slices_buf {
    Slice slices[];
};

layout(push_constant, std430) uniform pushConstants {
    int wavelet_depth;
    int slices_num;
};

#define DWT_LEVELS 5

void dequant(int plane, int idx, ivec2 pos, int qf, int qs) {
    int val = inBuffer[idx];
    if (val < 0) {
        val = -(((-val)*qf + qs) >> 2);
    } else if (val > 0) {
        val = ((val*qf + qs) >> 2);
    }
    imageStore(out_img[plane], pos, vec4(1.0));
}


void proc_slice(int slice_idx) {
    const int plane = int(gl_GlobalInvocationID.x) % 3;
    const int off_x = int(gl_NumWorkGroups.x) / 3;
    const int level = int(gl_GlobalInvocationID.y) % wavelet_depth;
    const int off_y = int(gl_NumWorkGroups.y) / wavelet_depth;
    const Slice s = slices[slice_idx];
    const int base_idx = slice_idx * DWT_LEVELS * 8;
        const SliceCoeffs c = s.coeffs[DWT_LEVELS * plane + level];
        int offs = s.offs[DWT_LEVELS * plane + level];
            int orient = int(bool(level));
        for(; orient < 4; orient++) {
            int y = int(gl_GlobalInvocationID.y);
            int qf = quantMatrix[base_idx + level * 8 + orient];
            int qs = quantMatrix[base_idx + level * 8 + 4 + orient];
            for(; y < c.tot_v; y += int(gl_NumWorkGroups.y)) {
                int x = int(gl_GlobalInvocationID.x) / 3;
                for(; x < c.tot_h; x += off_x) {
                    offs += off_x;
                    dequant(plane, offs, ivec2(c.left + x, c.top + y), qf, qs);
            }
        }
    }
}

void main()
{
    int idx = int(gl_GlobalInvocationID.z);
    for (; idx < slices_num; idx += int(gl_NumWorkGroups.z)) {
        proc_slice(idx);
    }
}
