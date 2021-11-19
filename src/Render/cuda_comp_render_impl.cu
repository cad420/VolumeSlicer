//
// Created by wyz on 2021/7/21.
//

#include <iostream>

#include "Algorithm/helper_math.h"
#include "Common/cuda_box.hpp"
#include "Common/cuda_utils.hpp"
#include "Render/cuda_comp_render_impl.cuh"

using namespace CUDARenderer;

namespace
{

__constant__ CUDACompRenderParameter cudaCompRenderParameter;
__constant__ CompVolumeParameter compVolumeParameter;
__constant__ LightParameter lightParameter;
__constant__ MPIRenderParameter mpiRenderParameter;
__constant__ uint4 *mappingTable;
__constant__ uint lodMappingTableOffset[10];
__constant__ cudaTextureObject_t cacheVolumes[10];
__constant__ cudaTextureObject_t transferFunc;
__constant__ cudaTextureObject_t preIntTransferFunc;
__constant__ uint *image;
__constant__ uint *missedBlocks;
__constant__ uint *cdfMap[10];

uint *d_image = nullptr;
int image_w = 0, image_h = 0;
uint4 *d_mappingTable = nullptr;
size_t mappingTableSize = 0;
cudaArray *tf = nullptr;
cudaArray *preInt_tf = nullptr;
uint *d_missedBlocks = nullptr;
uint *d_cdfMap[10] = {nullptr};

__device__ uint rgbaFloatToUInt(float4 rgba)
{
    rgba.x = __saturatef(rgba.x); // clamp to [0.0, 1.0]
    rgba.y = __saturatef(rgba.y);
    rgba.z = __saturatef(rgba.z);
    rgba.w = __saturatef(rgba.w);
    return (uint(rgba.w * 255) << 24) | (uint(rgba.z * 255) << 16) | (uint(rgba.y * 255) << 8) | uint(rgba.x * 255);
}

__device__ int evaluateLod(float distance)
{
    if (distance < 0.3f)
    {
        return 0;
    }
    else if (distance < 0.5f)
    {
        return 1;
    }
    else if (distance < 1.2f)
    {
        return 2;
    }
    else if (distance < 1.6f)
    {
        return 3;
    }
    else if (distance < 3.2f)
    {
        return 4;
    }
    else if (distance < 6.4f)
    {
        return 5;
    }
    else
    {
        return 6;
    }
}

/*
 * using raycast to calculate intersect lod blocks
 * not fast enough and cost a lot bit because of call atomicExch
 */
__global__ void CUDACalcBlockKernel()
{
    int image_x = blockIdx.x * blockDim.x + threadIdx.x;
    int image_y = blockIdx.y * blockDim.y + threadIdx.y;
    if (image_x >= cudaCompRenderParameter.w || image_y >= cudaCompRenderParameter.h)
        return;

    float scale = 2.f * tanf(radians(cudaCompRenderParameter.fov / 2)) / cudaCompRenderParameter.h;
    float x_offset = (image_x - cudaCompRenderParameter.w / 2) * scale * cudaCompRenderParameter.w /
                     cudaCompRenderParameter.h; // ratio
    float y_offset = (image_y - cudaCompRenderParameter.h / 2) * scale;
    if (cudaCompRenderParameter.mpi_render) // mpi_node_x_offset is measured in pixel
    {
        x_offset +=
            mpiRenderParameter.mpi_node_x_offset * scale * cudaCompRenderParameter.w / cudaCompRenderParameter.h;
        y_offset += mpiRenderParameter.mpi_node_y_offset * scale;
    }

    float3 pixel_view_pos = cudaCompRenderParameter.view_pos + cudaCompRenderParameter.view_direction +
                            x_offset * cudaCompRenderParameter.right - y_offset * cudaCompRenderParameter.up;
    float3 ray_direction = normalize(pixel_view_pos - cudaCompRenderParameter.view_pos);

    float3 start_pos = cudaCompRenderParameter.view_pos;
    float3 ray_pos = start_pos;
    int last_lod = 0;
    int lod_t = 1;
    float cur_step = cudaCompRenderParameter.step * 8;
    int3 block_dim = compVolumeParameter.block_dim;
    int no_padding_block_length = compVolumeParameter.no_padding_block_length;
    int steps = 0;
    int nsteps = cudaCompRenderParameter.steps;
    while (steps++ < nsteps / 4)
    {
        if (ray_pos.x < 0.f || ray_pos.x > compVolumeParameter.volume_board.x || ray_pos.y < 0.f ||
            ray_pos.y > compVolumeParameter.volume_board.y || ray_pos.z < 0.f ||
            ray_pos.z > compVolumeParameter.volume_board.z)
        {
            break;
        }
        int cur_lod = evaluateLod(length(ray_pos - start_pos));
        if (cur_lod > last_lod)
        {
            cur_step *= 2;
            lod_t *= 2;
            block_dim = (block_dim + 1) / 2;
            no_padding_block_length *= 2;
            last_lod = cur_lod;
        }
        if (cur_lod > 6)
            break;
        int3 block_idx = make_int3(ray_pos / cudaCompRenderParameter.space / no_padding_block_length);
        size_t flat_block_idx = block_idx.z * block_dim.x * block_dim.y + block_idx.y * block_dim.x + block_idx.x +
                                lodMappingTableOffset[cur_lod];

        if (missedBlocks[flat_block_idx] == 0)
        {
            atomicExch(&missedBlocks[flat_block_idx], 1);
        }

        ray_pos += ray_direction * cur_step;
    }
}

__device__ float3 GetCDFEmptySkipPos(int lod, int lod_t, const float3 &sample_pos, const float3 &sample_dir)
{
    int lod_no_padding_block_length = compVolumeParameter.no_padding_block_length * lod_t;
    int3 lod_block_dim = (compVolumeParameter.block_dim + lod_t - 1) / lod_t;
    int3 virtual_block_idx = make_int3(sample_pos / lod_no_padding_block_length);
    int flat_block_idx = virtual_block_idx.z * lod_block_dim.x * lod_block_dim.y +
                         virtual_block_idx.y * lod_block_dim.x + virtual_block_idx.x;
    float3 offset_in_no_padding_block =
        (sample_pos - make_float3(virtual_block_idx * lod_no_padding_block_length)) / lod_t;
    int3 cdf_block_idx = make_int3(offset_in_no_padding_block / compVolumeParameter.cdf_block_length);
    int flat_cdf_block_idx =
        (cdf_block_idx.z * compVolumeParameter.cdf_dim_len + cdf_block_idx.y) * compVolumeParameter.cdf_dim_len +
        cdf_block_idx.x;
    uint32_t cdf = cdfMap[lod][flat_block_idx * compVolumeParameter.cdf_block_num + flat_cdf_block_idx];
    if (!cdf)
    {
        return sample_pos;
    }
    float3 box_min_p = make_float3(virtual_block_idx * lod_no_padding_block_length +
                                   cdf_block_idx * compVolumeParameter.cdf_block_length * lod_t);
    auto box = CUDABox(box_min_p, box_min_p + compVolumeParameter.cdf_block_length * lod_t);
    auto t = IntersectWithAABB(box, CUDASimpleRay(sample_pos, sample_dir));
    return sample_pos + t.y * sample_dir;
}

/*
 * samplePos is measured in voxel
 */
__device__ int VirtualSample(int lod, int lod_t, const float3 &samplePos, float &scalar)
{
    int lod_no_padding_block_length = compVolumeParameter.no_padding_block_length * lod_t;
    int3 lod_block_dim = (compVolumeParameter.block_dim + lod_t - 1) / lod_t;
    int3 virtual_block_idx = make_int3(samplePos / lod_no_padding_block_length);

    size_t flat_block_idx = virtual_block_idx.z * lod_block_dim.x * lod_block_dim.y +
                            virtual_block_idx.y * lod_block_dim.x + virtual_block_idx.x + lodMappingTableOffset[lod];

    uint4 physical_block_idx = mappingTable[flat_block_idx];
    uint physical_block_flag = (physical_block_idx.w >> 16) & (0x0000ffff);

    if (physical_block_flag != 1)
    {
        return 0;
    }
    float3 offset_in_no_padding_block =
        (samplePos - make_float3(virtual_block_idx * lod_no_padding_block_length)) / lod_t;
    float3 physical_sample_pos = make_float3(physical_block_idx.x, physical_block_idx.y, physical_block_idx.z) *
                                     compVolumeParameter.block_length +
                                 offset_in_no_padding_block + make_float3(compVolumeParameter.padding);
    uint tex_id = physical_block_idx.w & (0x0000ffff);
    physical_sample_pos /= make_float3(compVolumeParameter.texture_shape);
    scalar = tex3D<float>(cacheVolumes[tex_id], physical_sample_pos.x, physical_sample_pos.y, physical_sample_pos.z);

    return 1;
}

__device__ int IntPow(int x, int y)
{
    int ans = 1;
    for (int i = 0; i < y; i++)
        ans *= x;
    return ans;
}

/*
 * samplePos is measured in voxel
 */
__device__ float3 PhongShading(int lod, int lod_t, const float3 &samplePos, float3 diffuseColor,
                               const float3 &view_direction)
{
    float3 N;
#undef CUBIC
#ifndef CUBIC
    float x1, x2;
    VirtualSample(lod, lod_t, samplePos + make_float3(lod_t, 0, 0), x1);
    VirtualSample(lod, lod_t, samplePos + make_float3(-lod_t, 0, 0), x2);
    N.x = x1 - x2;
    VirtualSample(lod, lod_t, samplePos + make_float3(0, lod_t, 0), x1);
    VirtualSample(lod, lod_t, samplePos + make_float3(0, -lod_t, 0), x2);
    N.y = x1 - x2;
    VirtualSample(lod, lod_t, samplePos + make_float3(0, 0, lod_t), x1);
    VirtualSample(lod, lod_t, samplePos + make_float3(0, 0, -lod_t), x2);
    N.z = x1 - x2;
#else
    float value[27];
    float t1[9];
    float t2[3];
    for (int k = -1; k < 2; k++)
    {
        for (int j = -1; j < 2; j++)
        {
            for (int i = -1; i < 2; i++)
            {
                float scalar = 0.f;
                float3 offset = make_float3(i, j, k) * lod_t;
                VirtualSample(lod, lod_t, samplePos + offset, scalar);
                value[(k + 1) * 9 + (j + 1) * 3 + i + 1] = scalar;
            }
        }
    }
    int x, y, z;
    // for x-direction
    for (z = 0; z < 3; z++)
    {
        for (y = 0; y < 3; y++)
        {
            t1[z * 3 + y] = (value[18 + y * 3 + z] - value[y * 3 + z]) / 2;
        }
    }
    for (z = 0; z < 3; z++)
        t2[z] = (t1[z * 3 + 0] + 4 * t1[z * 3 + 1] + t1[z * 3 + 2]) / 6;
    N.x = (t2[0] + t2[1] * 4 + t2[2]) / 6;

    // for y-direction
    for (z = 0; z < 3; z++)
    {
        for (x = 0; x < 3; x++)
        {
            t1[z * 3 + x] = (value[x * 9 + 6 + z] - value[x * 9 + z]) / 2;
        }
    }
    for (z = 0; z < 3; z++)
        t2[z] = (t1[z * 3 + 0] + 4 * t1[z * 3 + 1] + t1[z * 3 + 2]) / 6;
    N.y = (t2[0] + t2[1] * 4 + t2[2]) / 6;

    // for z-direction
    for (y = 0; y < 3; y++)
    {
        for (x = 0; x < 3; x++)
        {
            t1[y * 3 + x] = (value[x * 9 + y * 3 + 2] - value[x * 9 + y * 3]) / 2;
        }
    }
    for (y = 0; y < 3; y++)
        t2[y] = (t1[y * 3 + 0] + 4 * t1[y * 3 + 1] + t1[y * 3 + 2]) / 6;
    N.z = (t2[0] + t2[1] * 4 + t2[2]) / 6;
#endif
    N = normalize(-N);

    float3 L = {-view_direction.x, -view_direction.y, -view_direction.z};
    float3 R = L;
    if (dot(N, L) < 0.f)
        N = -N;

    float3 ambient =
//        lightParameter.ka
        0.05* diffuseColor;
    float3 specular =
//        lightParameter.ks
        1.f* pow(max(dot(N, (L + R) / 2.f), 0.f), lightParameter.shininess) * make_float3(1.f);
    float3 diffuse =
//        lightParameter.kd
        1.f* max(dot(N, L), 0.f) * diffuseColor;
    return ambient + specular + diffuse;
}

__global__ void CUDARenderKernel()
{

    int image_x = blockIdx.x * blockDim.x + threadIdx.x;
    int image_y = blockIdx.y * blockDim.y + threadIdx.y;
    if (image_x >= cudaCompRenderParameter.w || image_y >= cudaCompRenderParameter.h)
        return;
    uint64_t image_idx = (uint64_t)image_y * cudaCompRenderParameter.w + image_x;

    float scale = 2.f * tanf(radians(cudaCompRenderParameter.fov / 2)) / cudaCompRenderParameter.h;
    float x_offset = (image_x - cudaCompRenderParameter.w / 2) * scale * cudaCompRenderParameter.w /
                     cudaCompRenderParameter.h; // ratio
    float y_offset = (image_y - cudaCompRenderParameter.h / 2) * scale;
    if (cudaCompRenderParameter.mpi_render) // mpi_node_x_offset is measured in pixel
    {
        x_offset += mpiRenderParameter.mpi_node_x_offset * cudaCompRenderParameter.w * scale *
                    cudaCompRenderParameter.w / cudaCompRenderParameter.h;
        y_offset += mpiRenderParameter.mpi_node_y_offset * cudaCompRenderParameter.h * scale;
    }

    float3 pixel_view_pos = cudaCompRenderParameter.view_pos + cudaCompRenderParameter.view_direction +
                            x_offset * cudaCompRenderParameter.right - y_offset * cudaCompRenderParameter.up;
    float3 ray_direction = normalize(pixel_view_pos - cudaCompRenderParameter.view_pos);

    float3 start_pos = cudaCompRenderParameter.view_pos;

    float3 ray_pos = start_pos;
    int last_lod = 0;
    int lod_t = 1;
    float cur_step = cudaCompRenderParameter.step;
    float4 sample_color;
    float4 color = {0.f, 0.f, 0.f, 0.f};

    int steps = 0;
    int lod_steps = 0;
    float3 lod_sample_pos = start_pos;
    float last_scalar = 0.f;
    int cur_lod;
    int nsteps = cudaCompRenderParameter.steps;
    while (steps++ < nsteps)
    {
        if (ray_pos.x < 0.f || ray_pos.x > compVolumeParameter.volume_board.x || ray_pos.y < 0.f ||
            ray_pos.y > compVolumeParameter.volume_board.y || ray_pos.z < 0.f ||
            ray_pos.z > compVolumeParameter.volume_board.z)
        {
            color = {0.f, 0.f, 0.f, 1.f};
            break;
        }
        cur_lod = evaluateLod(length(ray_pos - start_pos));
        if (cur_lod > last_lod)
        {
            cur_step *= 2;
            lod_t *= 2;
            last_lod = cur_lod;
            lod_sample_pos = ray_pos;
            lod_steps = steps;
        }
        if (cur_lod > 6)
            break;

        float3 n_ray_pos = cudaCompRenderParameter.space *
                           GetCDFEmptySkipPos(cur_lod, lod_t, ray_pos / cudaCompRenderParameter.space, ray_direction);

        cur_lod = evaluateLod(length(n_ray_pos - start_pos));

        if (cur_lod > last_lod)
        {
        }
        else
        {
            steps += length(n_ray_pos - ray_pos) / cur_step;
        }
        if (cur_lod > 6)
            break;

        float sample_scalar = 0.f;

        int flag = VirtualSample(cur_lod, lod_t, ray_pos / cudaCompRenderParameter.space, sample_scalar);

        if (flag > 0)
        {

            if (sample_scalar > 0.30f)
            {
                sample_color = tex2D<float4>(preIntTransferFunc, last_scalar, sample_scalar);
                if (sample_color.w > 0.f)
                {
                    last_scalar = sample_scalar;
                    //                    color=sample_color;
                    //                    break;
                    auto shading_color = PhongShading(cur_lod,lod_t,ray_pos/cudaCompRenderParameter.space,make_float3(sample_color),ray_direction);
                    sample_color.x=shading_color.x;
                    sample_color.y=shading_color.y;
                    sample_color.z=shading_color.z;
                    color = color + sample_color * make_float4(sample_color.w, sample_color.w, sample_color.w, 1.f) *
                                        (1.f - color.w);
                    if (color.w > 0.9f)
                        break;
                }
            }
        }
        else
        {
        }
        ray_pos = lod_sample_pos + (steps - lod_steps) * ray_direction * cur_step;
    }
    double gamma = 1.0 / 2.2;
    color.x = powf(color.x, gamma);
    color.y = powf(color.y, gamma);
    color.z = powf(color.z, gamma);
    image[image_idx] = rgbaFloatToUInt(color);
}

} // namespace

static void CreateDeviceRenderImage(uint32_t w, uint32_t h)
{
    if (d_image)
    {
        CUDA_RUNTIME_API_CALL(cudaFree(d_image));
    }
    CUDA_RUNTIME_API_CALL(cudaMalloc(&d_image, (size_t)w * h * sizeof(uint)));
    image_w = w;
    image_h = h;
    assert(d_image);
    CUDA_RUNTIME_API_CALL(cudaMemcpyToSymbol(image, &d_image, sizeof(uint *)));
}

static void CreateDeviceMappingTable(const uint32_t *data, size_t size)
{
    assert(!d_mappingTable && !mappingTableSize);
    mappingTableSize = size / 4;
    CUDA_RUNTIME_API_CALL(cudaMalloc(&d_mappingTable, size * sizeof(size_t)));
    CUDA_RUNTIME_API_CALL(cudaMemcpyToSymbol(mappingTable, &d_mappingTable, sizeof(uint4 *)));
}

static void CreateDeviceMissedBlocks(size_t size)
{
    assert(!d_missedBlocks);
    CUDA_RUNTIME_API_CALL(cudaMalloc(&d_missedBlocks, size * sizeof(uint32_t)));
    CUDA_RUNTIME_API_CALL(cudaMemset(d_missedBlocks, 0, size * sizeof(uint32_t)));
    CUDA_RUNTIME_API_CALL(cudaMemcpyToSymbol(missedBlocks, &d_missedBlocks, sizeof(uint *)));
}

static void CreateDeviceCDFMap(const uint32_t **data, int n, size_t *size)
{
    assert(n <= 10);
    for (int i = 0; i < 10; i++)
    {
        if (d_cdfMap[i])
        {
            CUDA_RUNTIME_API_CALL(cudaFree(d_cdfMap[i]));
        }
    }
    for (int i = 0; i < n; i++)
    {
        CUDA_RUNTIME_API_CALL(cudaMalloc(&d_cdfMap[i], size[i] * sizeof(uint)));
        std::cout << d_cdfMap[i] << std::endl;
    }
    CUDA_RUNTIME_API_CALL(cudaMemcpyToSymbol(cdfMap, &d_cdfMap, sizeof(uint *) * 10));
}

namespace CUDARenderer
{

void UploadTransferFunc(float *data, size_t size)
{
    cudaTextureObject_t tf_tex;
    if (!tf)
    {
        CreateCUDATexture1D(256, &tf, &tf_tex);
    }
    UpdateCUDATexture1D((uint8_t *)data, tf, 256 * 4 * sizeof(float), 0);
    //__constant__ variable's address is decide at compile time, must use cudaMemcpyToSymbol
    CUDA_RUNTIME_API_CALL(cudaMemcpyToSymbol(transferFunc, &tf_tex, sizeof(cudaTextureObject_t)));
    assert(tf);
}

void UploadPreIntTransferFunc(float *data, size_t size)
{
    cudaTextureObject_t pre_tf_tex;
    if (!preInt_tf)
    {
        CreateCUDATexture2D(256, 256, &preInt_tf, &pre_tf_tex);
    }
    UpdateCUDATexture2D((uint8_t *)data, preInt_tf, 256 * 4 * sizeof(float), 256, 0, 0);
    CUDA_RUNTIME_API_CALL(cudaMemcpyToSymbol(preIntTransferFunc, &pre_tf_tex, sizeof(cudaTextureObject_t)));
}

cudaStream_t stream = nullptr;
void CUDACalcBlock(uint32_t *missed_blocks, size_t size, uint32_t w, uint32_t h)
{
    if (!stream)
    {
        cudaStreamCreate(&stream);
    }
    if (!d_missedBlocks)
    {
        CreateDeviceMissedBlocks(size);
    }
    CUDA_RUNTIME_API_CALL(cudaMemset(d_missedBlocks, 0, size * sizeof(uint32_t)));

    dim3 threads_per_block = {16, 16};
    dim3 blocks_per_grid = {(w + threads_per_block.x - 1) / threads_per_block.x,
                            (h + threads_per_block.y - 1) / threads_per_block.y};

    CUDACalcBlockKernel<<<blocks_per_grid, threads_per_block, 0, stream>>>();
    CUDA_RUNTIME_CHECK

    CUDA_RUNTIME_API_CALL(cudaMemcpy(missed_blocks, d_missedBlocks, size * sizeof(uint32_t), cudaMemcpyDeviceToHost));
}

void CUDARender(uint32_t w, uint32_t h, uint8_t *image)
{
    if (image_w != w || image_h != h)
    {
        CreateDeviceRenderImage(w, h);
    }

    dim3 threads_per_block = {16, 16};
    dim3 blocks_per_grid = {(w + threads_per_block.x - 1) / threads_per_block.x,
                            (h + threads_per_block.y - 1) / threads_per_block.y};

    CUDARenderKernel<<<blocks_per_grid, threads_per_block, 0, stream>>>();
    CUDA_RUNTIME_CHECK

    CUDA_RUNTIME_API_CALL(cudaMemcpy(image, d_image, (size_t)w * h * sizeof(uint), cudaMemcpyDeviceToHost));
}

void UploadLightParameter(const LightParameter &light)
{
    CUDA_RUNTIME_API_CALL(cudaMemcpyToSymbol(lightParameter, &light, sizeof(LightParameter)));
}

void UploadCompVolumeParameter(const CompVolumeParameter &comp_volume)
{
    CUDA_RUNTIME_API_CALL(cudaMemcpyToSymbol(compVolumeParameter, &comp_volume, sizeof(CompVolumeParameter)));
}

void UploadCUDACompRenderParameter(const CUDACompRenderParameter &comp_render)
{
    CUDA_RUNTIME_API_CALL(cudaMemcpyToSymbol(cudaCompRenderParameter, &comp_render, sizeof(CUDACompRenderParameter)));
}
void UploadMPIRenderParameter(const MPIRenderParameter &mpi_render)
{
    CUDA_RUNTIME_API_CALL(cudaMemcpyToSymbol(mpiRenderParameter, &mpi_render, sizeof(MPIRenderParameter)));
}

void SetCUDATextureObject(cudaTextureObject_t *textures, size_t size)
{
    CUDA_RUNTIME_API_CALL(cudaMemcpyToSymbol(cacheVolumes, textures, size * sizeof(cudaTextureObject_t)));
}

void UploadMappingTable(const uint32_t *data, size_t size)
{
    if (!d_mappingTable)
    {
        CreateDeviceMappingTable(data, size);
    }
    CUDA_RUNTIME_API_CALL(cudaMemcpy(d_mappingTable, data, size * sizeof(uint32_t), cudaMemcpyHostToDevice));
}

void UploadLodMappingTableOffset(const uint32_t *data, size_t size)
{
    CUDA_RUNTIME_API_CALL(cudaMemcpyToSymbol(lodMappingTableOffset, data, size * sizeof(uint32_t)));
}

void UploadCDFMap(const uint32_t **data, int n, size_t *size)
{
    if (!d_cdfMap[0])
    {
        CreateDeviceCDFMap(data, n, size);
    }
    for (int i = 0; i < n; i++)
    {
        CUDA_RUNTIME_API_CALL(cudaMemcpy(d_cdfMap[i], data[i], size[i] * sizeof(uint32_t), cudaMemcpyHostToDevice));
        int cnt = 0;
        for (int j = 0; j < size[i]; j++)
        {
            if (!data[i][j])
                cnt++;
        }
        std::cout << "Lod " << i << "Upload data size: " << size[i] << " no empty cnt: " << cnt << std::endl;
    }
}

} // namespace CUDARenderer
