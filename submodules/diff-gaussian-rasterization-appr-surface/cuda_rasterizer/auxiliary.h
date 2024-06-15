/*
 * Copyright (C) 2023, Inria
 * GRAPHDECO research group, https://team.inria.fr/graphdeco
 * All rights reserved.
 *
 * This software is free for non-commercial, research and evaluation use 
 * under the terms of the LICENSE.md file.
 *
 * For inquiries contact  george.drettakis@inria.fr
 */

#ifndef CUDA_RASTERIZER_AUXILIARY_H_INCLUDED
#define CUDA_RASTERIZER_AUXILIARY_H_INCLUDED

#include "config.h"
#include "stdio.h"

#define BLOCK_SIZE (BLOCK_X * BLOCK_Y)
#define NUM_WARPS (BLOCK_SIZE/32)

// Spherical harmonics coefficients
__device__ const float SH_C0 = 0.28209479177387814f;
__device__ const float SH_C1 = 0.4886025119029199f;
__device__ const float SH_C2[] = {
	1.0925484305920792f,
	-1.0925484305920792f,
	0.31539156525252005f,
	-1.0925484305920792f,
	0.5462742152960396f
};
__device__ const float SH_C3[] = {
	-0.5900435899266435f,
	2.890611442640554f,
	-0.4570457994644658f,
	0.3731763325901154f,
	-0.4570457994644658f,
	1.445305721320277f,
	-0.5900435899266435f
};

__forceinline__ __device__ float _get_ellipsoid_scaling(const float& scaling)
{
	return scaling * 2.0f;
	// if ( scaling <= 1.0f ) {
	// 	return scaling * 0.6501f;
	// } else if ( scaling <= 10.0f ) {
	// 	return 2.0999f * (scaling - 1.0f) + 0.6501f;
	// } else {
	// 	return 2.8212f * (scaling - 10.0f) + 19.5556f;
	// }
}

__forceinline__ __device__ float _get_ellipsoid_scaling_gradient(const float& scaling, const float& dL_dellipsoid_scaling)
{
	return dL_dellipsoid_scaling * 2.0f;
	// if ( scaling <= 1.0f ) {
	// 	return dL_dellipsoid_scaling * 0.6501f;
	// } else if ( scaling <= 10.0f ) {
	// 	return dL_dellipsoid_scaling * 2.0999f;
	// } else {
	// 	return dL_dellipsoid_scaling * 2.8212f;
	// }
}

__forceinline__ __device__ glm::vec3 get_ellipsoid_scaling(const glm::vec3& scaling)
{
	return glm::vec3(
		_get_ellipsoid_scaling(scaling.x), 
		_get_ellipsoid_scaling(scaling.y), 
		_get_ellipsoid_scaling(scaling.z)
	);
}

__forceinline__ __device__ glm::vec3 get_ellipsoid_scaling_gradient(const glm::vec3& scaling, const glm::vec3& dL_dellipsoid_scaling) {
	return glm::vec3(
		_get_ellipsoid_scaling_gradient(scaling.x, dL_dellipsoid_scaling.x), 
		_get_ellipsoid_scaling_gradient(scaling.y, dL_dellipsoid_scaling.y), 
		_get_ellipsoid_scaling_gradient(scaling.z, dL_dellipsoid_scaling.z)
	);
}

__forceinline__ __device__ float ndc2Pix(float v, int S)
{
	return ((v + 1.0) * S - 1.0) * 0.5;
}

__forceinline__ __device__ float Pix2ndc(float v, int S)
{
	return (2.0 * v + 1.0) / S - 1.0;
}

__forceinline__ __device__ glm::mat3 get_rotation_matrix_from_quaternion(const glm::vec4& q) {
	// Normalize quaternion to get valid rotation
	float r = q.x;
	float x = q.y;
	float y = q.z;
	float z = q.w;

	// Compute rotation matrix from quaternion
	return glm::mat3(
		1.f - 2.f * (y * y + z * z), 2.f * (x * y - r * z), 2.f * (x * z + r * y),
		2.f * (x * y + r * z), 1.f - 2.f * (x * x + z * z), 2.f * (y * z - r * x),
		2.f * (x * z - r * y), 2.f * (y * z + r * x), 1.f - 2.f * (x * x + y * y)
	);
}

__forceinline__ __device__ glm::vec4 get_rotation_matrix_from_quaternion_gradient(const glm::vec4& q, const glm::mat3& dL_dMt) {
	// Normalize quaternion to get valid rotation
	float r = q.x;
	float x = q.y;
	float y = q.z;
	float z = q.w;
	
	// Gradients of loss w.r.t. normalized quaternion
	glm::vec4 dL_dq;
	dL_dq.x = 2 * z * (dL_dMt[0][1] - dL_dMt[1][0]) + 2 * y * (dL_dMt[2][0] - dL_dMt[0][2]) + 2 * x * (dL_dMt[1][2] - dL_dMt[2][1]);
	dL_dq.y = 2 * y * (dL_dMt[1][0] + dL_dMt[0][1]) + 2 * z * (dL_dMt[2][0] + dL_dMt[0][2]) + 2 * r * (dL_dMt[1][2] - dL_dMt[2][1]) - 4 * x * (dL_dMt[2][2] + dL_dMt[1][1]);
	dL_dq.z = 2 * x * (dL_dMt[1][0] + dL_dMt[0][1]) + 2 * r * (dL_dMt[2][0] - dL_dMt[0][2]) + 2 * z * (dL_dMt[1][2] + dL_dMt[2][1]) - 4 * y * (dL_dMt[2][2] + dL_dMt[0][0]);
	dL_dq.w = 2 * r * (dL_dMt[0][1] - dL_dMt[1][0]) + 2 * x * (dL_dMt[2][0] + dL_dMt[0][2]) + 2 * y * (dL_dMt[1][2] + dL_dMt[2][1]) - 4 * z * (dL_dMt[1][1] + dL_dMt[0][0]);

	return dL_dq;
}

__forceinline__ __device__ void rotationGradient(const glm::vec3& p, const glm::vec3& dp, glm::mat3& drotation) {
	drotation[0][0] += dp.x * p.x;
	drotation[0][1] += dp.y * p.x;
	drotation[0][2] += dp.z * p.x;
	drotation[1][0] += dp.x * p.y;
	drotation[1][1] += dp.y * p.y;
	drotation[1][2] += dp.z * p.y;
	drotation[2][0] += dp.x * p.z;
	drotation[2][1] += dp.y * p.z;
	drotation[2][2] += dp.z * p.z;
}

__forceinline__ __device__ void rotationTransposeGradient(const glm::vec3& p, const glm::vec3& dp, glm::mat3& drotation) {
	drotation[0][0] += dp.x * p.x;
	drotation[1][0] += dp.y * p.x;
	drotation[2][0] += dp.z * p.x;
	drotation[0][1] += dp.x * p.y;
	drotation[1][1] += dp.y * p.y;
	drotation[2][1] += dp.z * p.y;
	drotation[0][2] += dp.x * p.z;
	drotation[1][2] += dp.y * p.z;
	drotation[2][2] += dp.z * p.z;
}

__forceinline__ __device__ void getRect(const float2 p, int max_radius, uint2& rect_min, uint2& rect_max, dim3 grid)
{
	rect_min = {
		min(grid.x, max((int)0, (int)((p.x - max_radius) / BLOCK_X))),
		min(grid.y, max((int)0, (int)((p.y - max_radius) / BLOCK_Y)))
	};
	rect_max = {
		min(grid.x, max((int)0, (int)((p.x + max_radius + BLOCK_X - 1) / BLOCK_X))),
		min(grid.y, max((int)0, (int)((p.y + max_radius + BLOCK_Y - 1) / BLOCK_Y)))
	};
}

__forceinline__ __device__ float3 rotatePoint4x3(const float3& p, const float* matrix)
{
	float3 transformed = {
		matrix[0] * p.x + matrix[4] * p.y + matrix[8] * p.z,
		matrix[1] * p.x + matrix[5] * p.y + matrix[9] * p.z,
		matrix[2] * p.x + matrix[6] * p.y + matrix[10] * p.z,
	};
	return transformed;
}

__forceinline__ __device__ float3 rotatePoint4x3Transpose(const float3& p, const float* matrix)
{
	float3 transformed = {
		matrix[0] * p.x + matrix[1] * p.y + matrix[2] * p.z,
		matrix[4] * p.x + matrix[5] * p.y + matrix[6] * p.z,
		matrix[8] * p.x + matrix[9] * p.y + matrix[10] * p.z,
	};
	return transformed;
}

__forceinline__ __device__ float3 transformPoint4x3(const float3& p, const float* matrix)
{
	float3 transformed = {
		matrix[0] * p.x + matrix[4] * p.y + matrix[8] * p.z + matrix[12],
		matrix[1] * p.x + matrix[5] * p.y + matrix[9] * p.z + matrix[13],
		matrix[2] * p.x + matrix[6] * p.y + matrix[10] * p.z + matrix[14],
	};
	return transformed;
}

__forceinline__ __device__ void transformPoint4x3MatrixGradient(const float3& p, const float3& dp, float* dmatrix)
{
	atomicAdd(&dmatrix[0], dp.x * p.x);
	atomicAdd(&dmatrix[1], dp.y * p.x);
	atomicAdd(&dmatrix[2], dp.z * p.x);

	atomicAdd(&dmatrix[4], dp.x * p.y);
	atomicAdd(&dmatrix[5], dp.y * p.y);
	atomicAdd(&dmatrix[6], dp.z * p.y);
	
	atomicAdd(&dmatrix[8], dp.x * p.z);
	atomicAdd(&dmatrix[9], dp.y * p.z);
	atomicAdd(&dmatrix[10], dp.z * p.z);
	
	atomicAdd(&dmatrix[12], dp.x);
	atomicAdd(&dmatrix[13], dp.y);
	atomicAdd(&dmatrix[14], dp.z);
}

__forceinline__ __device__ float3 transformPoint4x3Transpose(const float3& p, const float* matrix)
{
	float3 transformed = {
		matrix[0] * p.x + matrix[1] * p.y + matrix[2] * p.z,
		matrix[4] * p.x + matrix[5] * p.y + matrix[6] * p.z,
		matrix[8] * p.x + matrix[9] * p.y + matrix[10] * p.z,
	};
	return transformed;
}

__forceinline__ __device__ float4 transformPoint4x4(const float3& p, const float* matrix)
{
	float4 transformed = {
		matrix[0] * p.x + matrix[4] * p.y + matrix[8] * p.z + matrix[12],
		matrix[1] * p.x + matrix[5] * p.y + matrix[9] * p.z + matrix[13],
		matrix[2] * p.x + matrix[6] * p.y + matrix[10] * p.z + matrix[14],
		matrix[3] * p.x + matrix[7] * p.y + matrix[11] * p.z + matrix[15]
	};
	return transformed;
}

__forceinline__ __device__ void transformPoint4x4MatrixGradient(const float3& p, const float4& dp, float* dmatrix)
{
	atomicAdd(&dmatrix[0], dp.x * p.x);
	atomicAdd(&dmatrix[1], dp.y * p.x);
	atomicAdd(&dmatrix[2], dp.z * p.x);
	atomicAdd(&dmatrix[3], dp.w * p.x);
	atomicAdd(&dmatrix[4], dp.x * p.y);
	atomicAdd(&dmatrix[5], dp.y * p.y);
	atomicAdd(&dmatrix[6], dp.z * p.y);
	atomicAdd(&dmatrix[7], dp.w * p.y);
	atomicAdd(&dmatrix[8], dp.x * p.z);
	atomicAdd(&dmatrix[9], dp.y * p.z);
	atomicAdd(&dmatrix[10], dp.z * p.z);
	atomicAdd(&dmatrix[11], dp.w * p.z);
	atomicAdd(&dmatrix[12], dp.x);
	atomicAdd(&dmatrix[13], dp.y);
	atomicAdd(&dmatrix[14], dp.z);
	atomicAdd(&dmatrix[15], dp.w);
}

__forceinline__ __device__ float3 transformPoint4x4Transpose(const float4& p, const float* matrix)
{
	float3 transformed = {
		matrix[0] * p.x + matrix[1] * p.y + matrix[2] * p.z + matrix[3] * p.w,
		matrix[4] * p.x + matrix[5] * p.y + matrix[6] * p.z + matrix[7] * p.w,
		matrix[8] * p.x + matrix[9] * p.y + matrix[10] * p.z + matrix[11] * p.w,
	};
	return transformed;
}

__forceinline__ __device__ float3 transformVec4x3(const float3& p, const float* matrix)
{
	float3 transformed = {
		matrix[0] * p.x + matrix[4] * p.y + matrix[8] * p.z,
		matrix[1] * p.x + matrix[5] * p.y + matrix[9] * p.z,
		matrix[2] * p.x + matrix[6] * p.y + matrix[10] * p.z,
	};
	return transformed;
}

__forceinline__ __device__ float3 transformVec4x3Transpose(const float3& p, const float* matrix)
{
	float3 transformed = {
		matrix[0] * p.x + matrix[1] * p.y + matrix[2] * p.z,
		matrix[4] * p.x + matrix[5] * p.y + matrix[6] * p.z,
		matrix[8] * p.x + matrix[9] * p.y + matrix[10] * p.z,
	};
	return transformed;
}

__forceinline__ __device__ float dnormvdz(float3 v, float3 dv)
{
	float sum2 = v.x * v.x + v.y * v.y + v.z * v.z;
	float invsum32 = 1.0f / sqrt(sum2 * sum2 * sum2);
	float dnormvdz = (-v.x * v.z * dv.x - v.y * v.z * dv.y + (sum2 - v.z * v.z) * dv.z) * invsum32;
	return dnormvdz;
}

__forceinline__ __device__ float3 dnormvdv(float3 v, float3 dv)
{
	float sum2 = v.x * v.x + v.y * v.y + v.z * v.z;
	float invsum32 = 1.0f / sqrt(sum2 * sum2 * sum2);

	float3 dnormvdv;
	dnormvdv.x = ((+sum2 - v.x * v.x) * dv.x - v.y * v.x * dv.y - v.z * v.x * dv.z) * invsum32;
	dnormvdv.y = (-v.x * v.y * dv.x + (sum2 - v.y * v.y) * dv.y - v.z * v.y * dv.z) * invsum32;
	dnormvdv.z = (-v.x * v.z * dv.x - v.y * v.z * dv.y + (sum2 - v.z * v.z) * dv.z) * invsum32;
	return dnormvdv;
}

__forceinline__ __device__ float4 dnormvdv(float4 v, float4 dv)
{
	float sum2 = v.x * v.x + v.y * v.y + v.z * v.z + v.w * v.w;
	float invsum32 = 1.0f / sqrt(sum2 * sum2 * sum2);

	float4 vdv = { v.x * dv.x, v.y * dv.y, v.z * dv.z, v.w * dv.w };
	float vdv_sum = vdv.x + vdv.y + vdv.z + vdv.w;
	float4 dnormvdv;
	dnormvdv.x = ((sum2 - v.x * v.x) * dv.x - v.x * (vdv_sum - vdv.x)) * invsum32;
	dnormvdv.y = ((sum2 - v.y * v.y) * dv.y - v.y * (vdv_sum - vdv.y)) * invsum32;
	dnormvdv.z = ((sum2 - v.z * v.z) * dv.z - v.z * (vdv_sum - vdv.z)) * invsum32;
	dnormvdv.w = ((sum2 - v.w * v.w) * dv.w - v.w * (vdv_sum - vdv.w)) * invsum32;
	return dnormvdv;
}

__forceinline__ __device__ float sigmoid(float x)
{
	return 1.0f / (1.0f + expf(-x));
}

__forceinline__ __device__ bool in_frustum(int idx,
	const float* orig_points,
	const float* viewmatrix,
	const float* projmatrix,
	bool prefiltered,
	float3& p_view, 
	float& dist)
{
	float3 p_orig = { orig_points[3 * idx], orig_points[3 * idx + 1], orig_points[3 * idx + 2] };

	// Bring points to screen space
	float4 p_hom = transformPoint4x4(p_orig, projmatrix);
	float p_w = 1.0f / (p_hom.w + 0.0000001f);
	float3 p_proj = { p_hom.x * p_w, p_hom.y * p_w, p_hom.z * p_w };
	p_view = transformPoint4x3(p_orig, viewmatrix);
	dist = glm::length(glm::vec3(p_view.x, p_view.y, p_view.z));
	
	if (p_view.z <= 0.2f)// dist <= 0.2f || ((p_proj.x < -1.3 || p_proj.x > 1.3 || p_proj.y < -1.3 || p_proj.y > 1.3)))
	{
		if (prefiltered)
		{
			printf("Point is filtered although prefiltered is set. This shouldn't happen!");
			__trap();
		}
		return false;
	}
	return true;
}

__forceinline__ __device__ glm::vec3 get_ray_dir(const float2& uv, const float* intrinsicmatrix, const float* worldmatrix, const glm::vec3& ray_origins) {
	float fx = intrinsicmatrix[0];
	float fy = intrinsicmatrix[4];
	float cx = intrinsicmatrix[6];
	float cy = intrinsicmatrix[7];
	float sk = intrinsicmatrix[3];

	float x_cam = uv.x;
	float y_cam = uv.y;
	float z_cam = -1.0f;

	float x_lift = (x_cam - cx + cy * sk / fy - sk * y_cam / fy) / fx * z_cam;
	float y_lift = (y_cam - cy) / fy * z_cam;

	float3 world_rel_points = transformPoint4x3({ x_lift, y_lift, z_cam }, worldmatrix);

	return glm::vec3(
		- (world_rel_points.x - ray_origins.x), 
		- (world_rel_points.y - ray_origins.y), 
		- (world_rel_points.z - ray_origins.z)
	);
}

__forceinline__ __device__ void get_ray_dir_gradient(const float2& uv, const float* intrinsicmatrix, const float* worldmatrix, const glm::vec3& dL_dray_dirs, 
float* dL_dintrinsicmatrix, float* dL_dworldmatrix) {
	float fx = intrinsicmatrix[0];
	float fy = intrinsicmatrix[4];
	float cx = intrinsicmatrix[6];
	float cy = intrinsicmatrix[7];
	float sk = intrinsicmatrix[3];

	float x_cam = uv.x;
	float y_cam = uv.y;
	float z_cam = -1.0f;

	float x_lift = (x_cam - cx + cy * sk / fy - sk * y_cam / fy) / fx * z_cam;
	float y_lift = (y_cam - cy) / fy * z_cam;

	// dL_dray_origins += dL_dray_dirs;

	float dL_dworld_rel_points_x = - dL_dray_dirs.x;
	float dL_dworld_rel_points_y = - dL_dray_dirs.y;
	float dL_dworld_rel_points_z = - dL_dray_dirs.z;

	atomicAdd(&dL_dworldmatrix[0], dL_dworld_rel_points_x * x_lift);
	atomicAdd(&dL_dworldmatrix[4], dL_dworld_rel_points_x * y_lift);
	atomicAdd(&dL_dworldmatrix[8], dL_dworld_rel_points_x * z_cam);
	// atomicAdd(&dL_dworldmatrix[12], dL_dworld_rel_points_x);

	atomicAdd(&dL_dworldmatrix[1], dL_dworld_rel_points_y * x_lift);
	atomicAdd(&dL_dworldmatrix[5], dL_dworld_rel_points_y * y_lift);
	atomicAdd(&dL_dworldmatrix[9], dL_dworld_rel_points_y * z_cam);
	// atomicAdd(&dL_dworldmatrix[13], dL_dworld_rel_points_y);

	atomicAdd(&dL_dworldmatrix[2], dL_dworld_rel_points_z * x_lift);
	atomicAdd(&dL_dworldmatrix[6], dL_dworld_rel_points_z * y_lift);
	atomicAdd(&dL_dworldmatrix[10], dL_dworld_rel_points_z * z_cam);
	// atomicAdd(&dL_dworldmatrix[14], dL_dworld_rel_points_z);

	float3 dL_dxyz_lift = transformPoint4x3Transpose({ dL_dworld_rel_points_x, dL_dworld_rel_points_y, dL_dworld_rel_points_z }, worldmatrix);

	float dL_dx_lift = dL_dxyz_lift.x;
	float dL_dy_lift = dL_dxyz_lift.y;

	// float x_lift = (x_cam - cx + cy * sk / fy - sk * y_cam / fy) / fx * z_cam;
	// float y_lift = (y_cam - cy) / fy * z_cam;

	float one_over_fx = 1.0f / fx;
	float one_over_fy = 1.0f / fy;
	float ifx_ify = one_over_fx * one_over_fy;

	atomicAdd(&dL_dintrinsicmatrix[0], - one_over_fx * x_lift * dL_dx_lift);
	atomicAdd(&dL_dintrinsicmatrix[4], - one_over_fy * ((cy * sk - sk * y_cam) * ifx_ify * z_cam * dL_dx_lift + y_lift * dL_dy_lift));
	atomicAdd(&dL_dintrinsicmatrix[6], one_over_fx * z_cam * dL_dx_lift);
	atomicAdd(&dL_dintrinsicmatrix[7], sk * ifx_ify * z_cam * dL_dx_lift - one_over_fy * z_cam * dL_dy_lift);
	atomicAdd(&dL_dintrinsicmatrix[3], (cy - y_cam) * ifx_ify * z_cam * dL_dx_lift);
}

#define CHECK_CUDA(A, debug) \
A; if(debug) { \
auto ret = cudaDeviceSynchronize(); \
if (ret != cudaSuccess) { \
std::cerr << "\n[CUDA ERROR] in " << __FILE__ << "\nLine " << __LINE__ << ": " << cudaGetErrorString(ret); \
throw std::runtime_error(cudaGetErrorString(ret)); \
} \
}

#endif