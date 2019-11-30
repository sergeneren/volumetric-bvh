//--------------------------------------------------------------------------------
//
//	Redistribution and use in source and binary forms, with or without
//	modification, are permitted provided that the following conditions are met :
//
//	*Redistributions of source code must retain the above copyright notice, this
//	list of conditions and the following disclaimer.
//
//	* Redistributions in binary form must reproduce the above copyright notice,
//	this list of conditions and the following disclaimer in the documentation
//	and/or other materials provided with the distribution.
//	
//	* Neither the name of the copyright holder nor the names of its
//	contributors may be used to endorse or promote products derived from
//	this software without specific prior written permission.
//	
//	THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
//	AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
//	IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
//	DISCLAIMED.IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
//	FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
//	DAMAGES(INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
//	SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
//	CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
//	OR TORT(INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
//	OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
// 
// Copyright(c) 2019, Sergen Eren
// All rights reserved.
//----------------------------------------------------------------------------------
// 
//	Version 1.0: Sergen Eren, 29/11/2019
//
// File: This is the implementation file for BVH_Builder class functions 
//
//-----------------------------------------------

#include <thrust/random.h>
#include <thrust/iterator/counting_iterator.h>
#include <thrust/sequence.h>
#include <thrust/functional.h>
#include <thrust/remove.h>
#include <thrust/copy.h>
#include <thrust/host_vector.h>
#include <thrust/reduce.h>
#include <thrust/sort.h>

#include "bvh_builder.h"

// Initialize kernel launcher functions 
bvh_error_t BVH_Builder::init_functions(CUmodule &bvh_module) {

	CUresult error;
	error = cuModuleGetFunction(&debug_bvh_func, bvh_module, "DebugBVH");
	if (error != CUDA_SUCCESS) {
		printf("Unable to bind debug_bvh function! ");
		printf("error code: %i\n", error);
		return BVH_INIT_FUNC_ERR;
	}

	error = cuModuleGetFunction(&build_radix_tree_func, bvh_module, "BuildRadixTree");
	if (error != CUDA_SUCCESS) {
		printf("Unable to bind BuildRadixTree function! ");
		printf("error code: %i\n", error);
		return BVH_INIT_FUNC_ERR;
	}

	error = cuModuleGetFunction(&comp_morton_codes_func, bvh_module, "ComputeMortonCodes");
	if (error != CUDA_SUCCESS) {
		printf("Unable to bind comp_morton_codes function! ");
		printf("error code: %i\n", error);
		return BVH_INIT_FUNC_ERR;
	}

	error = cuModuleGetFunction(&construct_bvh_func, bvh_module, "ConstructBVH");
	if (error != CUDA_SUCCESS) {
		printf("Unable to bind ConstructBVH function! ");
		printf("error code: %i\n", error);
		return BVH_INIT_FUNC_ERR;
	}

	return BVH_NO_ERR;

}

// Initialize BVH_builder object with a cuda module pointing to the ptx file 
bvh_error_t BVH_Builder::init() {

	// Bind precomputation functions from ptx file 
	CUresult error = cuModuleLoad(&bvh_module, "bvh_kernels.ptx");
	if (error != CUDA_SUCCESS) printf("ERROR: cuModuleLoad, %i\n", error);

	bvh_error_t error_init = init_functions(bvh_module);
	if (error_init != BVH_NO_ERR) {

		printf("Unable to initialize functions!");
		return BVH_INIT_ERR;

	}
	return BVH_NO_ERR;
}


// Build the BVH that will be sent to the render kernel
bvh_error_t BVH_Builder::build_bvh(GPU_VDB *volumes, int num_volumes, AABB &sceneBounds) {
	
	bvh.BVHNodes = new BVHNode[num_volumes-1];
	bvh.BVHLeaves = new BVHNode[num_volumes];

	CUdeviceptr vol_ptr;
	cuMemAlloc(&vol_ptr, sizeof(GPU_VDB) * num_volumes);
	cuMemcpyHtoD(vol_ptr, volumes, sizeof(GPU_VDB) * num_volumes);


	int block_size = BLOCK_SIZE;
	int grid_size = (num_volumes + block_size - 1) / block_size;

	printf("Number of volumes: %i\n", num_volumes);

	// Compute bounding boxes 
	printf("Computing volume bounding boxes...\n");
	thrust::host_vector<AABB> boundingBoxes(num_volumes);
	for (int i = 0; i < num_volumes; ++i) {
		boundingBoxes[i] = volumes[i].Bounds();
	}

	// Compute scene bounding box
	std::cout << "Computing scene bounding box...";
	
	sceneBounds = thrust::reduce(boundingBoxes.begin(), boundingBoxes.end(), AABB(), AABBUnion());
	std::cout << " done! "<< std::endl;

	std::cout << "Scene boundingbox:\n";
	std::cout << "pmin: " << sceneBounds.pmin.x << ", " << sceneBounds.pmin.y << ", " << sceneBounds.pmin.z << std::endl;
	std::cout << "pmax: " << sceneBounds.pmax.x << ", " << sceneBounds.pmax.y << ", " << sceneBounds.pmax.z << std::endl;
	// Pre-process done, start building BVH
	
	// Compute Morton codes
	std::cout << "Computing Morton codes...";
	MortonCode *morton_codes = new MortonCode[num_volumes];
	CUdeviceptr morton_d_pointer;
	cuMemAlloc(&morton_d_pointer, sizeof(MortonCode) * num_volumes);
	cuMemcpyHtoD(morton_d_pointer, morton_codes, sizeof(MortonCode) * num_volumes);
	
	CUresult result;

	void *morton_params[] = { (void *)&vol_ptr, &num_volumes, &sceneBounds, (void*)&morton_d_pointer };
	result = cuLaunchKernel(comp_morton_codes_func, grid_size, 1, 1, block_size, 1, 1, 0, NULL, morton_params, NULL);
	checkCudaErrors(cudaDeviceSynchronize());
	if (result != CUDA_SUCCESS) {
		printf("Unable to launch comp_morton_codes_func! \n");
		return BVH_LAUNCH_ERR;
	}
	cuMemcpyDtoH(morton_codes, morton_d_pointer, sizeof(MortonCode) * num_volumes);
	
	thrust::host_vector<MortonCode> mortonCodes_h(morton_codes, morton_codes+num_volumes);

	std::cout << " done!" << std::endl;

	// Sort triangle indices with Morton code as key
	thrust::host_vector<int> volumeIDs(num_volumes);
	thrust::sequence(volumeIDs.begin(), volumeIDs.end());
	std::cout << "Sorting volumes...";

	try {
		thrust::sort_by_key(mortonCodes_h.begin(), mortonCodes_h.end(), volumeIDs.begin());
	}
	catch (thrust::system_error e) {
		std::cout << "Error inside sort: " << e.what() << std::endl;
	}
	std::cout << " done!" << std::endl;

	/*
	for (int i = 0; i < num_volumes; ++i) {
		std::cout << mortonCodes_h[i] << "\n";
	}*/

	
	// Build radix tree of BVH nodes
	CUdeviceptr bvh_nodes_ptr;
	CUdeviceptr bvh_leaves_ptr;
	
	cuMemAlloc(&bvh_nodes_ptr, (num_volumes - 1) * sizeof(BVHNode));
	cuMemcpyHtoD(bvh_nodes_ptr, &bvh.BVHNodes, (num_volumes - 1) * sizeof(BVHNode));

	cuMemAlloc(&bvh_leaves_ptr, num_volumes * sizeof(BVHNode));
	cuMemcpyHtoD(bvh_leaves_ptr, &bvh.BVHLeaves, num_volumes * sizeof(BVHNode));

	std::cout << "Building radix tree...";

	cuMemcpyHtoD(morton_d_pointer, &mortonCodes_h[0], sizeof(MortonCode) * num_volumes);
	CUdeviceptr volumeID_ptr;
	cuMemAlloc(&volumeID_ptr, sizeof(int) * num_volumes);
	cuMemcpyHtoD(volumeID_ptr, volumeIDs.data(), sizeof(int) * num_volumes);

	void *radix_params[] = { (void*)&bvh_nodes_ptr, (void*)&bvh_leaves_ptr, (void*)&morton_d_pointer, (void*)&volumeID_ptr, &num_volumes };
	result = cuLaunchKernel(build_radix_tree_func, grid_size, 1, 1, block_size, 1, 1, 0, NULL, radix_params, NULL);
	checkCudaErrors(cudaDeviceSynchronize());
	if (result != CUDA_SUCCESS) {
		printf("Unable to launch build_radix_tree_func! \n");
		return BVH_LAUNCH_ERR;
	}
	std::cout << " done!" << std::endl;

	
	// Build BVH
	int *node_counter = new int[num_volumes];
	CUdeviceptr node_counter_ptr;
	cuMemAlloc(&node_counter_ptr, sizeof(int) * num_volumes);
	cuMemcpyHtoD(node_counter_ptr, node_counter, sizeof(int) * num_volumes);

	std::cout << "Building BVH...";
	void *bvh_params[] = { (void*)&bvh_nodes_ptr, (void*)&bvh_leaves_ptr,  (void*)&node_counter_ptr, (void*)&vol_ptr, (void*)&volumeID_ptr, &num_volumes };
	result = cuLaunchKernel(construct_bvh_func, grid_size, 1, 1, block_size, 1, 1, 0, NULL, bvh_params, NULL);
	checkCudaErrors(cudaDeviceSynchronize());
	if (result != CUDA_SUCCESS) {
		printf("Unable to launch construct_bvh_func! \n");
		return BVH_LAUNCH_ERR;
	}
	std::cout << " done!" << std::endl;

	
	if (m_debug_bvh) {
		void *bvh_debug_params[] = { (void*)&bvh_nodes_ptr, (void*)&bvh_leaves_ptr, &num_volumes };
		result = cuLaunchKernel(debug_bvh_func, grid_size, 1, 1, block_size, 1, 1, 0, NULL, bvh_debug_params, NULL);
		checkCudaErrors(cudaDeviceSynchronize());
		if (result != CUDA_SUCCESS) {
			printf("Unable to launch debug_bvh_func! \n");
			return BVH_LAUNCH_ERR;
		}
	}

	cuMemcpyDtoH(bvh.BVHNodes, bvh_nodes_ptr, (num_volumes - 1) * sizeof(BVHNode));
	cuMemcpyDtoH(bvh.BVHLeaves, bvh_leaves_ptr, num_volumes * sizeof(BVHNode));
	

	return BVH_NO_ERR;

}