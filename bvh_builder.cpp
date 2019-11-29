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
#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include <thrust/reduce.h>
#include <thrust/sort.h>

#include "bvh_builder.h"

bvh_error_t BVH_Builder::init_functions(CUmodule &bvh_module) {

	CUresult error;
	error = cuModuleGetFunction(&debug_bvh_func, bvh_module, "DebugBVH");
	if (error != CUDA_SUCCESS) {
		printf("Unable to bind debug_bvh function!\n");
		return BVH_INIT_FUNC_ERR;
	}

	error = cuModuleGetFunction(&build_radix_tree_func, bvh_module, "BuildRadixTree");
	if (error != CUDA_SUCCESS) {
		printf("Unable to bind BuildRadixTree function!\n");
		return BVH_INIT_FUNC_ERR;
	}

	error = cuModuleGetFunction(&comp_morton_codes_func, bvh_module, "ComputeMortonCodes");
	if (error != CUDA_SUCCESS) {
		printf("Unable to bind comp_morton_codes function!\n");
		return BVH_INIT_FUNC_ERR;
	}

	error = cuModuleGetFunction(&construct_bvh_func, bvh_module, "ConstructBVH");
	if (error != CUDA_SUCCESS) {
		printf("Unable to bind ConstructBVH function!\n");
		return BVH_INIT_FUNC_ERR;
	}

	return BVH_NO_ERR;

}


bvh_error_t BVH_Builder::init() {

	// Bind precomputation functions from ptx file 
	CUresult error = cuModuleLoad(&bvh_module, "bvh_kernels.ptx");
	if (error != CUDA_SUCCESS) printf("ERROR: cuModuleLoad, %i\n", error);

	bvh_error_t error_init = init_functions(bvh_module);
	if (error_init != BVH_NO_ERR) {

		printf("Unable to init functions!");
		return BVH_INIT_ERR;

	}
	return BVH_NO_ERR;
}



bvh_error_t BVH_Builder::build_bvh(std::vector<GPU_VDB> volumes, AABB &sceneBounds) {
	
	int num_volumes = volumes.size();

	int block_size = BLOCK_SIZE;
	int grid_size = (volumes.size() + block_size - 1) / block_size;

	// Timings 
	float total = .0f;
	float elapsed = .0f;

	cudaEvent_t start, stop;

	printf("Number of volumes: %i\n", num_volumes);
	
	cudaEventCreate(&start);
	cudaEventCreate(&stop);

	// Compute bounding boxes 
	printf("Computing triangle bounding boxes...\n");
	thrust::host_vector<AABB> boundingBoxes(num_volumes);
	for (int i = 0; i < num_volumes; ++i) {
		boundingBoxes[i] = volumes.at(i).Bounds();
	}

	// Compute scene bounding box
	std::cout << "Computing scene bounding box...";
	cudaEventRecord(start, 0);
	
	sceneBounds = thrust::reduce(boundingBoxes.begin(), boundingBoxes.end(), AABB(), AABBUnion());
	cudaEventRecord(stop, 0);
	cudaEventSynchronize(stop);
	cudaEventElapsedTime(&elapsed, start, stop);
	std::cout << " done! Computation took " << elapsed << " ms." << std::endl;
	total += elapsed;
	std::cout << "Total pre-computation time for scene was " << total << " ms.\n" << std::endl;
	total = 0;

	std::cout << "Scene boundingbox:\n";
	std::cout << "pmin: " << sceneBounds.pmin.x << ", " << sceneBounds.pmin.y << ", " << sceneBounds.pmin.z << std::endl;
	std::cout << "pmax: " << sceneBounds.pmax.x << ", " << sceneBounds.pmax.y << ", " << sceneBounds.pmax.z << std::endl;
	// Pre-process done, start building BVH

	// Compute Morton codes
	thrust::host_vector<MortonCode> mortonCodes_h(num_volumes);
	thrust::device_vector<MortonCode> mortonCodes_d = mortonCodes_h;
	std::cout << "Computing Morton codes...";
	cudaEventRecord(start, 0);

	CUresult result;

	dim3 block(block_size, 1, 1);
	dim3 grid(grid_size, 1, 1);

	void *morton_params[] = { (void**)volumes.data(), &num_volumes, &sceneBounds, mortonCodes_d.data().get()};
	result = cuLaunchKernel(comp_morton_codes_func, grid.x, 1, 1, block.x, 1, 1, 0, NULL, morton_params, NULL);
	checkCudaErrors(cudaDeviceSynchronize());
	if (result != CUDA_SUCCESS) {
		printf("Unable to launch comp_morton_codes_func! \n");
		return BVH_LAUNCH_ERR;
	}

	mortonCodes_h = mortonCodes_d;
	cudaEventRecord(stop, 0);
	cudaEventSynchronize(stop);
	cudaEventElapsedTime(&elapsed, start, stop);
	std::cout << " done! Computation took " << elapsed << " ms." << std::endl;
	total += elapsed;

	// Sort triangle indices with Morton code as key
	thrust::host_vector<int> volumeIDs(num_volumes);
	thrust::sequence(volumeIDs.begin(), volumeIDs.end());
	std::cout << "Sort triangles...";
	cudaEventRecord(start, 0);

	try {
		thrust::sort_by_key(mortonCodes_h.begin(), mortonCodes_h.end(), volumeIDs.begin());
	}
	catch (thrust::system_error e) {
		std::cout << "Error inside sort: " << e.what() << std::endl;
	}
	cudaEventRecord(stop, 0);
	cudaEventSynchronize(stop);
	cudaEventElapsedTime(&elapsed, start, stop);
	std::cout << " done! Sorting took " << elapsed << " ms." << std::endl;
	total += elapsed;

	// Build radix tree of BVH nodes
	checkCudaErrors(cudaMalloc((void**)&bvh.BVHNodes, (num_volumes - 1) * sizeof(BVHNode)));
	checkCudaErrors(cudaMalloc((void**)&bvh.BVHLeaves, num_volumes * sizeof(BVHNode)));
	std::cout << "Building radix tree...";
	cudaEventRecord(start, 0);

	thrust::device_vector<int> volumeIDs_d = volumeIDs;

	void *radix_params[] = { (void**)bvh.BVHNodes, (void**)bvh.BVHLeaves, mortonCodes_d.data().get(), mortonCodes_d.data().get(), volumeIDs_d.data().get(), &num_volumes };
	result = cuLaunchKernel(build_radix_tree_func, grid.x, 1, 1, block.x, 1, 1, 0, NULL, radix_params, NULL);
	checkCudaErrors(cudaDeviceSynchronize());
	if (result != CUDA_SUCCESS) {
		printf("Unable to launch build_radix_tree_func! \n");
		return BVH_LAUNCH_ERR;
	}
	cudaEventRecord(stop, 0);
	cudaEventSynchronize(stop);
	cudaEventElapsedTime(&elapsed, start, stop);
	std::cout << " done! Took " << elapsed << " ms." << std::endl;
	total += elapsed;

	// Build BVH
	thrust::host_vector<int> nodeCounters_h(num_volumes);
	thrust::device_vector<int> nodeCounters_d = nodeCounters_h;

	std::cout << "Building BVH...";
	cudaEventRecord(start, 0);
	void *bvh_params[] = { (void**)bvh.BVHNodes, (void**)bvh.BVHLeaves, nodeCounters_d.data().get(), (void**)volumes.data(), volumeIDs_d.data().get(), &num_volumes };
	result = cuLaunchKernel(construct_bvh_func, grid.x, 1, 1, block.x, 1, 1, 0, NULL, bvh_params, NULL);
	checkCudaErrors(cudaDeviceSynchronize());
	if (result != CUDA_SUCCESS) {
		printf("Unable to launch construct_bvh_func! \n");
		return BVH_LAUNCH_ERR;
	}
	cudaEventRecord(stop, 0);
	cudaEventSynchronize(stop);
	cudaEventElapsedTime(&elapsed, start, stop);
	std::cout << " done! Took " << elapsed << " ms." << std::endl;
	total += elapsed;

	std::cout << "Total BVH construction time was " << total << " ms.\n" << std::endl;

	cudaEventDestroy(start);
	cudaEventDestroy(stop);

	if (m_debug_bvh) {


		void *bvh_debug_params[] = { (void**)bvh.BVHNodes, (void**)bvh.BVHLeaves, &num_volumes };
		result = cuLaunchKernel(debug_bvh_func, grid.x, 1, 1, block.x, 1, 1, 0, NULL, bvh_debug_params, NULL);
		checkCudaErrors(cudaDeviceSynchronize());
		if (result != CUDA_SUCCESS) {
			printf("Unable to launch debug_bvh_func! \n");
			return BVH_LAUNCH_ERR;
		}


	}

	return BVH_NO_ERR;

}