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
//	Version 1.0: Sergen Eren, 25/10/2019
//
// File: This is the implementation file for BVH_Builder class functions 
//
//-----------------------------------------------


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

	error = cuModuleGetFunction(&comp_bbox_func, bvh_module, "ComputeBoundingBoxes");
	if (error != CUDA_SUCCESS) {
		printf("Unable to bind comp_bbox function!\n");
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