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
// File: Contains the kernels for construction of volume bvh on gpu 
//		 from https://github.com/henrikdahlberg/GPUPathTracer
//
//-----------------------------------------------

#include <cuda_runtime.h>
#include <device_functions.h>
#include <device_launch_parameters.h>

#include <helper_math.h>

#include "bvh.h"
#include "gpu_vdb.h"



//////////////////////////////////////////////////////////////////////////
// Device functions
//////////////////////////////////////////////////////////////////////////

/**
* Longest common prefix for Morton code
*/
__device__ int LongestCommonPrefix(int i, int j, int numTriangles,
	MortonCode* mortonCodes, int* triangleIDs) {
	if (i < 0 || i > numTriangles - 1 || j < 0 || j > numTriangles - 1) {
		return -1;
	}

	MortonCode mi = mortonCodes[i];
	MortonCode mj = mortonCodes[j];

	if (mi == mj) {
		return __clzll(mi ^ mj) + __clzll(triangleIDs[i] ^ triangleIDs[j]);
	}
	else {
		return __clzll(mi ^ mj);
	}
}
/**
* Expand bits, used in Morton code calculation
*/
__device__ MortonCode bitExpansion(MortonCode i) {
	i = (i * 0x00010001u) & 0xFF0000FFu;
	i = (i * 0x00000101u) & 0x0F00F00Fu;
	i = (i * 0x00000011u) & 0xC30C30C3u;
	i = (i * 0x00000005u) & 0x49249249u;
	return i;
}

/**
* Compute morton code given volume centroid scaled to [0,1] of scene bounding box
*/
__device__ MortonCode ComputeMortonCode(float x, float y, float z) {

	x = min(max(x * 1024.0f, 0.0f), 1023.0f);
	y = min(max(y * 1024.0f, 0.0f), 1023.0f);
	z = min(max(z * 1024.0f, 0.0f), 1023.0f);
	MortonCode xx = bitExpansion((MortonCode)x);
	MortonCode yy = bitExpansion((MortonCode)y);
	MortonCode zz = bitExpansion((MortonCode)z);
	return xx * 4 + yy * 2 + zz;

}



//////////////////////////////////////////////////////////////////////////
// Kernels
//////////////////////////////////////////////////////////////////////////


extern "C" __global__ void DebugBVH(BVHNode* BVHLeaves, BVHNode* BVHNodes, int numVolumes) {

	int i = blockIdx.x * blockDim.x + threadIdx.x;

	// do in serial
	if (i == 0) {
		for (int j = 0; j < numVolumes; j++) {
			BVHNode* currentNode = BVHLeaves + j;
			printf("BBox for volumeIdx %d: pmin: (%f,%f,%f), pmax: (%f,%f,%f)\n",
				(BVHLeaves + j)->volIndex,
				currentNode->boundingBox.pmin.x,
				currentNode->boundingBox.pmin.y,
				currentNode->boundingBox.pmin.z,
				currentNode->boundingBox.pmax.x,
				currentNode->boundingBox.pmax.y,
				currentNode->boundingBox.pmax.z);
		}
		//parents:
		for (int j = 0; j < numVolumes; j++) {
			BVHNode* currentNode = (BVHLeaves + j)->parent;
			printf("BBox for parent node of triangleIdx %d: pmin: (%f,%f,%f), pmax: (%f,%f,%f)\n",
				(BVHLeaves + j)->volIndex,
				currentNode->boundingBox.pmin.x,
				currentNode->boundingBox.pmin.y,
				currentNode->boundingBox.pmin.z,
				currentNode->boundingBox.pmax.x,
				currentNode->boundingBox.pmax.y,
				currentNode->boundingBox.pmax.z);
		}

		for (int j = 0; j < numVolumes; j++) {
			BVHNode* currentNode = (BVHLeaves + j)->parent->parent;
			printf("BBox for parents parent node of triangleIdx %d: pmin: (%f,%f,%f), pmax: (%f,%f,%f)\n",
				(BVHLeaves + j)->volIndex,
				currentNode->boundingBox.pmin.x,
				currentNode->boundingBox.pmin.y,
				currentNode->boundingBox.pmin.z,
				currentNode->boundingBox.pmax.x,
				currentNode->boundingBox.pmax.y,
				currentNode->boundingBox.pmax.z);
		}

		for (int j = 0; j < numVolumes; j++) {
			BVHNode* currentNode = (BVHLeaves + j)->parent->parent->parent;
			printf("BBox for parents parents parent node of triangleIdx %d: pmin: (%f,%f,%f), pmax: (%f,%f,%f)\n",
				(BVHLeaves + j)->volIndex,
				currentNode->boundingBox.pmin.x,
				currentNode->boundingBox.pmin.y,
				currentNode->boundingBox.pmin.z,
				currentNode->boundingBox.pmax.x,
				currentNode->boundingBox.pmax.y,
				currentNode->boundingBox.pmax.z);
		}

	}

}

extern "C" __global__ void ComputeMortonCodes(const GPU_VDB* volumes, int numTriangles, AABB sceneBounds, MortonCode* mortonCodes) {

	int i = blockIdx.x * blockDim.x + threadIdx.x;

	
	if (i < numTriangles) {

		// Compute volume centroid
		float3 centroid = volumes[i].Bounds().Centroid();

		// Normalize triangle centroid to lie within [0,1] of scene bounding box
		float x = (centroid.x - sceneBounds.pmin.x) / (sceneBounds.pmax.x - sceneBounds.pmin.x);
		float y = (centroid.y - sceneBounds.pmin.y) / (sceneBounds.pmax.y - sceneBounds.pmin.y);
		float z = (centroid.z - sceneBounds.pmin.z) / (sceneBounds.pmax.z - sceneBounds.pmin.z);

		// Compute morton code
		mortonCodes[i] = ComputeMortonCode(x, y, z);
	}
	
}

extern "C" __global__ void ConstructBVH(BVHNode* BVHNodes, BVHNode* BVHLeaves,
	int* nodeCounter,
	GPU_VDB* volumes,
	int* volumeIDs,
	int numVolumes) {

	int i = blockIdx.x * blockDim.x + threadIdx.x;

	if (i < numVolumes) {
		BVHNode* leaf = BVHLeaves + i;

		int volumeIdx = volumeIDs[i];

		// Handle leaf first
		leaf->volIndex = volumeIdx;
		//printf("%d, %d\n", leaf->triangleIdx, (BVHLeaves + i)->triangleIdx);
		leaf->boundingBox = volumes[volumeIdx].Bounds();

		BVHNode* current = leaf->parent;
		int currentIndex = current - BVHNodes;

		int res = atomicAdd(nodeCounter + currentIndex, 1);

		// Go up and handle internal nodes
		while (true) {
			if (res == 0) {
				return;
			}
			AABB leftBoundingBox = current->leftChild->boundingBox;
			AABB rightBoundingBox = current->rightChild->boundingBox;

			// Compute current bounding box
			current->boundingBox = UnionB(leftBoundingBox,
				rightBoundingBox);

			// If current is root, return
			if (current == BVHNodes) {
				return;
			}
			current = current->parent;
			currentIndex = current - BVHNodes;
			res = atomicAdd(nodeCounter + currentIndex, 1);
		}
	}
}

extern "C" __global__ void BuildRadixTree(BVHNode* radixTreeNodes,
	BVHNode* radixTreeLeaves,
	MortonCode* mortonCodes,
	int* volumeIds,
	int numVolumes) {

	int i = blockIdx.x * blockDim.x + threadIdx.x;

	if (i < numVolumes - 1) {

		// Run radix tree construction algorithm
		// Determine direction of the range (+1 or -1)
		int d = LongestCommonPrefix(i, i + 1, numVolumes, mortonCodes, volumeIds) -
			LongestCommonPrefix(i, i - 1, numVolumes, mortonCodes, volumeIds) >= 0 ? 1 : -1;

		// Compute upper bound for the length of the range
		int deltaMin = LongestCommonPrefix(i, i - d, numVolumes, mortonCodes, volumeIds);
		//int lmax = 128;
		int lmax = 2;

		while (LongestCommonPrefix(i, i + lmax * d, numVolumes, mortonCodes, volumeIds) > deltaMin) {
			//lmax = lmax * 4;
			lmax = lmax * 2;
		}

		// Find the other end using binary search
		int l = 0;
		int divider = 2;
		for (int t = lmax / divider; t >= 1; divider *= 2) {
			if (LongestCommonPrefix(i, i + (l + t) * d, numVolumes, mortonCodes, volumeIds) > deltaMin) {
				l = l + t;
			}
			if (t == 1) break;
			t = lmax / divider;
		}

		int j = i + l * d;

		// Find the split position using binary search
		int deltaNode = LongestCommonPrefix(i, j, numVolumes, mortonCodes, volumeIds);
		int s = 0;
		divider = 2;
		for (int t = (l + (divider - 1)) / divider; t >= 1; divider *= 2) {
			if (LongestCommonPrefix(i, i + (s + t) * d, numVolumes, mortonCodes, volumeIds) > deltaNode) {
				s = s + t;
			}
			if (t == 1) break;
			t = (l + (divider - 1)) / divider;
		}

		int gamma = i + s * d + min(d, 0);

		//printf("i:%d, d:%d, deltaMin:%d, deltaNode:%d, lmax:%d, l:%d, j:%d, gamma:%d. \n", i, d, deltaMin, deltaNode, lmax, l, j, gamma);

		// Output child pointers
		BVHNode* current = radixTreeNodes + i;

		if (min(i, j) == gamma) {
			current->leftChild = radixTreeLeaves + gamma;
			(radixTreeLeaves + gamma)->parent = current;
		}
		else {
			current->leftChild = radixTreeNodes + gamma;
			(radixTreeNodes + gamma)->parent = current;
		}

		if (max(i, j) == gamma + 1) {
			current->rightChild = radixTreeLeaves + gamma + 1;
			(radixTreeLeaves + gamma + 1)->parent = current;
		}
		else {
			current->rightChild = radixTreeNodes + gamma + 1;
			(radixTreeNodes + gamma + 1)->parent = current;
		}

		current->minId = min(i, j);
		current->maxId = max(i, j);
	}
}