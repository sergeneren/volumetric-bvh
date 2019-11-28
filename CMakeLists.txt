cmake_minimum_required( VERSION 3.8 )
project( VDB_BVH LANGUAGES CXX CUDA)


file(GLOB SOURCE_FILES
			"*.h"
            "*.cpp"
			"*.cu")

add_executable(gpu_bvh ${SOURCE_FILES})

set_target_properties(gpu_bvh PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
target_compile_features(gpu_bvh PUBLIC cxx_std_11)
SET(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS} -gencode arch=compute_30,code=sm_30)

include_directories(${CMAKE_CURRENT_DIR})