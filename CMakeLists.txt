cmake_minimum_required( VERSION 3.8 )
project( VDB_BVH LANGUAGES CXX CUDA)

set(CMAKE_INCLUDE_CURRENT_DIR ON)
set(_VCPKG_INCLUDE_DIR "${_VCPKG_INSTALLED_DIR}/${VCPKG_TARGET_TRIPLET}/include/")

find_package(glut REQUIRED)

file(GLOB SOURCE_FILES
			"*.h"
            "*.cpp"
			"*.cu")

add_executable(gpu_bvh ${SOURCE_FILES})

set_target_properties(gpu_bvh PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
target_compile_features(gpu_bvh PUBLIC cxx_std_11)
SET(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS} -gencode arch=compute_30,code=sm_30)


target_link_libraries(gpu_bvh PRIVATE cudart cuda openvdb)

include_directories(${_VCPKG_INCLUDE_DIR})