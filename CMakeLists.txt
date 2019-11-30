cmake_minimum_required( VERSION 3.8 )
project( VDB_BVH LANGUAGES CXX CUDA)


## Set output directories

set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bvh/lib)
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bvh/lib)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bvh/bin)

set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY_DEBUG ${CMAKE_BINARY_DIR}/bvh/lib)
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY_DEBUG ${CMAKE_BINARY_DIR}/bvh/lib)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY_DEBUG ${CMAKE_BINARY_DIR}/bvh/bin)

set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY_RELEASE ${CMAKE_BINARY_DIR}/bvh/lib)
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY_RELEASE ${CMAKE_BINARY_DIR}/bvh/lib)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY_RELEASE ${CMAKE_BINARY_DIR}/bvh/bin)


set(CMAKE_INCLUDE_CURRENT_DIR ON)
set(_VCPKG_INCLUDE_DIR "${_VCPKG_INSTALLED_DIR}/${VCPKG_TARGET_TRIPLET}/include/")

find_package(glut REQUIRED)

file(GLOB SOURCE_FILES
			"*.h"
            "*.cpp")
			
file(GLOB COMMON_FILES
			"common/*.h"
            "common/*.cpp")

file(GLOB CUDA_FILES "*.cu")

source_group(common FILES ${COMMON_FILES})

if(false)
#------------------------------------ CROSS-PLATFORM PTX COMPILE 
#
# _COMPILEPTX( SOURCES file1.cu file2.cu TARGET_PATH <path where ptxs should be stored> GENERATED_FILES ptx_sources NVCC_OPTIONS -arch=sm_20)
# Generates ptx files for the given source files. ptx_sources will contain the list of generated files.
#
FUNCTION( _COMPILEPTX )
  set(options "")
  set(oneValueArgs TARGET_PATH GENERATED GENPATHS INCLUDE)  
  set(multiValueArgs OPTIONS SOURCES)
  CMAKE_PARSE_ARGUMENTS( _COMPILEPTX "${options}" "${oneValueArgs}" "${multiValueArgs}" ${ARGN})
  
  # Match the bitness of the ptx to the bitness of the application
  set( MACHINE "--machine=32" )
  if( CMAKE_SIZEOF_VOID_P EQUAL 8)
    set( MACHINE "--machine=64" )
  endif()
  unset ( PTX_FILES CACHE )
  unset ( PTX_FILES_PATH CACHE )

  set( USE_DEBUG_PTX OFF CACHE BOOL "Enable CUDA debugging with NSight")  
  if ( USE_DEBUG_PTX )
	 set ( DEBUG_FLAGS ";-g;-G;-D_DEBUG;-DEBUG")
  else()
	 set ( DEBUG_FLAGS "")
  endif()
  
  if ( WIN32 ) 
		# Windows - PTX compile
		file ( MAKE_DIRECTORY "${_COMPILEPTX_TARGET_PATH}" )
		string (REPLACE ";" " " _COMPILEPTX_OPTIONS "${_COMPILEPTX_OPTIONS}")  
		separate_arguments( _OPTS WINDOWS_COMMAND "${_COMPILEPTX_OPTIONS}" )
		message ( STATUS "NVCC Options: ${_COMPILEPTX_OPTIONS}" )  
		message ( STATUS "NVCC Include: ${_COMPILEPTX_INCLUDE}" )

        set ( INCL "-I\"${_COMPILEPTX_INCLUDE}\"" )

		# Custom build rule to generate ptx files from cuda files
		FOREACH( input ${_COMPILEPTX_SOURCES} )
			get_filename_component( input_ext ${input} EXT )									# Input extension
			get_filename_component( input_without_ext ${input} NAME_WE )						# Input base
			if ( ${input_ext} STREQUAL ".cu" )			
				
				# Set output names
				set( output "${input_without_ext}.ptx" )							# Output name
				set( output_with_path "${_COMPILEPTX_TARGET_PATH}/${input_without_ext}.ptx" )	# Output with path
				set( output_with_quote "\"${output_with_path}\"" )
				LIST( APPEND PTX_FILES ${output} )		# Append to output list
				LIST( APPEND PTX_FILES_PATH ${output_with_path} )
    
				message( STATUS "NVCC Compile: ${CUDA_NVCC_EXECUTABLE} ${MACHINE} --ptx ${_COMPILEPTX_OPTIONS} ${input} ${INCL} -o ${output_with_path} WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}")
    
				add_custom_command(
					OUTPUT  ${output_with_path}
					MAIN_DEPENDENCY ${input}
					COMMAND ${CUDA_NVCC_EXECUTABLE} ${MACHINE} --ptx ${_OPTS} ${input} ${INCL} -o ${output_with_quote} WORKING_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}"
				)			
			endif()
		ENDFOREACH( )
  else ()
		# Linux - PTX compile
		file ( MAKE_DIRECTORY "${_COMPILEPTX_TARGET_PATH}" )
    FOREACH(input ${_COMPILEPTX_SOURCES})
      get_filename_component( input_ext ${input} EXT )									# Input extension
      get_filename_component( input_without_ext ${input} NAME_WE )						# Input base
      if ( ${input_ext} STREQUAL ".cu" )			
        # Set output names
        set( output "${input_without_ext}.ptx" ) # Output name
        set( output_with_path "${_COMPILEPTX_TARGET_PATH}/${input_without_ext}.ptx" )	# Output with path

        set( compile_target_ptx "${input_without_ext}_PTX")
        set( custom_command_var "${input_without_ext}_OUTPUT")
        # compile ptx
        cuda_compile_ptx(custom_command_var ${input} OPTIONS "${DEBUG_FLAGS}")
        # This will only configure file generation, we need to add a target to
        # generate a file cuda_generated_<counter>_${input_without_ext}.ptx
        # Add custom command to rename to simply ${input_without_ext}.ptx
        add_custom_command(OUTPUT ${output_with_path}
                          COMMAND ${CMAKE_COMMAND} -E rename ${custom_command_var} ${output_with_path}
                          DEPENDS ${custom_command_var})
        add_custom_target(${compile_target_ptx} ALL DEPENDS ${input} ${output_with_path} SOURCES ${input})

        # Add this output file to list of generated ptx files
        LIST(APPEND PTX_FILES ${output})
        LIST(APPEND PTX_FILES_PATH ${output_with_path} )
      endif()
    ENDFOREACH()
  endif()

  set( ${_COMPILEPTX_GENERATED} ${PTX_FILES} PARENT_SCOPE)
  set( ${_COMPILEPTX_GENPATHS} ${PTX_FILES_PATH} PARENT_SCOPE)

ENDFUNCTION()



find_program(CUDA_NVCC_EXECUTABLE nvcc)
message ( STATUS "Build CUDA kernels: ${CUDA_FILES}" )
_COMPILEPTX ( SOURCES ${CUDA_FILES} 
	TARGET_PATH ${CMAKE_RUNTIME_OUTPUT_DIRECTORY} 
	GENERATED CUDA_PTX GENPATHS CUDA_PTX_PATHS 
	INCLUDE "${CMAKE_CURRENT_SOURCE_DIR},${_VCPKG_INCLUDE_DIR},${CMAKE_CURRENT_SOURCE_DIR}/common" 
	OPTIONS -arch=compute_30 -code=sm_30 --ptxas-options=-v -O3 --use_fast_math --maxrregcount=128)
endif()

add_executable(gpu_bvh ${SOURCE_FILES} ${COMMON_FILES} ${CUDA_FILES})

target_link_libraries(gpu_bvh PRIVATE cudart cuda openvdb)

include_directories(${_VCPKG_INCLUDE_DIR})
include_directories(${CMAKE_SOURCE_DIR}/common)