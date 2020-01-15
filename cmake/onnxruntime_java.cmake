# Copyright (c) 2019, Oracle and/or its affiliates. All rights reserved.
# Licensed under the MIT License.

#set(CMAKE_VERBOSE_MAKEFILE on)

# Setup Java compilation
include(FindJava)
find_package(Java REQUIRED)
find_package(JNI REQUIRED)
include(UseJava)
include_directories(${JNI_INCLUDE_DIRS})
set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -std=c11")

set(JAVA_ROOT ${REPO_ROOT}/java)
set(CMAKE_JAVA_COMPILE_FLAGS "-source" "1.8" "-target" "1.8" "-encoding" "UTF-8")
if (onnxruntime_RUN_ONNX_TESTS)
  set(JAVA_DEPENDS onnxruntime ${test_data_target})
else()
  set(JAVA_DEPENDS onnxruntime)
endif()

# Specify the Java source files
file(GLOB onnxruntime4j_src 
    "${REPO_ROOT}/java/src/main/java/ai/onnxruntime/*.java"
    )

# Build the jar and generate the native headers
add_jar(onnxruntime4j SOURCES ${onnxruntime4j_src} VERSION ${ORT_VERSION} GENERATE_NATIVE_HEADERS onnxruntime4j_generated DESTINATION ${CMAKE_CURRENT_BINARY_DIR}/java-headers/)

# Specify the native sources (without the generated headers)
file(GLOB onnxruntime4j_native_src 
    "${REPO_ROOT}/java/src/main/native/*.c"
    "${REPO_ROOT}/java/src/main/native/*.h"
    "${REPO_ROOT}/include/onnxruntime/core/session/*.h"
    )

# Build the JNI library
add_library(onnxruntime4j_jni SHARED ${onnxruntime4j_native_src} ${onnxruntime4j_generated})
onnxruntime_add_include_to_target(onnxruntime4j_jni onnxruntime_session)
target_include_directories(onnxruntime4j_jni PRIVATE ${REPO_ROOT}/include ${REPO_ROOT}/java/src/main/native)
target_link_libraries(onnxruntime4j_jni PUBLIC ${JNI_LIBRARIES} onnxruntime onnxruntime4j_generated)

if(APPLE)
	set(CLASSIFIER_OS macosx)
endif()
if(WIN32)
	set(CLASSIFIER_OS windows)
endif()
if(UNIX AND NOT APPLE)
	set(CLASSIFIER_OS linux)
endif()

set(GRADLE_SUBPROJECT cpu)
if (onnxruntime_USE_CUDA)
	set(GRADLE_SUBPROJECT gpu)
endif()

if(CLASSIFIER_OS)
	set(JAR_BASE_OUTPUT_DIR build/jni-output/${CLASSIFIER_OS}-${CMAKE_SYSTEM_PROCESSOR}/ai/onnxruntime/native/)
	set(JAR_JNILIB_OUTPUT_DIR ${REPO_ROOT}/java/${JAR_BASE_OUTPUT_DIR})
	# TODO: gpu
	set(JAR_LIB_OUTPUT_DIR ${REPO_ROOT}/java/runtime-${GRADLE_SUBPROJECT}/${JAR_BASE_OUTPUT_DIR})
	
	add_custom_command(TARGET onnxruntime4j_jni POST_BUILD
	    COMMAND ${CMAKE_COMMAND} -E make_directory ${JAR_JNILIB_OUTPUT_DIR})
	add_custom_command(TARGET onnxruntime4j_jni POST_BUILD
	    COMMAND ${CMAKE_COMMAND} -E make_directory ${JAR_LIB_OUTPUT_DIR})
	add_custom_command(TARGET onnxruntime4j_jni POST_BUILD
		COMMAND ${CMAKE_COMMAND} -E copy $<TARGET_FILE:onnxruntime4j_jni> ${JAR_JNILIB_OUTPUT_DIR})
	add_custom_command(TARGET onnxruntime4j_jni POST_BUILD
		COMMAND ${CMAKE_COMMAND} -E copy $<TARGET_LINKER_FILE_NAME:onnxruntime> ${JAR_LIB_OUTPUT_DIR})
endif()
