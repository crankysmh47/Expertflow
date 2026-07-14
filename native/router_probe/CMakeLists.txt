cmake_minimum_required(VERSION 3.20)

project(expertflow_router_probe LANGUAGES CXX)

set(LLAMA_CPP_SOURCE "" CACHE PATH "Path to the pinned llama.cpp source tree")
if(NOT EXISTS "${LLAMA_CPP_SOURCE}/include/llama.h" OR
   NOT EXISTS "${LLAMA_CPP_SOURCE}/ggml/include/ggml.h")
    message(FATAL_ERROR "LLAMA_CPP_SOURCE must contain the pinned llama.cpp headers")
endif()

set(LLAMA_RUNTIME "" CACHE PATH "Path to the verified b10002 runtime DLLs")
set(LLAMA_IMPORT_DIR "" CACHE PATH "Path to generated MinGW import libraries")
foreach(required_file
    "${LLAMA_RUNTIME}/llama.dll"
    "${LLAMA_RUNTIME}/ggml.dll"
    "${LLAMA_RUNTIME}/ggml-base.dll"
    "${LLAMA_IMPORT_DIR}/libllama.dll.a"
    "${LLAMA_IMPORT_DIR}/libggml.dll.a"
    "${LLAMA_IMPORT_DIR}/libggml-base.dll.a"
)
    if(NOT EXISTS "${required_file}")
        message(FATAL_ERROR "Required pinned-runtime file is missing: ${required_file}")
    endif()
endforeach()

add_library(llama_runtime SHARED IMPORTED GLOBAL)
set_target_properties(llama_runtime PROPERTIES
    IMPORTED_LOCATION "${LLAMA_RUNTIME}/llama.dll"
    IMPORTED_IMPLIB "${LLAMA_IMPORT_DIR}/libllama.dll.a"
)
add_library(ggml_runtime SHARED IMPORTED GLOBAL)
set_target_properties(ggml_runtime PROPERTIES
    IMPORTED_LOCATION "${LLAMA_RUNTIME}/ggml.dll"
    IMPORTED_IMPLIB "${LLAMA_IMPORT_DIR}/libggml.dll.a"
)
add_library(ggml_base_runtime SHARED IMPORTED GLOBAL)
set_target_properties(ggml_base_runtime PROPERTIES
    IMPORTED_LOCATION "${LLAMA_RUNTIME}/ggml-base.dll"
    IMPORTED_IMPLIB "${LLAMA_IMPORT_DIR}/libggml-base.dll.a"
)

add_executable(expertflow-router-probe main.cpp)
target_compile_features(expertflow-router-probe PRIVATE cxx_std_17)
target_include_directories(
    expertflow-router-probe
    PRIVATE
        "${LLAMA_CPP_SOURCE}/include"
        "${LLAMA_CPP_SOURCE}/ggml/include"
)
target_link_libraries(
    expertflow-router-probe
    PRIVATE llama_runtime ggml_runtime ggml_base_runtime
)
if(MINGW)
    target_link_options(
        expertflow-router-probe PRIVATE -static -static-libgcc -static-libstdc++
    )
endif()
set_target_properties(
    expertflow-router-probe
    PROPERTIES RUNTIME_OUTPUT_DIRECTORY "${LLAMA_RUNTIME}"
)
