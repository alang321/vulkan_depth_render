# CMakeList.txt : CMake project for vulkan_guide, include source and define
# project specific logic here.
#
cmake_minimum_required (VERSION 3.8)

project("vulkan_test" LANGUAGES CXX CUDA)

find_package(Vulkan REQUIRED)
find_package(nlohmann_json 3.2.0 REQUIRED)
find_package(CUDAToolkit REQUIRED)

add_subdirectory(third_party)

if(CMAKE_BUILD_TYPE STREQUAL "Debug")
  add_definitions(-DDEBUG_BUILD)
endif()

set (CMAKE_RUNTIME_OUTPUT_DIRECTORY "${PROJECT_SOURCE_DIR}/bin")
set (CMAKE_LIBRARY_OUTPUT_DIRECTORY "${PROJECT_SOURCE_DIR}/bin")

set(OpenCV_DIR "$ENV{HOME}/opencv-cuda/install/lib/cmake/opencv4")
find_package(OpenCV 4 REQUIRED)

add_executable (vulkan_test 
  src/main_vulkan_test.cpp
  src/vk_descriptors.cpp
  src/vk_engine.cpp
  src/vk_images.cpp
  src/vk_initializers.cpp
  src/vk_pipelines.cpp
  src/parse_flightplan.cpp
  src/vk_cuda_interop.cpp
)

if(SAVE_IMAGES)
  message(STATUS "Saving images")
  add_definitions(-DSAVE_IMAGES=1)
endif()

set_property(TARGET vulkan_test PROPERTY CXX_STANDARD 20)

target_include_directories(vulkan_test PUBLIC include ${OpenCV_INCLUDE_DIRS})


target_link_libraries(vulkan_test 
                    vkbootstrap 
                    vma 
                    glm 
                    Vulkan::Vulkan 
                    stb_image 
                    ${OpenCV_LIBS} 
                    nlohmann_json::nlohmann_json 
                    CUDA::cudart 
                    spdlog 
                    nvinfer 
                    nvonnxparser)

find_program(GLSL_VALIDATOR glslangValidator 
  HINTS /usr/bin /usr/local/bin $ENV{VULKAN_SDK}/Bin/ $ENV{VULKAN_SDK}/Bin32/
)

file(GLOB_RECURSE GLSL_SOURCE_FILES
    "${PROJECT_SOURCE_DIR}/shaders/*.frag"
    "${PROJECT_SOURCE_DIR}/shaders/*.vert"
    "${PROJECT_SOURCE_DIR}/shaders/*.comp"
)

message(STATUS "GLSL source files: ${GLSL_SOURCE_FILES}")

foreach(GLSL ${GLSL_SOURCE_FILES})
  get_filename_component(FILE_NAME ${GLSL} NAME)
  set(SPIRV "${PROJECT_SOURCE_DIR}/shaders/${FILE_NAME}.spv")
  
  add_custom_command(
    OUTPUT ${SPIRV}
    COMMAND ${GLSL_VALIDATOR} -V ${GLSL} -o ${SPIRV}
    DEPENDS ${GLSL}
    VERBATIM
  )
  list(APPEND SPIRV_BINARY_FILES ${SPIRV})
endforeach()

add_custom_target(Shaders DEPENDS ${SPIRV_BINARY_FILES})

add_dependencies(vulkan_test Shaders)
