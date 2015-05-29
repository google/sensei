cmake_minimum_required (VERSION 2.8.8)
project (Sensei)

# Handle the dependencies
file(GLOB BASE_SRCS "sensei/base/*.cc")
file(GLOB CONTAINERS_SRCS "sensei/containers/*.cc")
file(GLOB FILE_SRCS "sensei/file/*.cc")
file(GLOB PROTO_SRCS "sensei/proto/*.cc")
file(GLOB STRINGS_SRCS "sensei/strings/*.cc")
file(GLOB THREAD_SRCS "sensei/thread/*.cc")
file(GLOB UTIL_SRCS "sensei/util/*.cc")
file(GLOB UTF_SRCS "sensei/util/utf/*.c")

set(DEPS_SRCS
  ${BASE_SRCS}
  ${CONTAINERS_SRCS}
  ${FILE_SRCS}
  ${PROTO_SRCS}
  ${STRINGS_SRCS}
  ${THREAD_SRCS}
  ${UTIL_SRCS}
  ${UTF_SRCS}
)

# Compile protos
find_package(Protobuf REQUIRED)
include_directories(${PROTOBUF_INCLUDE_DIRS})

set(PROTOS common common_config config log internal)

set(PB_GENERATED_SRCS)
# PROTOBUF_GENERATE_CPP is too stupid, we need a loop
foreach(PROTO ${PROTOS})
  list(APPEND PB_GENERATED_SRCS "${PROJECT_SOURCE_DIR}/sensei/${PROTO}.pb.cc")
  list(APPEND PB_GENERATED_SRCS "${PROJECT_SOURCE_DIR}/sensei/${PROTO}.pb.h")
  set(PROTO_PATH "${PROJECT_SOURCE_DIR}/sensei/${PROTO}.proto")
  add_custom_command(
    OUTPUT "${PROJECT_SOURCE_DIR}/sensei/${PROTO}.pb.cc"
           "${PROJECT_SOURCE_DIR}/sensei/${PROTO}.pb.h"
    COMMAND  ${PROTOBUF_PROTOC_EXECUTABLE}
    ARGS --cpp_out  ${PROJECT_SOURCE_DIR} -I ${PROJECT_SOURCE_DIR} ${PROTO_PATH}
    DEPENDS ${PROTO_PATH} ${PROTOBUF_PROTOC_EXECUTABLE}
    COMMENT "Running C++ protocol buffer compiler on ${PROTO}"
    VERBATIM )
endforeach()


include_directories("${PROJECT_SOURCE_DIR}")

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11 -O2")
set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS} -std=c++11 -ggdb3 -Wall -Wextra -Wswitch-enum -Wunused")

set(SENSEI_SRCS
        ${DEPS_SRCS}
        ${PB_GENERATED_SRCS}
        sensei/batch_training.cc
        sensei/config.cc
        sensei/data.cc
        sensei/data_reader.cc
        sensei/feature_exploration.cc
        sensei/feature_map.cc
        sensei/feature_pruning.cc
        sensei/feature_scoring.cc
        sensei/logger.cc
        sensei/read_data.cc
        sensei/score_rows.cc
        sensei/world.cc
        sensei/write_model.cc
)

set(SENSEI_LIBS
        glog
        gflags
        ${PROTOBUF_LIBRARIES}
)

add_library(batch_training ${SENSEI_SRCS})
add_executable(sensei_bin sensei/sensei.cc)
target_link_libraries(sensei_bin batch_training ${SENSEI_LIBS})

find_package(GTest)

if((NOT ${GTEST_FOUND}) AND (EXISTS gtest-1.7.0/CMakeLists.txt))
  message(setting)
  add_subdirectory(gtest-1.7.0)
  set(GTEST_FOUND TRUE)
  set(GTEST_BOTH_LIBRARIES gtest gtest_main)
  set(GTEST_INCLUDE_DIRS gtest-1.7.0/include)
endif()

if(${GTEST_FOUND})
  enable_testing()
  set(TEST_LIBS ${SENSEI_LIBS} ${GTEST_BOTH_LIBRARIES} pthread)
  include_directories(${GTEST_INCLUDE_DIRS})

  add_executable(regression_test sensei/regression_test.cc)
  target_link_libraries(regression_test batch_training ${TEST_LIBS})
  add_test(RegressionTest regression_test)
endif()
