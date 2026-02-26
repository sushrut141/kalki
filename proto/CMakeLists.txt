set(PROTO_FILE ${CMAKE_CURRENT_SOURCE_DIR}/kalki.proto)
set(PROTO_OUT_DIR ${CMAKE_CURRENT_BINARY_DIR}/generated)

file(MAKE_DIRECTORY ${PROTO_OUT_DIR})

set(PROTO_SRCS
  ${PROTO_OUT_DIR}/kalki.pb.cc
  ${PROTO_OUT_DIR}/kalki.grpc.pb.cc
)
set(PROTO_HDRS
  ${PROTO_OUT_DIR}/kalki.pb.h
  ${PROTO_OUT_DIR}/kalki.grpc.pb.h
)

add_custom_command(
  OUTPUT ${PROTO_SRCS} ${PROTO_HDRS}
  COMMAND protobuf::protoc
  ARGS
    --proto_path=${CMAKE_CURRENT_SOURCE_DIR}
    --cpp_out=${PROTO_OUT_DIR}
    --grpc_out=${PROTO_OUT_DIR}
    --plugin=protoc-gen-grpc=$<TARGET_FILE:gRPC::grpc_cpp_plugin>
    ${PROTO_FILE}
  DEPENDS ${PROTO_FILE}
  VERBATIM
)

add_library(kalki_proto SHARED ${PROTO_SRCS})
add_library(kalki::proto ALIAS kalki_proto)

target_include_directories(kalki_proto
  PUBLIC
    ${PROTO_OUT_DIR}
)

target_link_libraries(kalki_proto
  PUBLIC
    protobuf::libprotobuf
    gRPC::grpc++
)
