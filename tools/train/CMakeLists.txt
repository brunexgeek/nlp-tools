include_directories(
	"../../modules/post/include")

file(GLOB SOURCES
	"source/*.cc")

add_executable(tool_train ${SOURCES})
target_link_libraries(tool_train module_post)
set_target_properties(tool_train PROPERTIES
    OUTPUT_NAME "tool_train"
    ARCHIVE_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}"
    LIBRARY_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}"
    RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}" )
