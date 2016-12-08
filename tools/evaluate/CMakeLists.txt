include_directories(
	"../../modules/post/include")

file(GLOB SOURCES
	"source/*.cc")

add_executable(tool_evaluate ${SOURCES})
target_link_libraries(tool_evaluate module_post)
set_target_properties(tool_evaluate PROPERTIES
    OUTPUT_NAME "tool_evaluate"
    ARCHIVE_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}"
    LIBRARY_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}"
    RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}" )
