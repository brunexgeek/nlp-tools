include_directories(
	"include")

file(GLOB SOURCES
	"source/*.cc")

add_library(module_post STATIC ${SOURCES})
set_target_properties(module_post PROPERTIES
    OUTPUT_NAME "post"
    ARCHIVE_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}"
    LIBRARY_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}"
    RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}"
    PREFIX "lib" )
