add_library(igiggle-helper-lib STATIC
    data_manager.cpp
    data_manager.h
)

set(IGIGGLE_HELPER_INCLUDE_DIR "${IGIGGLE_SOURCE_DIR}/src" PARENT_SCOPE)

set_target_properties(igiggle-helper-lib PROPERTIES CXX_STANDARD 11)
