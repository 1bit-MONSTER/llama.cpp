set(TARGET kvshare-adopt-check)

add_executable(${TARGET} kvshare-adopt-check.cpp)
target_link_libraries(${TARGET} PRIVATE llama)
target_include_directories(${TARGET} PRIVATE ../../src)
install(TARGETS ${TARGET} RUNTIME)
