set(TARGET kvshare-tensor-map)

add_executable(${TARGET} kvshare-tensor-map.cpp)
target_link_libraries(${TARGET} PRIVATE llama)
target_include_directories(${TARGET} PRIVATE ../../src)
install(TARGETS ${TARGET} RUNTIME)
