set(TARGET kvshare-layout-roundtrip)

add_executable(${TARGET} kvshare-layout-roundtrip.cpp)
target_link_libraries(${TARGET} PRIVATE llama)
target_include_directories(${TARGET} PRIVATE ../../src)
install(TARGETS ${TARGET} RUNTIME)
