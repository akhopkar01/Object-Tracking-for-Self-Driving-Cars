file(GLOB src_files *.cpp)

add_executable(app ${src_files})
target_link_libraries(app ${OpenCV_LIBS})
