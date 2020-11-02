set(GTEST_SHUFFLE 1)

file(GLOB tests *.cpp)
file(GLOB modules ${CMAKE_SOURCE_DIR}/app/*.cpp)
list(REMOVE_ITEM modules ${CMAKE_SOURCE_DIR}/app/main.cpp)

add_executable(unit_tests ${tests} ${modules})
target_include_directories(unit_tests PUBLIC ${CMAKE_SOURCE_DIR}/vendor/googletest/googletest/include)
target_link_libraries(unit_tests PUBLIC gtest ${OpenCV_LIBS})
