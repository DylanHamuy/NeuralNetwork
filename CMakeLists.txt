cmake_minimum_required(VERSION 3.13)  # CMake version check
    project(NeuralNetwork)
    include(CTest)
    enable_testing()
    set(CMAKE_CXX_STANDARD 14)

    add_executable(NeuralNetwork src/Main.cpp src/NeuralNetwork.cpp)
    include_directories(${CMAKE_CURRENT_SOURCE_DIR}/include)
    include_directories(${CMAKE_CURRENT_SOURCE_DIR}/lib/SFML-2.6.1-macOS-clang-arm64/include)
    set(CPACK_PROJECT_NAME ${PROJECT_NAME})
    include(CPack)



