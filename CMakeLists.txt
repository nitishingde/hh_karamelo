cmake_minimum_required(VERSION 3.5)

project(hh_karamelo CXX)

set(CMAKE_CXX_STANDARD 20)

find_package(Git)
if(Git_FOUND)
  message("Git found: ${GIT_EXECUTABLE}")
endif()

# Download git repos via cmake instead of using submodules
include(FetchContent)
FetchContent_Declare(
    Eigen3
    GIT_REPOSITORY https://gitlab.com/libeigen/eigen.git
    GIT_TAG 3.4.0
    OVERRIDE_FIND_PACKAGE
)
FetchContent_MakeAvailable(Eigen3)

if(NOT Hedgehog_INCLUDE_DIR)
  # Hedgehog 3.0
  FetchContent_Declare(
    Hedgehog
    GIT_REPOSITORY https://github.com/usnistgov/hedgehog.git
    GIT_TAG        master
  )
  FetchContent_GetProperties(Hedgehog)
  if(NOT hedgehog_POPULATED)
    FetchContent_Populate(Hedgehog)
  endif(NOT hedgehog_POPULATED)

  # Need to set variable Hedgehog_INCLUDE_DIR to the downloaded hedgehog src code
  set(Hedgehog_INCLUDE_DIR "${hedgehog_SOURCE_DIR}" CACHE STRING "${hedgehog_SOURCE_DIR}" FORCE)
endif(NOT Hedgehog_INCLUDE_DIR)

# Test NewHedgehog library
set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} "${CMAKE_CURRENT_SOURCE_DIR}/cmake-modules")
find_package(Hedgehog REQUIRED)

cmake_policy(SET CMP0004 OLD)

if(WIN32)
  set(EIGEN3_INCLUDE_DIR "C:/Program Files/Eigen")
else()
  find_package(Eigen3 REQUIRED NO_MODULE)
endif()

find_package(MPI REQUIRED)
include_directories(${MPI_INCLUDE_PATH})
include_directories(${EIGEN3_INCLUDE_DIR})
include_directories(${PROJECT_SOURCE_DIR})

# the commit's SHA1, and whether the building workspace was dirty or not
execute_process(COMMAND
  "${GIT_EXECUTABLE}" describe --match=NeVeRmAtCh --always --abbrev=40 --dirty
  WORKING_DIRECTORY "${CMAKE_SOURCE_DIR}"
  OUTPUT_VARIABLE GIT_SHA1
  ERROR_QUIET OUTPUT_STRIP_TRAILING_WHITESPACE)

# the date of the commit
execute_process(COMMAND
  "${GIT_EXECUTABLE}" log -1 --format=%ad --date=local
  WORKING_DIRECTORY "${CMAKE_SOURCE_DIR}"
  OUTPUT_VARIABLE GIT_DATE
  ERROR_QUIET OUTPUT_STRIP_TRAILING_WHITESPACE)

# the subject of the commit
execute_process(COMMAND
  "${GIT_EXECUTABLE}" log -1 --format=%s
  WORKING_DIRECTORY "${CMAKE_SOURCE_DIR}"
  OUTPUT_VARIABLE GIT_COMMIT_SUBJECT
  ERROR_QUIET OUTPUT_STRIP_TRAILING_WHITESPACE)

# generate version.cpp
configure_file("${CMAKE_CURRENT_SOURCE_DIR}/src/version.cpp.in" "${CMAKE_CURRENT_SOURCE_DIR}/src/version.cpp" @ONLY)

# the commit's SHA1, and whether the building workspace was dirty or not
execute_process(COMMAND
  "${GIT_EXECUTABLE}" describe --match=NeVeRmAtCh --always --abbrev=40 --dirty
  WORKING_DIRECTORY "${CMAKE_SOURCE_DIR}"
  OUTPUT_VARIABLE GIT_SHA1
  ERROR_QUIET OUTPUT_STRIP_TRAILING_WHITESPACE)

# the date of the commit
execute_process(COMMAND
  "${GIT_EXECUTABLE}" log -1 --format=%ad --date=local
  WORKING_DIRECTORY "${CMAKE_SOURCE_DIR}"
  OUTPUT_VARIABLE GIT_DATE
  ERROR_QUIET OUTPUT_STRIP_TRAILING_WHITESPACE)

# the subject of the commit
execute_process(COMMAND
  "${GIT_EXECUTABLE}" log -1 --format=%s
  WORKING_DIRECTORY "${CMAKE_SOURCE_DIR}"
  OUTPUT_VARIABLE GIT_COMMIT_SUBJECT
  ERROR_QUIET OUTPUT_STRIP_TRAILING_WHITESPACE)

# generate version.cpp
configure_file("${CMAKE_CURRENT_SOURCE_DIR}/src/version.cpp.in" "${CMAKE_CURRENT_BINARY_DIR}/src/version.cpp" @ONLY)

file(GLOB MyCSources src/*.cpp)
list(REMOVE_ITEM MyCSources "${CMAKE_SOURCE_DIR}/src/hh3_taylorbar_demo.cpp")
add_executable(karamelo ${MyCSources})

#add_subdirectory(docs EXCLUDE_FROM_ALL)
add_subdirectory(third-party/gzstream)

set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} -DDEBUG")

target_compile_options(karamelo PRIVATE ${MPI_CXX_COMPILE_FLAGS})
if(!WIN32)
  target_compile_options(karamelo PRIVATE "-march=native")
  set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} -O3")
  set(CMAKE_CXX_FLAGS_PROFILING "${CMAKE_CXX_FLAGS_PROFILING} -O3 -g")
else()
  set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} -O2")
  set(CMAKE_CXX_FLAGS_PROFILING "${CMAKE_CXX_FLAGS_PROFILING} -O2 -g")
endif()

target_link_libraries(karamelo PUBLIC ${MPI_CXX_LIBRARIES} ${MPI_CXX_LINK_FLAGS} gzstream Eigen3::Eigen)

list(REMOVE_ITEM MyCSources "${CMAKE_SOURCE_DIR}/src/main.cpp")
add_executable(hh3_taylorbar_demo src/hh3_taylorbar_demo.cpp ${MyCSources})
target_include_directories(hh3_taylorbar_demo PUBLIC ${Hedgehog_INCLUDE_DIRS})
target_link_libraries(hh3_taylorbar_demo PUBLIC ${MPI_CXX_LIBRARIES} ${MPI_CXX_LINK_FLAGS} gzstream Eigen3::Eigen ${Hedgehog_LIBRARIES})
