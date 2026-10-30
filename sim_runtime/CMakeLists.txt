cmake_minimum_required(VERSION 3.16)
project(usv_vla_sim_runtime LANGUAGES CXX)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_BUILD_RPATH "$ORIGIN")
set(CMAKE_INSTALL_RPATH "$ORIGIN")
find_package(gz-sim8 REQUIRED)
find_package(gz-common5 REQUIRED COMPONENTS graphics)
find_package(gz-math7 REQUIRED)
find_package(gz-transport13 REQUIRED)
find_package(gz-msgs10 REQUIRED)
find_package(gz-plugin2 REQUIRED COMPONENTS register)
find_package(gz-rendering8 REQUIRED)
find_package(gz-utils2 REQUIRED)
find_package(sdformat14 REQUIRED)
find_package(Eigen3 REQUIRED)

add_library(Waves SHARED src/Wavefield.cc)
target_link_libraries(Waves PUBLIC gz-sim8::core gz-common5::gz-common5
  gz-math7::gz-math7 gz-msgs10::gz-msgs10 Eigen3::Eigen)
add_library(PolyhedraBuoyancyDrag SHARED src/PolyhedraBuoyancyDrag.cc
  src/PolyhedronVolume.cc src/ShapeVolume.cc)
target_link_libraries(PolyhedraBuoyancyDrag PUBLIC Waves gz-sim8::core
  gz-common5::gz-common5 gz-math7::gz-math7 gz-transport13::gz-transport13
  gz-plugin2::register)
add_library(PublisherPlugin SHARED src/PublisherPlugin.cc)
target_link_libraries(PublisherPlugin PUBLIC gz-sim8::core gz-common5::gz-common5
  gz-transport13::gz-transport13 gz-plugin2::register)
add_library(WaveVisual SHARED src/WaveVisual.cc)
target_link_libraries(WaveVisual PUBLIC Waves gz-sim8::core gz-common5::gz-common5
  gz-transport13::gz-transport13 gz-plugin2::register
  gz-rendering8::gz-rendering8 gz-utils2::gz-utils2)
install(TARGETS Waves PolyhedraBuoyancyDrag PublisherPlugin WaveVisual
  LIBRARY DESTINATION lib)
