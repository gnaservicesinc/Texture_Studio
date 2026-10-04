cmake_minimum_required(VERSION 3.24)
project(IPDE VERSION 0.1.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

find_package(Qt6 6.8 REQUIRED COMPONENTS Widgets Network)
qt_standard_project_setup()

set(IPDE_PYTHON_EXECUTABLE
    "/Library/Frameworks/Python.framework/Versions/3.14/bin/python3.14"
    CACHE FILEPATH "Python interpreter used by the IPDE GUI")

qt_add_executable(ipde_gui MACOSX_BUNDLE
    src/gui/main.cpp
)
qt_add_executable(raft_studio MACOSX_BUNDLE src/gui/trainer.cpp)
qt_add_executable(ipde_studio MACOSX_BUNDLE src/gui/studio.cpp)
target_link_libraries(ipde_studio PRIVATE Qt6::Widgets Qt6::Network)
target_compile_options(ipde_studio PRIVATE -Wall -Wextra -Wpedantic)
set_target_properties(ipde_studio PROPERTIES
    OUTPUT_NAME "IPDE Studio"
    MACOSX_BUNDLE_BUNDLE_NAME "IPDE Studio"
    MACOSX_BUNDLE_GUI_IDENTIFIER "org.ipde.studio"
    MACOSX_BUNDLE_BUNDLE_VERSION "${PROJECT_VERSION}"
    MACOSX_BUNDLE_SHORT_VERSION_STRING "${PROJECT_VERSION}"
)
target_link_libraries(raft_studio PRIVATE Qt6::Widgets Qt6::Network)
target_compile_options(raft_studio PRIVATE -Wall -Wextra -Wpedantic)
target_compile_definitions(raft_studio PRIVATE
    RAFT_STUDIO_SOURCE_SCRIPT="${CMAKE_CURRENT_SOURCE_DIR}/raft_studio.py"
    IPDE_PYTHON_EXECUTABLE="${IPDE_PYTHON_EXECUTABLE}"
)
set_target_properties(raft_studio PROPERTIES
    OUTPUT_NAME "RAFT Studio"
    MACOSX_BUNDLE_BUNDLE_NAME "RAFT Studio"
    MACOSX_BUNDLE_GUI_IDENTIFIER "org.ipde.raft-studio"
    MACOSX_BUNDLE_BUNDLE_VERSION "${PROJECT_VERSION}"
    MACOSX_BUNDLE_SHORT_VERSION_STRING "${PROJECT_VERSION}"
)

set(IPDE_PYTHON_SOURCES
    "${CMAKE_CURRENT_SOURCE_DIR}/ipde_extract.py"
    "${CMAKE_CURRENT_SOURCE_DIR}/src/ipde/__init__.py"
    "${CMAKE_CURRENT_SOURCE_DIR}/src/ipde/__main__.py"
    "${CMAKE_CURRENT_SOURCE_DIR}/src/ipde/apple_imageio.py"
    "${CMAKE_CURRENT_SOURCE_DIR}/src/ipde/cli.py"
    "${CMAKE_CURRENT_SOURCE_DIR}/src/ipde/extractor.py"
    "${CMAKE_CURRENT_SOURCE_DIR}/src/ipde/formats.py"
    "${CMAKE_CURRENT_SOURCE_DIR}/src/ipde/libheif_aux.py"
    "${CMAKE_CURRENT_SOURCE_DIR}/src/ipde/learned_depth.py"
    "${CMAKE_CURRENT_SOURCE_DIR}/src/ipde/registration.py"
    "${CMAKE_CURRENT_SOURCE_DIR}/src/ipde/spatial.py"
)

target_link_libraries(ipde_gui PRIVATE Qt6::Widgets Qt6::Network)
target_compile_options(ipde_gui PRIVATE -Wall -Wextra -Wpedantic)

if(APPLE)
    # This workstation also has a newer third-party libc++ in /usr/local/lib.
    # Apple Clang injects that search path before the SDK and would otherwise
    # produce an @rpath/libc++.1.dylib dependency that cannot launch. Link the
    # platform libc++ stub explicitly so the app uses macOS's dyld shared cache.
    execute_process(
        COMMAND xcrun --sdk macosx --show-sdk-path
        OUTPUT_VARIABLE IPDE_MACOS_SDK
        OUTPUT_STRIP_TRAILING_WHITESPACE
        RESULT_VARIABLE IPDE_XCRUN_RESULT
    )
    if(IPDE_XCRUN_RESULT EQUAL 0 AND EXISTS "${IPDE_MACOS_SDK}/usr/lib/libc++.tbd")
        target_link_options(ipde_gui PRIVATE -nostdlib++)
        target_link_libraries(ipde_gui PRIVATE "${IPDE_MACOS_SDK}/usr/lib/libc++.tbd")
        target_link_options(raft_studio PRIVATE -nostdlib++)
        target_link_libraries(raft_studio PRIVATE "${IPDE_MACOS_SDK}/usr/lib/libc++.tbd")
        target_link_options(ipde_studio PRIVATE -nostdlib++)
        target_link_libraries(ipde_studio PRIVATE "${IPDE_MACOS_SDK}/usr/lib/libc++.tbd")
    endif()
endif()
target_compile_definitions(ipde_gui PRIVATE
    IPDE_SOURCE_SCRIPT="${CMAKE_CURRENT_SOURCE_DIR}/ipde_extract.py"
    IPDE_PYTHON_EXECUTABLE="${IPDE_PYTHON_EXECUTABLE}"
)
set_property(TARGET ipde_gui APPEND PROPERTY LINK_DEPENDS "${IPDE_PYTHON_SOURCES}")
set(RAFT_STUDIO_PYTHON_SOURCES ${IPDE_PYTHON_SOURCES})
list(REMOVE_ITEM RAFT_STUDIO_PYTHON_SOURCES "${CMAKE_CURRENT_SOURCE_DIR}/ipde_extract.py")
list(APPEND RAFT_STUDIO_PYTHON_SOURCES
    "${CMAKE_CURRENT_SOURCE_DIR}/src/ipde/trainer_cli.py"
    "${CMAKE_CURRENT_SOURCE_DIR}/src/ipde/dataset.py"
    "${CMAKE_CURRENT_SOURCE_DIR}/src/ipde/dataset_review.py"
    "${CMAKE_CURRENT_SOURCE_DIR}/src/ipde/training.py"
    "${CMAKE_CURRENT_SOURCE_DIR}/src/ipde/pseudo_calibration.py"
    "${CMAKE_CURRENT_SOURCE_DIR}/src/ipde/array_storage.py"
    "${CMAKE_CURRENT_SOURCE_DIR}/src/ipde/dataset_collection.py"
    "${CMAKE_CURRENT_SOURCE_DIR}/src/ipde/spatial_scan.py"
    "${CMAKE_CURRENT_SOURCE_DIR}/src/ipde/model_comparison.py"
    "${CMAKE_CURRENT_SOURCE_DIR}/src/ipde/huggingface_datasets.py"
)
set_property(TARGET raft_studio APPEND PROPERTY LINK_DEPENDS
    "${RAFT_STUDIO_PYTHON_SOURCES}" "${CMAKE_CURRENT_SOURCE_DIR}/raft_studio.py")

set_target_properties(ipde_gui PROPERTIES
    OUTPUT_NAME "IPDE"
    MACOSX_BUNDLE_BUNDLE_NAME "IPDE"
    MACOSX_BUNDLE_GUI_IDENTIFIER "org.openai.ipde"
    MACOSX_BUNDLE_BUNDLE_VERSION "${PROJECT_VERSION}"
    MACOSX_BUNDLE_SHORT_VERSION_STRING "${PROJECT_VERSION}"
)

if(APPLE)
    set(resource_root "$<TARGET_BUNDLE_DIR:ipde_gui>/Contents/Resources")
    add_custom_command(TARGET ipde_gui POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E remove_directory "${resource_root}/src/ipde"
        COMMAND ${CMAKE_COMMAND} -E make_directory "${resource_root}/src/ipde"
        COMMAND ${CMAKE_COMMAND} -E copy_if_different
            "${CMAKE_CURRENT_SOURCE_DIR}/ipde_extract.py" "${resource_root}/ipde_extract.py"
        COMMAND ${CMAKE_COMMAND} -E copy_if_different
            "${CMAKE_CURRENT_SOURCE_DIR}/src/ipde/__init__.py" "${resource_root}/src/ipde/__init__.py"
        COMMAND ${CMAKE_COMMAND} -E copy_if_different
            "${CMAKE_CURRENT_SOURCE_DIR}/src/ipde/__main__.py" "${resource_root}/src/ipde/__main__.py"
        COMMAND ${CMAKE_COMMAND} -E copy_if_different
            "${CMAKE_CURRENT_SOURCE_DIR}/src/ipde/apple_imageio.py" "${resource_root}/src/ipde/apple_imageio.py"
        COMMAND ${CMAKE_COMMAND} -E copy_if_different
            "${CMAKE_CURRENT_SOURCE_DIR}/src/ipde/cli.py" "${resource_root}/src/ipde/cli.py"
        COMMAND ${CMAKE_COMMAND} -E copy_if_different
            "${CMAKE_CURRENT_SOURCE_DIR}/src/ipde/extractor.py" "${resource_root}/src/ipde/extractor.py"
        COMMAND ${CMAKE_COMMAND} -E copy_if_different
            "${CMAKE_CURRENT_SOURCE_DIR}/src/ipde/formats.py" "${resource_root}/src/ipde/formats.py"
        COMMAND ${CMAKE_COMMAND} -E copy_if_different
            "${CMAKE_CURRENT_SOURCE_DIR}/src/ipde/libheif_aux.py" "${resource_root}/src/ipde/libheif_aux.py"
        COMMAND ${CMAKE_COMMAND} -E copy_if_different
            "${CMAKE_CURRENT_SOURCE_DIR}/src/ipde/spatial.py" "${resource_root}/src/ipde/spatial.py"
        COMMAND ${CMAKE_COMMAND} -E copy_if_different
            "${CMAKE_CURRENT_SOURCE_DIR}/src/ipde/learned_depth.py" "${resource_root}/src/ipde/learned_depth.py"
        COMMAND ${CMAKE_COMMAND} -E copy_if_different
            "${CMAKE_CURRENT_SOURCE_DIR}/src/ipde/registration.py" "${resource_root}/src/ipde/registration.py"
        COMMENT "Embedding the Python extraction engine in IPDE.app"
    )
endif()

if(APPLE)
    add_dependencies(ipde_studio ipde_gui raft_studio)
    set(studio_apps "$<TARGET_BUNDLE_DIR:ipde_studio>/Contents/Applications")
    add_custom_target(ipde_studio_bundle ALL
        COMMAND ${CMAKE_COMMAND} -E make_directory "${studio_apps}"
        COMMAND ${CMAKE_COMMAND} -E copy_directory "$<TARGET_BUNDLE_DIR:ipde_gui>" "${studio_apps}/IPDE.app"
        COMMAND ${CMAKE_COMMAND} -E copy_directory "$<TARGET_BUNDLE_DIR:raft_studio>" "${studio_apps}/RAFT Studio.app"
        DEPENDS ipde_studio ipde_gui raft_studio
        COMMENT "Packing IPDE subapps into IPDE Studio"
    )
endif()

if(APPLE)
    set(trainer_resources "$<TARGET_BUNDLE_DIR:raft_studio>/Contents/Resources")
    add_custom_command(TARGET raft_studio POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E remove_directory "${trainer_resources}/src/ipde"
        COMMAND ${CMAKE_COMMAND} -E make_directory "${trainer_resources}/src/ipde"
        COMMAND ${CMAKE_COMMAND} -E copy_if_different "${CMAKE_CURRENT_SOURCE_DIR}/raft_studio.py" "${trainer_resources}/raft_studio.py"
        COMMAND ${CMAKE_COMMAND} -E copy_if_different ${RAFT_STUDIO_PYTHON_SOURCES} "${trainer_resources}/src/ipde"
        COMMENT "Embedding the separate RAFT Studio workflow"
    )
endif()
