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
qt_add_executable(dataset_studio MACOSX_BUNDLE src/gui/trainer.cpp)
target_link_libraries(dataset_studio PRIVATE Qt6::Widgets Qt6::Network)
target_compile_options(dataset_studio PRIVATE -Wall -Wextra -Wpedantic)
target_compile_definitions(dataset_studio PRIVATE IPDE_DATASET_STUDIO=1
    RAFT_STUDIO_SOURCE_SCRIPT="${CMAKE_CURRENT_SOURCE_DIR}/raft_studio.py"
    IPDE_PYTHON_EXECUTABLE="${IPDE_PYTHON_EXECUTABLE}")
set_target_properties(dataset_studio PROPERTIES OUTPUT_NAME "Dataset Studio"
    MACOSX_BUNDLE_BUNDLE_NAME "Dataset Studio" MACOSX_BUNDLE_GUI_IDENTIFIER "org.ipde.dataset-studio"
    MACOSX_BUNDLE_BUNDLE_VERSION "${PROJECT_VERSION}" MACOSX_BUNDLE_SHORT_VERSION_STRING "${PROJECT_VERSION}")
qt_add_executable(ipde_studio MACOSX_BUNDLE src/gui/studio.cpp)
qt_add_executable(photo_studio MACOSX_BUNDLE src/gui/media.cpp)
qt_add_executable(raw_studio MACOSX_BUNDLE src/gui/media.cpp)
target_compile_definitions(photo_studio PRIVATE PHOTO_STUDIO=1)
foreach(media_target IN ITEMS photo_studio raw_studio)
    target_link_libraries(${media_target} PRIVATE Qt6::Widgets Qt6::Network)
    target_compile_options(${media_target} PRIVATE -Wall -Wextra -Wpedantic)
    target_compile_definitions(${media_target} PRIVATE
        MEDIA_STUDIO_SOURCE_SCRIPT="${CMAKE_CURRENT_SOURCE_DIR}/media_studio.py"
        IPDE_PYTHON_EXECUTABLE="${IPDE_PYTHON_EXECUTABLE}")
endforeach()
set_target_properties(photo_studio PROPERTIES OUTPUT_NAME "Photo Studio"
    MACOSX_BUNDLE_BUNDLE_NAME "Photo Studio" MACOSX_BUNDLE_GUI_IDENTIFIER "org.ipde.photo-studio"
    MACOSX_BUNDLE_BUNDLE_VERSION "${PROJECT_VERSION}" MACOSX_BUNDLE_SHORT_VERSION_STRING "${PROJECT_VERSION}")
set_target_properties(raw_studio PROPERTIES OUTPUT_NAME "Raw Studio"
    MACOSX_BUNDLE_BUNDLE_NAME "Raw Studio" MACOSX_BUNDLE_GUI_IDENTIFIER "org.ipde.raw-studio"
    MACOSX_BUNDLE_BUNDLE_VERSION "${PROJECT_VERSION}" MACOSX_BUNDLE_SHORT_VERSION_STRING "${PROJECT_VERSION}")
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
        foreach(media_target IN ITEMS photo_studio raw_studio dataset_studio)
            target_link_options(${media_target} PRIVATE -nostdlib++)
            target_link_libraries(${media_target} PRIVATE "${IPDE_MACOS_SDK}/usr/lib/libc++.tbd")
        endforeach()
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
    "${CMAKE_CURRENT_SOURCE_DIR}/src/ipde/dataset_edit.py"
    "${CMAKE_CURRENT_SOURCE_DIR}/src/ipde/training.py"
    "${CMAKE_CURRENT_SOURCE_DIR}/src/ipde/concurrency.py"
    "${CMAKE_CURRENT_SOURCE_DIR}/src/ipde/resource_lock.py"
    "${CMAKE_CURRENT_SOURCE_DIR}/src/ipde/workspace_cleanup.py"
    "${CMAKE_CURRENT_SOURCE_DIR}/src/ipde/pseudo_calibration.py"
    "${CMAKE_CURRENT_SOURCE_DIR}/src/ipde/array_storage.py"
    "${CMAKE_CURRENT_SOURCE_DIR}/src/ipde/dataset_collection.py"
    "${CMAKE_CURRENT_SOURCE_DIR}/src/ipde/spatial_scan.py"
    "${CMAKE_CURRENT_SOURCE_DIR}/src/ipde/model_comparison.py"
    "${CMAKE_CURRENT_SOURCE_DIR}/src/ipde/huggingface_datasets.py"
)
set_property(TARGET raft_studio APPEND PROPERTY LINK_DEPENDS
    "${RAFT_STUDIO_PYTHON_SOURCES}" "${CMAKE_CURRENT_SOURCE_DIR}/raft_studio.py")
set_property(TARGET dataset_studio APPEND PROPERTY LINK_DEPENDS
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
    add_dependencies(ipde_studio ipde_gui raft_studio dataset_studio photo_studio raw_studio)
    set(studio_apps "$<TARGET_BUNDLE_DIR:ipde_studio>/Contents/Applications")
    add_custom_target(ipde_studio_bundle ALL
        COMMAND ${CMAKE_COMMAND} -E make_directory "${studio_apps}"
        COMMAND ${CMAKE_COMMAND} -E copy_directory "$<TARGET_BUNDLE_DIR:ipde_gui>" "${studio_apps}/IPDE.app"
        COMMAND ${CMAKE_COMMAND} -E copy_directory "$<TARGET_BUNDLE_DIR:raft_studio>" "${studio_apps}/RAFT Studio.app"
        COMMAND ${CMAKE_COMMAND} -E copy_directory "$<TARGET_BUNDLE_DIR:dataset_studio>" "${studio_apps}/Dataset Studio.app"
        COMMAND ${CMAKE_COMMAND} -E copy_directory "$<TARGET_BUNDLE_DIR:photo_studio>" "${studio_apps}/Photo Studio.app"
        COMMAND ${CMAKE_COMMAND} -E copy_directory "$<TARGET_BUNDLE_DIR:raw_studio>" "${studio_apps}/Raw Studio.app"
        DEPENDS ipde_studio ipde_gui raft_studio dataset_studio photo_studio raw_studio
        COMMENT "Packing IPDE subapps into IPDE Studio"
    )
endif()

set(MEDIA_STUDIO_PYTHON_SOURCES ${RAFT_STUDIO_PYTHON_SOURCES}
    "${CMAKE_CURRENT_SOURCE_DIR}/src/ipde/photo_workflow.py"
    "${CMAKE_CURRENT_SOURCE_DIR}/src/ipde/raw_workflow.py"
    "${CMAKE_CURRENT_SOURCE_DIR}/src/ipde/heif_privacy.py"
    "${CMAKE_CURRENT_SOURCE_DIR}/src/ipde/media_cli.py")
foreach(media_target IN ITEMS photo_studio raw_studio)
    set_property(TARGET ${media_target} APPEND PROPERTY LINK_DEPENDS
        "${MEDIA_STUDIO_PYTHON_SOURCES}" "${CMAKE_CURRENT_SOURCE_DIR}/media_studio.py")
    if(APPLE)
        set(media_resources "$<TARGET_BUNDLE_DIR:${media_target}>/Contents/Resources")
        add_custom_command(TARGET ${media_target} POST_BUILD
            COMMAND ${CMAKE_COMMAND} -E remove_directory "${media_resources}/src/ipde"
            COMMAND ${CMAKE_COMMAND} -E make_directory "${media_resources}/src/ipde"
            COMMAND ${CMAKE_COMMAND} -E copy_if_different "${CMAKE_CURRENT_SOURCE_DIR}/media_studio.py" "${media_resources}/media_studio.py"
            COMMAND ${CMAKE_COMMAND} -E copy_if_different ${MEDIA_STUDIO_PYTHON_SOURCES} "${media_resources}/src/ipde"
            COMMENT "Embedding precision photo and RAW workflows")
    endif()
endforeach()

if(APPLE)
    # Draw the same vector symbols for Qt controls and the native macOS bundles.
    qt_add_executable(ipde_icon_builder src/gui/icon_builder.cpp)
    target_link_libraries(ipde_icon_builder PRIVATE Qt6::Widgets)
    if(IPDE_XCRUN_RESULT EQUAL 0)
        target_link_options(ipde_icon_builder PRIVATE -nostdlib++)
        target_link_libraries(ipde_icon_builder PRIVATE "${IPDE_MACOS_SDK}/usr/lib/libc++.tbd")
    endif()
    set(icon_dir "${CMAKE_CURRENT_BINARY_DIR}/generated-icons")
    set(icon_outputs)
    foreach(role IN ITEMS studio extractor datasets trainer photo raw)
        list(APPEND icon_outputs "${icon_dir}/${role}.icns")
    endforeach()
    add_custom_command(OUTPUT ${icon_outputs}
        COMMAND ${CMAKE_COMMAND} -E env QT_QPA_PLATFORM=offscreen "$<TARGET_FILE:ipde_icon_builder>" "${icon_dir}"
        DEPENDS ipde_icon_builder src/gui/studio_icons.h COMMENT "Creating the IPDE icon family")
    add_custom_target(ipde_icon_assets DEPENDS ${icon_outputs})
    foreach(pair IN ITEMS "ipde_gui:extractor" "raft_studio:trainer" "dataset_studio:datasets" "ipde_studio:studio" "photo_studio:photo" "raw_studio:raw")
        string(REPLACE ":" ";" parts "${pair}")
        list(GET parts 0 app_target)
        list(GET parts 1 role)
        set(icon_file "${icon_dir}/${role}.icns")
        set_source_files_properties("${icon_file}" PROPERTIES GENERATED TRUE MACOSX_PACKAGE_LOCATION Resources)
        target_sources(${app_target} PRIVATE "${icon_file}")
        set_target_properties(${app_target} PROPERTIES MACOSX_BUNDLE_ICON_FILE "${role}.icns")
        add_dependencies(${app_target} ipde_icon_assets)
    endforeach()

    set(media_bridge "${CMAKE_CURRENT_BINARY_DIR}/media-bridge")
    add_custom_command(OUTPUT "${media_bridge}"
        COMMAND xcrun swiftc "${CMAKE_CURRENT_SOURCE_DIR}/native/media_bridge.swift" -o "${media_bridge}"
            -framework Foundation -framework ImageIO -framework Vision -framework CoreImage -framework CoreGraphics
        DEPENDS native/media_bridge.swift COMMENT "Compiling native HEIC and RAW support")
    add_custom_target(ipde_media_bridge DEPENDS "${media_bridge}")
    foreach(media_target IN ITEMS photo_studio raw_studio)
        add_dependencies(${media_target} ipde_media_bridge)
        set_source_files_properties("${media_bridge}" PROPERTIES GENERATED TRUE MACOSX_PACKAGE_LOCATION Resources)
        target_sources(${media_target} PRIVATE "${media_bridge}")
    endforeach()
endif()

include(CTest)
if(BUILD_TESTING)
    qt_add_executable(project_session_regression "${CMAKE_CURRENT_SOURCE_DIR}/verification/project_session_regression.cpp")
    target_include_directories(project_session_regression PRIVATE src/gui)
    target_link_libraries(project_session_regression PRIVATE Qt6::Widgets Qt6::Network)
    if(APPLE AND IPDE_XCRUN_RESULT EQUAL 0)
        target_link_options(project_session_regression PRIVATE -nostdlib++)
        target_link_libraries(project_session_regression PRIVATE "${IPDE_MACOS_SDK}/usr/lib/libc++.tbd")
    endif()
    add_test(NAME project_session_lifetime COMMAND project_session_regression)
    set_tests_properties(project_session_lifetime PROPERTIES ENVIRONMENT "QT_QPA_PLATFORM=offscreen")

    foreach(regression IN ITEMS window_layout_regression studio_preview_regression trainer_workflow_regression dataset_management_regression studio_routing_regression)
        qt_add_executable(${regression} "${CMAKE_CURRENT_SOURCE_DIR}/verification/${regression}.cpp")
        target_include_directories(${regression} PRIVATE src/gui)
        target_link_libraries(${regression} PRIVATE Qt6::Widgets Qt6::Network)
        target_compile_definitions(${regression} PRIVATE
            RAFT_STUDIO_SOURCE_SCRIPT="${CMAKE_CURRENT_SOURCE_DIR}/raft_studio.py"
            IPDE_PYTHON_EXECUTABLE="${IPDE_PYTHON_EXECUTABLE}")
        if(APPLE AND IPDE_XCRUN_RESULT EQUAL 0)
            target_link_options(${regression} PRIVATE -nostdlib++)
            target_link_libraries(${regression} PRIVATE "${IPDE_MACOS_SDK}/usr/lib/libc++.tbd")
        endif()
        add_test(NAME ${regression} COMMAND ${regression})
        set_tests_properties(${regression} PROPERTIES ENVIRONMENT "QT_QPA_PLATFORM=offscreen")
    endforeach()
endif()

if(APPLE)
    foreach(trainer_target IN ITEMS raft_studio dataset_studio)
    set(trainer_resources "$<TARGET_BUNDLE_DIR:${trainer_target}>/Contents/Resources")
    add_custom_command(TARGET ${trainer_target} POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E remove_directory "${trainer_resources}/src/ipde"
        COMMAND ${CMAKE_COMMAND} -E make_directory "${trainer_resources}/src/ipde"
        COMMAND ${CMAKE_COMMAND} -E copy_if_different "${CMAKE_CURRENT_SOURCE_DIR}/raft_studio.py" "${trainer_resources}/raft_studio.py"
        COMMAND ${CMAKE_COMMAND} -E copy_if_different ${RAFT_STUDIO_PYTHON_SOURCES} "${trainer_resources}/src/ipde"
        COMMENT "Embedding the separate RAFT Studio workflow"
    )
    endforeach()
endif()
