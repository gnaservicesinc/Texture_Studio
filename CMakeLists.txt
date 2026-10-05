cmake_minimum_required(VERSION 3.24)
project(IPDE VERSION 0.9.3 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

find_package(Qt6 6.8 REQUIRED COMPONENTS Widgets Network)
if(APPLE)
    include(cmake/MacOSDeployment.cmake)
    ipde_select_macos_deployment(IPDE_SELECTED_MACOS_TARGET)
    set(CMAKE_OSX_DEPLOYMENT_TARGET "${IPDE_SELECTED_MACOS_TARGET}"
        CACHE STRING "Minimum supported macOS version; empty selects the Qt/runtime minimum" FORCE)
    message(STATUS "IPDE macOS deployment target: ${CMAKE_OSX_DEPLOYMENT_TARGET} (Qt ${Qt6_VERSION})")
    execute_process(COMMAND xcrun --sdk macosx --show-sdk-version
        OUTPUT_VARIABLE IPDE_MACOS_SDK_VERSION OUTPUT_STRIP_TRAILING_WHITESPACE
        RESULT_VARIABLE IPDE_MACOS_SDK_RESULT ERROR_VARIABLE IPDE_MACOS_SDK_ERROR)
    set(IPDE_REQUIRED_MACOS_SDK "${CMAKE_OSX_DEPLOYMENT_TARGET}")
    if(QT_SUPPORTED_MIN_MACOS_SDK_VERSION VERSION_GREATER IPDE_REQUIRED_MACOS_SDK)
        set(IPDE_REQUIRED_MACOS_SDK "${QT_SUPPORTED_MIN_MACOS_SDK_VERSION}")
    endif()
    if(NOT IPDE_MACOS_SDK_RESULT EQUAL 0 OR IPDE_MACOS_SDK_VERSION VERSION_LESS IPDE_REQUIRED_MACOS_SDK)
        message(FATAL_ERROR "Qt ${Qt6_VERSION} and target ${CMAKE_OSX_DEPLOYMENT_TARGET} require macOS SDK ${IPDE_REQUIRED_MACOS_SDK} or newer; detected '${IPDE_MACOS_SDK_VERSION}'. ${IPDE_MACOS_SDK_ERROR}")
    endif()
endif()
qt_standard_project_setup()

# Development formats may change until 1.0.0. This same version is used in
# Python, About windows, native bundle metadata and tag release checks.
set(IPDE_PRERELEASE 0)
if(PROJECT_VERSION VERSION_LESS "1.0.0")
    set(IPDE_PRERELEASE 1)
endif()
# Preserve the build's configure date across automatic CMake regeneration.
# A wall-clock value generated on every configure forced every GUI to recompile.
if(NOT IPDE_BUILD_DATE)
    string(TIMESTAMP IPDE_BUILD_DATE "%Y-%m-%dT%H:%M:%SZ" UTC)
    set(IPDE_BUILD_DATE "${IPDE_BUILD_DATE}" CACHE STRING "Date this build directory was configured")
endif()
execute_process(COMMAND git rev-parse --short=12 HEAD
    WORKING_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}"
    OUTPUT_VARIABLE IPDE_BUILD_REVISION OUTPUT_STRIP_TRAILING_WHITESPACE ERROR_QUIET)
if(NOT IPDE_BUILD_REVISION)
    set(IPDE_BUILD_REVISION "unknown")
endif()
execute_process(COMMAND git diff-index --quiet HEAD -- . ":(exclude)build" ":(exclude)out"
    WORKING_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}" RESULT_VARIABLE IPDE_SOURCE_DIRTY ERROR_QUIET)
if(IPDE_SOURCE_DIRTY EQUAL 1)
    string(APPEND IPDE_BUILD_REVISION "-dirty")
endif()
string(TOUPPER "${CMAKE_BUILD_TYPE}" IPDE_BUILD_TYPE_UPPER)
set(IPDE_BUILD_FLAGS "${CMAKE_CXX_FLAGS} ${CMAKE_CXX_FLAGS_${IPDE_BUILD_TYPE_UPPER}}; C++${CMAKE_CXX_STANDARD}; architectures=${CMAKE_OSX_ARCHITECTURES}; deployment=${CMAKE_OSX_DEPLOYMENT_TARGET}; ad-hoc signed; not notarized")
string(REPLACE "\\" "\\\\" IPDE_BUILD_FLAGS "${IPDE_BUILD_FLAGS}")
string(REPLACE "\"" "\\\"" IPDE_BUILD_FLAGS "${IPDE_BUILD_FLAGS}")
configure_file(cmake/ipde_build_info.h.in generated/ipde_build_info.h @ONLY)
include_directories("${CMAKE_CURRENT_BINARY_DIR}/generated")
add_compile_definitions(IPDE_DOCS_DIR="${CMAKE_CURRENT_SOURCE_DIR}/docs")

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
    "${CMAKE_CURRENT_SOURCE_DIR}/src/ipde/display_student.py"
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
set(RAFT_STUDIO_PYTHON_SOURCES ${IPDE_PYTHON_SOURCES})
list(REMOVE_ITEM RAFT_STUDIO_PYTHON_SOURCES "${CMAKE_CURRENT_SOURCE_DIR}/ipde_extract.py")
list(APPEND RAFT_STUDIO_PYTHON_SOURCES
    "${CMAKE_CURRENT_SOURCE_DIR}/src/ipde/trainer_cli.py"
    "${CMAKE_CURRENT_SOURCE_DIR}/src/ipde/dataset.py"
    "${CMAKE_CURRENT_SOURCE_DIR}/src/ipde/dataset_review.py"
    "${CMAKE_CURRENT_SOURCE_DIR}/src/ipde/dataset_edit.py"
    "${CMAKE_CURRENT_SOURCE_DIR}/src/ipde/dataset_teachers.py"
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

set_target_properties(ipde_gui PROPERTIES
    OUTPUT_NAME "IPDE"
    MACOSX_BUNDLE_BUNDLE_NAME "IPDE"
    MACOSX_BUNDLE_GUI_IDENTIFIER "org.openai.ipde"
    MACOSX_BUNDLE_BUNDLE_VERSION "${PROJECT_VERSION}"
    MACOSX_BUNDLE_SHORT_VERSION_STRING "${PROJECT_VERSION}"
)

set(MEDIA_STUDIO_PYTHON_SOURCES ${RAFT_STUDIO_PYTHON_SOURCES}
    "${CMAKE_CURRENT_SOURCE_DIR}/src/ipde/photo_workflow.py"
    "${CMAKE_CURRENT_SOURCE_DIR}/src/ipde/raw_workflow.py"
    "${CMAKE_CURRENT_SOURCE_DIR}/src/ipde/heif_privacy.py"
    "${CMAKE_CURRENT_SOURCE_DIR}/src/ipde/media_cli.py")

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
    if(CMAKE_OSX_ARCHITECTURES)
        list(GET CMAKE_OSX_ARCHITECTURES 0 IPDE_SWIFT_ARCH)
    else()
        set(IPDE_SWIFT_ARCH "${CMAKE_SYSTEM_PROCESSOR}")
    endif()
    add_custom_command(OUTPUT "${media_bridge}"
        COMMAND xcrun swiftc "${CMAKE_CURRENT_SOURCE_DIR}/native/media_bridge.swift" -o "${media_bridge}"
            -target "${IPDE_SWIFT_ARCH}-apple-macosx${CMAKE_OSX_DEPLOYMENT_TARGET}"
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

    foreach(regression IN ITEMS window_layout_regression studio_preview_regression trainer_workflow_regression dataset_management_regression studio_routing_regression help_server_regression extractor_model_regression)
        qt_add_executable(${regression} "${CMAKE_CURRENT_SOURCE_DIR}/verification/${regression}.cpp")
        target_include_directories(${regression} PRIVATE src/gui)
        target_link_libraries(${regression} PRIVATE Qt6::Widgets Qt6::Network)
        target_compile_definitions(${regression} PRIVATE
            IPDE_SOURCE_SCRIPT="${CMAKE_CURRENT_SOURCE_DIR}/ipde_extract.py"
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

# Every native app carries the Qt libraries/plugins it uses, so replacing the
# development Qt installation cannot prevent an existing bundle from opening.
if(APPLE)
    if(TARGET Qt6::macdeployqt)
        set(IPDE_MACDEPLOYQT "$<TARGET_FILE:Qt6::macdeployqt>")
    else()
        find_program(IPDE_MACDEPLOYQT NAMES macdeployqt
            HINTS "${Qt6_DIR}/../../../bin" REQUIRED)
    endif()
    # Qt's deployer selects Cocoa for macOS. Keep offscreen as well for the
    # suite's noninteractive startup checks, previews and release validation.
    find_file(IPDE_QT_OFFSCREEN NAMES libqoffscreen.dylib
        HINTS "${Qt6_DIR}/../../../plugins/platforms" REQUIRED)
    file(GLOB_RECURSE IPDE_DOCUMENTATION CONFIGURE_DEPENDS "${CMAKE_CURRENT_SOURCE_DIR}/docs/*")
    file(GLOB IPDE_BACKEND_MODULES CONFIGURE_DEPENDS "${CMAKE_CURRENT_SOURCE_DIR}/src/ipde/*.py")
    # A removed help file disappears from DEPENDS after regeneration. Track
    # the inventory too so resource copies remove obsolete files in that case.
    file(GENERATE OUTPUT "${CMAKE_CURRENT_BINARY_DIR}/generated/resource-inventory.txt"
        CONTENT "${IPDE_DOCUMENTATION}\n${IPDE_BACKEND_MODULES}\n")
    set(IPDE_APP_RESOURCE_STAMPS)
    set(IPDE_APP_DEPLOY_STAMPS)
    foreach(app_target IN ITEMS ipde_gui raft_studio dataset_studio ipde_studio photo_studio raw_studio)
        set_target_properties(${app_target} PROPERTIES
            MACOSX_BUNDLE_INFO_PLIST "${CMAKE_CURRENT_SOURCE_DIR}/cmake/Info.plist.in")
        target_sources(${app_target} PRIVATE src/gui/python_runtime.h)
        set(app_resources "$<TARGET_BUNDLE_DIR:${app_target}>/Contents/Resources")
        set(resource_stamp "${CMAKE_CURRENT_BINARY_DIR}/stamps/${app_target}-resources.stamp")
        set(deploy_stamp "${CMAKE_CURRENT_BINARY_DIR}/stamps/${app_target}-qt.stamp")
        set(entrypoints)
        if(app_target STREQUAL "ipde_gui")
            list(APPEND entrypoints "${CMAKE_CURRENT_SOURCE_DIR}/ipde_extract.py")
        elseif(app_target STREQUAL "raft_studio" OR app_target STREQUAL "dataset_studio")
            list(APPEND entrypoints "${CMAKE_CURRENT_SOURCE_DIR}/raft_studio.py")
        elseif(app_target STREQUAL "photo_studio" OR app_target STREQUAL "raw_studio")
            list(APPEND entrypoints "${CMAKE_CURRENT_SOURCE_DIR}/media_studio.py")
        endif()
        set(copy_entrypoints)
        if(entrypoints)
            list(APPEND copy_entrypoints COMMAND ${CMAKE_COMMAND} -E copy_if_different ${entrypoints} "${app_resources}")
        endif()
        # Python and documentation are resources, not link inputs. Updating them
        # must refresh the bundle without relinking six unchanged native apps.
        add_custom_command(OUTPUT "${resource_stamp}"
            COMMAND ${CMAKE_COMMAND} -E make_directory "${CMAKE_CURRENT_BINARY_DIR}/stamps"
            COMMAND ${CMAKE_COMMAND} -E remove_directory "${app_resources}/src/ipde"
            COMMAND ${CMAKE_COMMAND} -E make_directory "${app_resources}/src/ipde"
            COMMAND ${CMAKE_COMMAND} -E copy_if_different ${IPDE_BACKEND_MODULES} "${app_resources}/src/ipde"
            ${copy_entrypoints}
            COMMAND ${CMAKE_COMMAND} -E remove_directory "${app_resources}/docs"
            COMMAND ${CMAKE_COMMAND} -E copy_directory "${CMAKE_CURRENT_SOURCE_DIR}/docs" "${app_resources}/docs"
            COMMAND ${CMAKE_COMMAND} -E copy_if_different "${CMAKE_CURRENT_SOURCE_DIR}/LICENSE" "${app_resources}/LICENSE"
            COMMAND ${CMAKE_COMMAND} -E touch "${resource_stamp}"
            DEPENDS ${app_target} "${deploy_stamp}" ${IPDE_BACKEND_MODULES} ${IPDE_DOCUMENTATION} ${entrypoints}
                "${CMAKE_CURRENT_SOURCE_DIR}/LICENSE"
                "${CMAKE_CURRENT_BINARY_DIR}/generated/resource-inventory.txt"
            COMMENT "Updating Python and documentation resources in ${app_target}")
        add_custom_command(OUTPUT "${deploy_stamp}"
            COMMAND ${CMAKE_COMMAND} -E make_directory "${CMAKE_CURRENT_BINARY_DIR}/stamps"
            COMMAND ${CMAKE_COMMAND} -E make_directory "$<TARGET_BUNDLE_DIR:${app_target}>/Contents/PlugIns/platforms"
            COMMAND ${CMAKE_COMMAND} -E copy_if_different "${IPDE_QT_OFFSCREEN}"
                "$<TARGET_BUNDLE_DIR:${app_target}>/Contents/PlugIns/platforms/libqoffscreen.dylib"
            COMMAND "${IPDE_MACDEPLOYQT}" "$<TARGET_BUNDLE_DIR:${app_target}>"
                -always-overwrite -no-strip -no-codesign -verbose=0
            COMMAND ${CMAKE_COMMAND} -E touch "${deploy_stamp}"
            DEPENDS ${app_target} "${IPDE_QT_OFFSCREEN}" "${IPDE_MACDEPLOYQT}"
            COMMENT "Deploying changed Qt dependencies in ${app_target}")
        list(APPEND IPDE_APP_RESOURCE_STAMPS "${resource_stamp}")
        list(APPEND IPDE_APP_DEPLOY_STAMPS "${deploy_stamp}")
    endforeach()
    set(studio_stamp "${CMAKE_CURRENT_BINARY_DIR}/stamps/ipde-studio-suite.stamp")
    add_custom_command(OUTPUT "${studio_stamp}"
        COMMAND "${IPDE_PYTHON_EXECUTABLE}" "${CMAKE_CURRENT_SOURCE_DIR}/scripts/bundle_macos.py"
            "$<TARGET_BUNDLE_DIR:ipde_studio>" --assemble
            "$<TARGET_BUNDLE_DIR:ipde_gui>" "$<TARGET_BUNDLE_DIR:raft_studio>"
            "$<TARGET_BUNDLE_DIR:dataset_studio>" "$<TARGET_BUNDLE_DIR:photo_studio>"
            "$<TARGET_BUNDLE_DIR:raw_studio>" --skip-audit
        COMMAND ${CMAKE_COMMAND} -E touch "${studio_stamp}"
        DEPENDS ${IPDE_APP_RESOURCE_STAMPS} ${IPDE_APP_DEPLOY_STAMPS}
            "${CMAKE_CURRENT_SOURCE_DIR}/scripts/bundle_macos.py"
        COMMENT "Updating changed IPDE Studio subapps and signatures")
    add_custom_target(ipde_studio_bundle ALL DEPENDS "${studio_stamp}")

    set(distribution_bundle "${CMAKE_CURRENT_BINARY_DIR}/dist/IPDE Studio.app")
    set(runtime_stamp "${CMAKE_CURRENT_BINARY_DIR}/stamps/python-runtime-inputs.json")
    # This cheap input scan detects changes to the interpreter, locked package
    # files or external native runtime dependencies without importing models.
    add_custom_target(ipde_runtime_inputs
        COMMAND "${IPDE_PYTHON_EXECUTABLE}" "${CMAKE_CURRENT_SOURCE_DIR}/scripts/bundle_macos.py"
            "${distribution_bundle}" --runtime-inputs "${runtime_stamp}"
        BYPRODUCTS "${runtime_stamp}"
        COMMENT "Checking packaged Python input versions and file changes")
    set(distribution_stamp "${CMAKE_CURRENT_BINARY_DIR}/stamps/ipde-distribution.stamp")
    add_custom_command(OUTPUT "${distribution_stamp}"
        COMMAND "${IPDE_PYTHON_EXECUTABLE}" "${CMAKE_CURRENT_SOURCE_DIR}/scripts/bundle_macos.py"
            "$<TARGET_BUNDLE_DIR:ipde_studio>" --output "${distribution_bundle}"
            --bundle-python --skip-audit
        COMMAND "${IPDE_PYTHON_EXECUTABLE}" "${CMAKE_CURRENT_SOURCE_DIR}/scripts/bundle_macos.py"
            "${distribution_bundle}" --runtime-inputs "${runtime_stamp}"
        COMMAND ${CMAKE_COMMAND} -E touch "${distribution_stamp}"
        BYPRODUCTS "${distribution_bundle}/Contents/Info.plist"
            "${distribution_bundle}/Contents/Resources/runtime-build.json"
            "${distribution_bundle}/Contents/Resources/python/bin/python3"
        DEPENDS "${studio_stamp}" "${runtime_stamp}"
            "${CMAKE_CURRENT_SOURCE_DIR}/scripts/bundle_macos.py"
            "${CMAKE_CURRENT_SOURCE_DIR}/requirements-release-macos.txt"
        COMMENT "Updating standalone suite; reusing unchanged packaged Python")
    add_custom_target(distribution DEPENDS "${distribution_stamp}")
    add_dependencies(distribution ipde_runtime_inputs)
    add_custom_target(release-check
        COMMAND "${IPDE_PYTHON_EXECUTABLE}" "${CMAKE_CURRENT_SOURCE_DIR}/scripts/bundle_macos.py"
            "${distribution_bundle}" --release-check
        DEPENDS distribution
        COMMENT "Auditing all native dependencies and bundled Python imports")
endif()
