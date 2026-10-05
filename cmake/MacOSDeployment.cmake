# Qt publishes its supported minimum in Qt6ConfigExtras.cmake. The bundled
# Python/PyTorch runtime also requires macOS 14.0. Keep an explicit target only
# when both runtime requirements can be met.
function(ipde_select_macos_deployment output)
    set(required "14.0")
    if(QT_SUPPORTED_MIN_MACOS_VERSION VERSION_GREATER required)
        set(required "${QT_SUPPORTED_MIN_MACOS_VERSION}")
    endif()
    if(NOT CMAKE_OSX_DEPLOYMENT_TARGET)
        set(selected "${required}")
    elseif(CMAKE_OSX_DEPLOYMENT_TARGET VERSION_LESS required)
        message(FATAL_ERROR
            "macOS deployment target ${CMAKE_OSX_DEPLOYMENT_TARGET} is below the required ${required} "
            "(Qt ${Qt6_VERSION} requires ${QT_SUPPORTED_MIN_MACOS_VERSION}; bundled Python requires 14.0). "
            "Use automatic selection with -DCMAKE_OSX_DEPLOYMENT_TARGET=, or select a compatible Qt kit.")
    else()
        set(selected "${CMAKE_OSX_DEPLOYMENT_TARGET}")
    endif()
    set(${output} "${selected}" PARENT_SCOPE)
endfunction()
