#pragma once

#include <QCoreApplication>
#include <QDir>
#include <QFileInfo>
#include <QStandardPaths>

#ifndef IPDE_PYTHON_EXECUTABLE
#define IPDE_PYTHON_EXECUTABLE "python3"
#endif

namespace IPDE {
inline QString pythonExecutable() {
    // Subapps share the runtime owned by the outer Studio bundle. Walking up
    // also supports a separately packaged subapp without absolute build paths.
    QDir directory(QCoreApplication::applicationDirPath());
    for (int depth = 0; depth < 8; ++depth) {
        const QString candidate = directory.filePath("Resources/python/bin/python3");
        if (QFileInfo(candidate).isExecutable()) return QDir::cleanPath(candidate);
        if (!directory.cdUp()) break;
    }
    const QString configured = QString::fromUtf8(IPDE_PYTHON_EXECUTABLE);
    if (QFileInfo(configured).isExecutable()) return configured;
    const QString found = QStandardPaths::findExecutable(configured);
    return found.isEmpty() ? configured : found;
}
}
