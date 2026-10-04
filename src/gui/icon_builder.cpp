#include "studio_icons.h"
#include <QGuiApplication>
#include <QDir>
#include <QProcess>

int main(int argc, char **argv) {
    QGuiApplication app(argc, argv);
    const auto args = app.arguments(); if (args.size() != 2) return 2;
    QDir root(args[1]); if (!root.mkpath(".")) return 3;
    for (const QString &role : {"studio", "extractor", "datasets", "trainer", "photo", "raw"}) {
        const QString folder = root.filePath(role + ".iconset"); QDir().mkpath(folder);
        for (int points : {16, 32, 128, 256, 512}) for (int scale : {1, 2}) {
            const QString name = QString("icon_%1x%1%2.png").arg(points).arg(scale == 2 ? "@2x" : "");
            if (!IPDE::appIconImage(role, points * scale).save(QDir(folder).filePath(name))) return 4;
        }
        if (!IPDE::appIconImage(role, 512).save(root.filePath(role + ".png"))) return 4;
        if (QProcess::execute("/usr/bin/iconutil", {"-c", "icns", "-o", root.filePath(role + ".icns"), folder}) != 0) return 5;
    }
    return 0;
}
