#pragma once

#include <QIcon>
#include <QImage>
#include <QPainter>
#include <QPainterPath>
#include <QPixmap>

namespace IPDE {
// One vector drawing serves toolbar, Dock and bundle icons at every size.
inline QImage appIconImage(const QString &role, int size = 256, bool monochrome = false) {
    QImage image(size, size, QImage::Format_ARGB32_Premultiplied);
    image.fill(Qt::transparent);
    QPainter p(&image); p.setRenderHint(QPainter::Antialiasing); p.scale(size / 256.0, size / 256.0);
    const QColor accent = role == "photo" ? QColor("#ffad94") : role == "raw" ? QColor("#ebc56c")
        : role == "trainer" ? QColor("#b9a1ff") : role == "datasets" ? QColor("#81d6b2")
        : role == "extractor" ? QColor("#7bc7ff") : QColor("#8bdfdd");
    if (!monochrome) {
        QLinearGradient background(24, 20, 230, 240); background.setColorAt(0, QColor("#243951")); background.setColorAt(1, QColor("#101e32"));
        p.setPen(QPen(QColor("#476076"), 2)); p.setBrush(background); p.drawRoundedRect(QRectF(15, 15, 226, 226), 51, 51);
        p.setPen(QPen(QColor(255, 255, 255, 24), 2)); p.setBrush(Qt::NoBrush); p.drawRoundedRect(QRectF(23, 23, 210, 210), 44, 44);
    }
    const QColor ink = monochrome ? QColor(Qt::black) : accent;
    p.setPen(QPen(ink, 8, Qt::SolidLine, Qt::RoundCap, Qt::RoundJoin)); p.setBrush(Qt::NoBrush);
    if (role == "photo") {
        p.drawRoundedRect(QRectF(55, 53, 146, 150), 21, 21); p.drawEllipse(QRectF(108, 78, 40, 43));
        QPainterPath shoulders; shoulders.moveTo(85, 173); shoulders.cubicTo(87, 127, 169, 127, 171, 173); p.drawPath(shoulders);
        p.drawLine(66, 190, 189, 190);
    } else if (role == "raw") {
        for (int y = 0; y < 3; ++y) for (int x = 0; x < 3; ++x) {
            QColor cell = ink; if (!monochrome) cell.setAlpha(x == y ? 255 : 110);
            p.setPen(Qt::NoPen); p.setBrush(cell); p.drawRoundedRect(QRectF(65 + x * 44, 65 + y * 44, 36, 36), 8, 8);
        }
    } else if (role == "trainer") {
        const QList<QPointF> nodes{{68, 77}, {68, 176}, {129, 65}, {129, 128}, {129, 191}, {192, 101}, {192, 155}};
        p.setPen(QPen(ink, 5));
        for (int a : {0, 1}) for (int b : {2, 3, 4}) p.drawLine(nodes[a], nodes[b]);
        for (int a : {2, 3, 4}) for (int b : {5, 6}) p.drawLine(nodes[a], nodes[b]);
        p.setPen(Qt::NoPen); p.setBrush(ink); for (const auto &node : nodes) p.drawEllipse(node, 12, 12);
    } else {
        for (int layer = 2; layer >= 0; --layer) {
            const double y = 75 + layer * 38;
            QPainterPath plane; plane.moveTo(56, y + 26); plane.lineTo(127, y - 12); plane.lineTo(200, y + 26); plane.lineTo(127, y + 65); plane.closeSubpath();
            QColor fill = ink; fill.setAlpha(monochrome ? 255 : 28 + (2 - layer) * 22);
            p.setPen(QPen(ink, 7, Qt::SolidLine, Qt::RoundCap, Qt::RoundJoin)); p.setBrush(fill); p.drawPath(plane);
        }
        if (role == "extractor") {
            p.setPen(QPen(QColor("#f3f8ff"), 7, Qt::SolidLine, Qt::RoundCap, Qt::RoundJoin));
            p.drawLine(128, 60, 128, 119); p.drawLine(111, 103, 128, 120); p.drawLine(145, 103, 128, 120);
        } else if (role == "datasets") {
            p.setPen(Qt::NoPen); p.setBrush(ink); p.drawEllipse(QPointF(128, 100), 10, 10);
        }
    }
    return image;
}
inline QIcon appIcon(const QString &role) {
    QIcon icon;
    for (int size : {16, 32, 64, 128, 256, 512}) icon.addPixmap(QPixmap::fromImage(appIconImage(role, size)));
    return icon;
}
inline QIcon menuBarIcon() { QIcon icon(QPixmap::fromImage(appIconImage("studio", 32, true))); icon.setIsMask(true); return icon; }
} // namespace IPDE
