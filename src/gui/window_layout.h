#pragma once

#include <QApplication>
#include <QEvent>
#include <QLayout>
#include <QMainWindow>
#include <QScreen>
#include <QScrollArea>
#include <QTimer>

namespace IPDE {

// Keep the frame (including the title bar) inside the desktop's usable area.
// Content constraints belong to the scroll widget, not the top-level window.
inline void fitWindowToAvailableGeometry(QWidget *window, const QRect &available) {
    if (!available.isValid() || window->isMaximized() || window->isFullScreen()) return;
    const QRect bounds = available.adjusted(8, 8, -8, -8);
    const QSize frameExtra = window->frameGeometry().size() - window->size();
    const QSize maximum = bounds.size() - frameExtra;
    window->resize(window->size().boundedTo(maximum));
    const QSize frame = window->frameGeometry().size();
    const QPoint position = window->frameGeometry().topLeft();
    window->move(qBound(bounds.left(), position.x(), qMax(bounds.left(), bounds.right() - frame.width() + 1)),
                 qBound(bounds.top(), position.y(), qMax(bounds.top(), bounds.bottom() - frame.height() + 1)));
}

class WindowScreenGuard final : public QObject {
public:
    explicit WindowScreenGuard(QWidget *window) : QObject(window), window_(window) {
        window->installEventFilter(this);
        for (auto *screen : QGuiApplication::screens()) {
            connect(screen, &QScreen::availableGeometryChanged, this, [this] { scheduleFit(); });
        }
    }
protected:
    bool eventFilter(QObject *, QEvent *event) override {
        if (event->type() == QEvent::Show || event->type() == QEvent::ScreenChangeInternal) scheduleFit();
        return false;
    }
private:
    void scheduleFit() {
        QTimer::singleShot(0, this, [this] {
            if (auto *screen = window_->screen()) fitWindowToAvailableGeometry(window_, screen->availableGeometry());
        });
    }
    QWidget *window_;
};

inline QScrollArea *setScrollableCentralWidget(QMainWindow *window, QWidget *content, const QSize &preferred) {
    if (content->layout()) content->layout()->setSizeConstraint(QLayout::SetMinimumSize);
    auto *scroll = new QScrollArea(window);
    scroll->setObjectName("windowContentScroll");
    scroll->setFrameShape(QFrame::NoFrame);
    scroll->setWidgetResizable(true);
    scroll->setSizeAdjustPolicy(QAbstractScrollArea::AdjustIgnored);
    scroll->setWidget(content);
    window->setCentralWidget(scroll);
    window->setMinimumSize(320, 240);
    const auto *screen = window->screen();
    window->resize(screen ? preferred.boundedTo(screen->availableGeometry().size() - QSize(32, 64)) : preferred);
    new WindowScreenGuard(window);
    return scroll;
}

} // namespace IPDE
