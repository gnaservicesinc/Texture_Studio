#include "window_layout.h"

#include <QLabel>
#include <QPushButton>
#include <QScrollBar>
#include <QVBoxLayout>
#include <iostream>
#include <stdexcept>

namespace {
void require(bool condition, const char *message) {
    if (!condition) throw std::runtime_error(message);
}
void settle() {
    for (int i = 0; i < 5; ++i) QCoreApplication::processEvents();
}

void reachableControlsOnSmallScreens() {
    QMainWindow window;
    auto *content = new QWidget;
    auto *layout = new QVBoxLayout(content);
    layout->addWidget(new QLabel("Studio controls", content));
    auto *settings = new QWidget(content);
    settings->setMinimumSize(720, 900);
    layout->addWidget(settings);
    auto *action = new QPushButton("Export or continue", content);
    layout->addWidget(action);
    auto *scroll = IPDE::setScrollableCentralWidget(&window, content, QSize(1300, 1200));
    window.show(); settle();
    const QRect desktop(50, 30, 800, 600);
    IPDE::fitWindowToAvailableGeometry(&window, desktop); settle();
    require(desktop.contains(window.frameGeometry()), "window extends outside usable screen");
    require(scroll->verticalScrollBar()->maximum() > 0, "tall content has no vertical scrollbar");
    require(content->height() >= 900, "scroll content was compressed below its controls");
    scroll->verticalScrollBar()->setValue(scroll->verticalScrollBar()->maximum()); settle();
    require(scroll->viewport()->rect().contains(action->mapTo(scroll->viewport(), action->rect().center())),
            "bottom action cannot be reached by scrolling");

    // A smaller monitor may need both scroll directions; it must not force
    // the application window to match the form's minimum width or height.
    const QRect smallDesktop(0, 0, 600, 420);
    IPDE::fitWindowToAvailableGeometry(&window, smallDesktop); settle();
    require(smallDesktop.contains(window.frameGeometry()), "window cannot fit a smaller monitor");
    require(scroll->horizontalScrollBar()->maximum() > 0, "wide controls have no horizontal scrollbar");

    const int previousMaximum = scroll->verticalScrollBar()->maximum();
    settings->setMinimumHeight(1300); settle();
    require(scroll->verticalScrollBar()->maximum() > previousMaximum, "expanded settings did not update scroll range");
    scroll->ensureWidgetVisible(action); settle();
    require(scroll->viewport()->rect().contains(action->mapTo(scroll->viewport(), action->rect().center())),
            "bottom action is unreachable after settings expansion");
}
} // namespace

int main(int argc, char **argv) {
    QApplication app(argc, argv);
    try {
        reachableControlsOnSmallScreens();
        std::cout << "Window screen fitting and scroll reachability passed\n";
        return 0;
    } catch (const std::exception &error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
