#define IPDE_STUDIO_REGRESSION
#define main ipdeStudioApplicationMain
#include "../src/gui/trainer.cpp"
#undef main

#include <iostream>
#include <stdexcept>

namespace {
void require(bool condition, const char *message) {
    if (!condition) throw std::runtime_error(message);
}

void settle() {
    for (int pass = 0; pass < 8; ++pass) QCoreApplication::processEvents(QEventLoop::AllEvents, 10);
    QCoreApplication::sendPostedEvents(nullptr, QEvent::DeferredDelete);
}

void mouse(QWidget *widget, QEvent::Type type, const QPoint &global, Qt::MouseButton button, Qt::MouseButtons buttons) {
    QMouseEvent event(type, widget->mapFromGlobal(global), global, button, buttons, Qt::NoModifier);
    QApplication::sendEvent(widget, &event);
}

void nativePreviewInteraction(const QString &imagePath) {
    DepthPreview preview(nullptr); preview.resize(420, 380); preview.show();
    require(preview.load(imagePath), "fixture image failed to load"); settle();
    const QSize hint = preview.sizeHint(), minimumHint = preview.minimumSizeHint();
    preview.resize(760, 620); settle();
    require(preview.sizeHint() == hint && preview.minimumSizeHint() == minimumHint,
        "fitted image changed the preview's layout size hints");
    const QSize nativeSize(2600, 2200);
    const QSize fitted = nativeSize.scaled(preview.size(), Qt::KeepAspectRatio);
    const QPointF point(.60, .65);
    const QPoint local(qRound((preview.width() - fitted.width()) / 2. + fitted.width() * point.x()),
                       qRound((preview.height() - fitted.height()) / 2. + fitted.height() * point.y()));
    const QPoint global = preview.mapToGlobal(local);
    auto open = [&] {
        mouse(&preview, QEvent::MouseButtonPress, global, Qt::LeftButton, Qt::LeftButton);
        mouse(&preview, QEvent::MouseButtonRelease, global, Qt::LeftButton, Qt::NoButton);
        settle();
        return preview.findChild<QDialog *>("nativePixelPopup");
    };
    QDialog *popup = open(); require(popup && popup->isVisible(), "image click did not open native preview");
    require(popup->windowType() == Qt::Popup, "native preview lacks outside-click popup behavior");
    auto *scroll = popup->findChild<QScrollArea *>("nativePixelScroll");
    auto *full = popup->findChild<QLabel *>("nativePixelImage");
    require(scroll && full && full->size() == nativeSize, "native preview resized its display pixels");
    const QPointF actualPoint((local.x() - (preview.width() - fitted.width()) / 2.) / fitted.width(),
                             (local.y() - (preview.height() - fitted.height()) / 2.) / fitted.height());
    require(std::abs(scroll->horizontalScrollBar()->value() + scroll->viewport()->width() / 2. - full->x() - actualPoint.x() * nativeSize.width()) <= 1.,
        "native preview did not center the clicked horizontal position");
    require(std::abs(scroll->verticalScrollBar()->value() + scroll->viewport()->height() / 2. - full->y() - actualPoint.y() * nativeSize.height()) <= 1.,
        "native preview did not center the clicked vertical position");
    require(QGuiApplication::primaryScreen()->availableGeometry().contains(popup->geometry()), "native preview extends beyond the screen");
    const QPoint start = scroll->viewport()->mapToGlobal(scroll->viewport()->rect().center());
    const int startX = scroll->horizontalScrollBar()->value(), startY = scroll->verticalScrollBar()->value();
    mouse(full, QEvent::MouseButtonPress, start, Qt::LeftButton, Qt::LeftButton);
    mouse(full, QEvent::MouseMove, start + QPoint(50, 40), Qt::NoButton, Qt::LeftButton);
    mouse(full, QEvent::MouseButtonRelease, start + QPoint(50, 40), Qt::LeftButton, Qt::NoButton);
    require(popup->isVisible(), "left drag dismissed the native preview");
    require(scroll->horizontalScrollBar()->value() == startX - 50 && scroll->verticalScrollBar()->value() == startY - 40,
        "dragging did not pan the native image");
    mouse(full, QEvent::MouseButtonPress, start, Qt::RightButton, Qt::RightButton); settle();
    require(!preview.findChild<QDialog *>("nativePixelPopup"), "right-click did not close native preview");
    popup = open(); require(popup && popup->isVisible(), "native preview could not reopen");
    QWidget outside;
    const QPoint outsidePoint = popup->frameGeometry().bottomRight() + QPoint(20, 20);
    mouse(&outside, QEvent::MouseButtonPress, outsidePoint, Qt::LeftButton, Qt::LeftButton); settle();
    require(!preview.findChild<QDialog *>("nativePixelPopup"), "outside click did not close native preview");
    NativePixelPreview *edge = new NativePixelPreview(QPixmap(imagePath), QPointF(.02, .98), global, &preview);
    edge->show(); settle();
    scroll = edge->findChild<QScrollArea *>("nativePixelScroll"); full = edge->findChild<QLabel *>("nativePixelImage");
    require(std::abs(scroll->horizontalScrollBar()->value() + scroll->viewport()->width() / 2. - full->x() - .02 * nativeSize.width()) <= 1.
        && std::abs(scroll->verticalScrollBar()->value() + scroll->viewport()->height() / 2. - full->y() - .98 * nativeSize.height()) <= 1.,
        "clicks near an image edge were clamped away from the popup center");
    edge->close(); settle();
}

void reviewCaptionsHaveSpace(const QString &imagePath) {
    TrainerWindow window; window.resize(940, 680); window.show();
    auto *tabs = window.findChild<QTabWidget *>(); require(tabs, "studio tabs missing"); tabs->setCurrentIndex(1);
    auto *content = window.findChild<QWidget *>("reviewPreviewContent");
    auto *scroll = window.findChild<QScrollArea *>("reviewPreviewScroll");
    auto *source = window.findChild<QLabel *>("reviewSourcePreview");
    auto *depth = window.findChild<QLabel *>("reviewDepthPreview");
    auto *stats = window.findChild<QLabel *>("reviewPreviewStats");
    auto *legend = window.findChild<QLabel *>("reviewPreviewLegend");
    require(content && scroll && source && depth && stats && legend, "review gallery missing");
    require(dynamic_cast<DepthPreview *>(source)->load(imagePath) && dynamic_cast<DepthPreview *>(depth)->load(imagePath),
        "review gallery fixture could not load");
    stats->setText("Selected training target: Teacher with estimated meter scale · 2688 × 2016 samples · 100.0% valid · range 0.5976515 to 1.75468 meters. "
                   "This deliberately long caption must wrap below the source and depth images without covering any pixels.");
    settle();
    require(source->width() > content->width() / 3, "hidden comparison columns still reserve image gallery space");
    auto bounds = [content](QWidget *widget) { return QRect(widget->mapTo(content, QPoint()), widget->size()); };
    require(bounds(stats).top() > bounds(source).bottom() && bounds(stats).top() > bounds(depth).bottom(),
        "review statistics overlap the image gallery");
    require(bounds(legend).top() > bounds(stats).bottom(), "review legend overlaps its statistics");
    require(stats->height() >= stats->heightForWidth(stats->width()) && legend->height() >= legend->heightForWidth(legend->width()),
        "wrapped review captions were allocated too little height");
    const QString screenshot = qEnvironmentVariable("IPDE_PREVIEW_REGRESSION_SCREENSHOT");
    if (!screenshot.isEmpty()) require(content->grab().save(screenshot), "review regression screenshot could not be saved");
    scroll->setMaximumHeight(240); settle();
    require(scroll->verticalScrollBar()->maximum() > 0, "compressed review gallery has no scrolling path");
    require(bounds(stats).top() > bounds(depth).bottom() && bounds(legend).top() > bounds(stats).bottom(),
        "review captions overlap when gallery is compressed");
}

void datasetReviewUsesAvailableWidth(const QString &imagePath) {
    TrainerWindow window(true); window.autosavePaused_ = true; window.requestedReviewOpen_ = false;
    { QSignalBlocker blocker(window.tabs_); window.tabs_->setCurrentIndex(0); }
    window.requestedReviewPath_ = "/fixture/review-layout";
    window.populateReview({{"dataset_path", window.requestedReviewPath_}, {"samples", QJsonArray{
        QJsonObject{{"id", "photo-depthpro"}, {"source_id", "photo"}, {"source_path", "/fixture/IMG_1829_full_display.HEIC"},
            {"teacher_id", "depthpro"}, {"teacher_model", "depthpro"}, {"split", "train"}, {"included", true}, {"group_id", "room"}}}}});
    { QSignalBlocker blocker(window.tabs_); window.tabs_->setCurrentIndex(1); }
    window.show(); settle(); window.resize(1250, 850);
    auto *scroll = window.findChild<QScrollArea *>("reviewPreviewScroll");
    auto *content = window.findChild<QWidget *>("reviewPreviewContent");
    auto *split = window.findChild<QSplitter *>("datasetReviewSplit");
    require(scroll && content && split && split->orientation() == Qt::Vertical, "dataset photo list still competes horizontally with preview columns");
    for (auto *image : window.depthPreviews_) { image->load(imagePath); image->show(); }
    window.rgbPreview_->load(imagePath);
    for (auto *title : window.depthTitles_) { title->setText("Teacher target with full processing provenance and an intentionally long descriptive caption"); title->show(); }
    window.previewStats_->setText(QString("Full display teacher · 5712 × 4284 · relative_depth · model input 1036 × 770. Source: ") + QString(300, 'x') + ".HEIC\nGeometry must be checked beside its own display photograph.");
    window.reviewCount_->setText("Included 400 teacher entries · Training 320 · Validation 80 · changes saved. Set aside independent scenes before training.");
    settle();
    auto *photo = window.reviewSamples_->topLevelItem(0);
    for (int column : {6, 7, 8}) {
        auto *button = qobject_cast<QPushButton *>(window.reviewSamples_->itemWidget(photo, column));
        require(button && button->width() >= button->sizeHint().width() && window.reviewSamples_->viewport()->rect().contains(button->geometry()),
                "teacher button text is clipped or requires horizontal scrolling at a normal window size");
    }
    require(window.reviewSamples_->columnWidth(0) >= 180, "photo names retained a cramped fixed column despite available width");
    require(content->width() <= scroll->viewport()->width() && scroll->horizontalScrollBar()->maximum() == 0,
            "long review metadata forced horizontal preview overflow");
    auto bounds = [content](QWidget *widget) { return QRect(widget->mapTo(content, QPoint()), widget->size()); };
    for (auto *image : window.depthPreviews_) require(bounds(window.previewStats_).top() > bounds(image).bottom(), "review metadata overlaps a comparison image");
    require(window.depthPreviews_[1]->width() >= content->width() / 3 && window.depthPreviews_[2]->width() >= content->width() / 3,
            "comparison images were squeezed into four narrow columns");
    require(window.previewStats_->height() >= window.previewStats_->heightForWidth(window.previewStats_->width()), "long review information was clipped vertically");
}
}

int main(int argc, char **argv) {
    (void)argc;
    char smoke[] = "--smoke-test"; char *applicationArguments[]{argv[0], smoke, nullptr}; int applicationArgc = 2;
    QApplication application(applicationArgc, applicationArguments);
    QSettings::setDefaultFormat(QSettings::IniFormat);
    QTemporaryDir settings; QSettings::setPath(QSettings::IniFormat, QSettings::UserScope, settings.path());
    QTemporaryDir fixture;
    QPixmap image(2600, 2200); image.fill(QColor(30, 90, 160));
    const QString path = fixture.filePath("preview.png");
    try {
        require(image.save(path), "could not save preview fixture");
        nativePreviewInteraction(path);
        reviewCaptionsHaveSpace(path);
        datasetReviewUsesAvailableWidth(path);
        std::cout << "Native preview interaction and review caption layout passed.\n";
        return 0;
    } catch (const std::exception &error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
