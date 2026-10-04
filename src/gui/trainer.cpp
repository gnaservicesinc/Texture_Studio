#include <QApplication>
#include <QComboBox>
#include <QCheckBox>
#include <QDateTime>
#include <QDoubleSpinBox>
#include <QDesktopServices>
#include <QDialog>
#include <QDir>
#include <QFileDialog>
#include <QFileInfo>
#include <QFont>
#include <QFormLayout>
#include <QGroupBox>
#include <QHeaderView>
#include <QJsonArray>
#include <QJsonDocument>
#include <QJsonObject>
#include <QLabel>
#include <QLineEdit>
#include <QMainWindow>
#include <QMessageBox>
#include <QMouseEvent>
#include <QPainter>
#include <QPointer>
#include <QPlainTextEdit>
#include <QPixmap>
#include <QProcess>
#include <QProgressBar>
#include <QPushButton>
#include <QRegularExpression>
#include <QSettings>
#include <QSet>
#include <QScrollArea>
#include <QScrollBar>
#include <QScreen>
#include <QSignalBlocker>
#include <QSpinBox>
#include <QSplitter>
#include <QStandardPaths>
#include <QStatusBar>
#include <QStyledItemDelegate>
#include <QTabWidget>
#include <QTemporaryFile>
#include <QTemporaryDir>
#include <QTimer>
#include <QTreeWidget>
#include <QUrl>
#include <QVector3D>
#include <QVBoxLayout>

#include <memory>
#include <functional>
#include <cmath>
#include <algorithm>

#include "project_session.h"
#include "studio_icons.h"
#include "window_layout.h"

#ifndef IPDE_DATASET_STUDIO
#define IPDE_DATASET_STUDIO 0
#endif

#ifndef IPDE_TRAINER_SCRIPT
#ifdef RAFT_STUDIO_SOURCE_SCRIPT
#define IPDE_TRAINER_SCRIPT RAFT_STUDIO_SOURCE_SCRIPT
#else
#define IPDE_TRAINER_SCRIPT "raft_studio.py"
#endif
#endif
#ifndef IPDE_PYTHON_EXECUTABLE
#define IPDE_PYTHON_EXECUTABLE "python3"
#endif

namespace {

QString scriptPath() {
    const QString bundled = QDir(QCoreApplication::applicationDirPath()).absoluteFilePath("../Resources/raft_studio.py");
    return QFileInfo::exists(bundled) ? QDir::cleanPath(bundled) : QString::fromUtf8(IPDE_TRAINER_SCRIPT);
}

QString pythonPath() {
    const QString configured = QString::fromUtf8(IPDE_PYTHON_EXECUTABLE);
    if (QFileInfo::exists(configured)) return configured;
    const QString found = QStandardPaths::findExecutable(configured);
    return found.isEmpty() ? configured : found;
}

class GroupDelegate final : public QStyledItemDelegate {
public:
    using QStyledItemDelegate::QStyledItemDelegate;
    QWidget *createEditor(QWidget *parent, const QStyleOptionViewItem &option, const QModelIndex &index) const override {
        return index.column() == 1 ? QStyledItemDelegate::createEditor(parent, option, index) : nullptr;
    }
};

class NativePixelPreview final : public QDialog {
public:
    NativePixelPreview(const QPixmap &display, const QPointF &position, const QPoint &globalPosition, QWidget *parent)
        : QDialog(parent, Qt::Popup) {
        setAttribute(Qt::WA_DeleteOnClose);
        setObjectName("nativePixelPopup");
        setWindowTitle("Native pixel preview (display copy)");
        auto *layout = new QVBoxLayout(this);
        auto *help = new QLabel("Native pixels · Drag to pan · Right-click or click outside to close", this);
        help->setWordWrap(true); layout->addWidget(help);
        scroll_ = new QScrollArea(this); scroll_->setObjectName("nativePixelScroll");
        scroll_->setAlignment(Qt::AlignCenter);
        scroll_->setHorizontalScrollBarPolicy(Qt::ScrollBarAlwaysOn);
        scroll_->setVerticalScrollBarPolicy(Qt::ScrollBarAlwaysOn);
        canvas_ = new QWidget; canvas_->setFixedSize(display.size()); canvas_->setCursor(Qt::OpenHandCursor);
        auto *full = new QLabel(canvas_); image_ = full; full->setObjectName("nativePixelImage");
        full->setPixmap(display); full->setFixedSize(display.size());
        full->setCursor(Qt::OpenHandCursor); full->installEventFilter(this);
        scroll_->viewport()->setCursor(Qt::OpenHandCursor); scroll_->viewport()->installEventFilter(this);
        scroll_->setWidget(canvas_); layout->addWidget(scroll_, 1);
        qApp->installEventFilter(this);
        QScreen *screen = QGuiApplication::screenAt(globalPosition);
        if (!screen) screen = QGuiApplication::primaryScreen();
        const QRect available = screen ? screen->availableGeometry() : QRect(globalPosition - QPoint(550, 400), QSize(1100, 800));
        resize(qMin(1100, available.width()), qMin(800, available.height()));
        move(qBound(available.left(), globalPosition.x() - width() / 2, available.right() - width() + 1),
             qBound(available.top(), globalPosition.y() - height() / 2, available.bottom() - height() + 1));
        // Scroll ranges are available only after the popup's first layout pass.
        QTimer::singleShot(0, this, [this, available, position, imageSize = display.size()] {
            IPDE::fitWindowToAvailableGeometry(this, available);
            this->layout()->activate();
            updateCanvas();
            scroll_->horizontalScrollBar()->setValue(qRound(image_->x() + position.x() * imageSize.width() - scroll_->viewport()->width() / 2.));
            scroll_->verticalScrollBar()->setValue(qRound(image_->y() + position.y() * imageSize.height() - scroll_->viewport()->height() / 2.));
        });
    }
    ~NativePixelPreview() override {
        if (qApp) qApp->removeEventFilter(this);
        scroll_->viewport()->removeEventFilter(this);
        image_->removeEventFilter(this);
    }
protected:
    bool eventFilter(QObject *watched, QEvent *event) override {
        if (watched == scroll_->viewport() && event->type() == QEvent::Resize) updateCanvas();
        if (!isVisible()) return QDialog::eventFilter(watched, event);
        if (event->type() == QEvent::MouseButtonPress) {
            auto *mouse = static_cast<QMouseEvent *>(event);
            if (!frameGeometry().contains(mouse->globalPosition().toPoint())) { close(); return true; }
            if (mouse->button() == Qt::RightButton) { close(); return true; }
        }
        if (event->type() == QEvent::MouseMove && dragging_) {
            const QPoint delta = (static_cast<QMouseEvent *>(event)->globalPosition() - dragStart_).toPoint();
            scroll_->horizontalScrollBar()->setValue(scrollStart_.x() - delta.x());
            scroll_->verticalScrollBar()->setValue(scrollStart_.y() - delta.y()); return true;
        }
        if (event->type() == QEvent::MouseButtonRelease && dragging_) {
            dragging_ = false; canvas_->setCursor(Qt::OpenHandCursor); image_->setCursor(Qt::OpenHandCursor); scroll_->viewport()->setCursor(Qt::OpenHandCursor); return true;
        }
        if (watched != canvas_ && watched != image_ && watched != scroll_->viewport()) return QDialog::eventFilter(watched, event);
        if (event->type() == QEvent::MouseButtonPress) {
            auto *mouse = static_cast<QMouseEvent *>(event);
            if (mouse->button() == Qt::LeftButton) {
                dragging_ = true; dragStart_ = mouse->globalPosition();
                scrollStart_ = QPoint(scroll_->horizontalScrollBar()->value(), scroll_->verticalScrollBar()->value());
                canvas_->setCursor(Qt::ClosedHandCursor); image_->setCursor(Qt::ClosedHandCursor); scroll_->viewport()->setCursor(Qt::ClosedHandCursor); return true;
            }
        }
        return QDialog::eventFilter(watched, event);
    }
    void mousePressEvent(QMouseEvent *event) override {
        if (event->button() == Qt::RightButton) { close(); return; }
        QDialog::mousePressEvent(event);
    }
private:
    void updateCanvas() {
        // Allow edge pixels to reach the center without resampling the image.
        const QSize viewport = scroll_->viewport()->size();
        canvas_->setFixedSize(image_->size() + viewport);
        image_->move(viewport.width() / 2, viewport.height() / 2);
    }
    QScrollArea *scroll_ = nullptr;
    QWidget *canvas_ = nullptr;
    QLabel *image_ = nullptr;
    QPointF dragStart_; QPoint scrollStart_; bool dragging_ = false;
};

class DepthPreview final : public QLabel {
public:
    explicit DepthPreview(QWidget *parent) : QLabel(parent) {
        setAlignment(Qt::AlignCenter); setMinimumSize(160, 200); setMouseTracking(true);
        setSizePolicy(QSizePolicy::Expanding, QSizePolicy::Expanding);
        setStyleSheet("background: #202124; color: #eeeeee; border-radius: 4px;");
        setWordWrap(true);
    }
    // QLabel's pixmap size hint otherwise grows with the last fitted image,
    // leaving too little room for the wrapped captions below the gallery.
    QSize sizeHint() const override { return QSize(260, 220); }
    QSize minimumSizeHint() const override { return QSize(160, 200); }
    void setVisible(bool visible) override {
        if (onVisibility) onVisibility(visible);
        QLabel::setVisible(visible);
    }
    bool load(const QString &path) {
        original_ = QPixmap(path);
        if (original_.isNull()) { reset("Preview could not be loaded."); return false; }
        updateImage(); return true;
    }
    void setSource(const QString &path) { source_ = QPixmap(path); updateImage(); }
    void setSurface(const QJsonObject &surface) { surface_ = surface; update(); }
    void setView(const QString &view) { view_ = view; updateImage(); update(); }
    void setAngles(double yaw, double pitch) { yaw_ = yaw; pitch_ = pitch; update(); }
    std::function<void(const QPointF &)> onHover;
    std::function<void(double, double)> onRotate;
    std::function<void(bool)> onVisibility;
    void showMagnifier(const QPointF &position) {
        hover_ = position; update();
    }
    void reset(const QString &message) { original_ = QPixmap(); surface_ = {}; clear(); setText(message); update(); }
protected:
    void resizeEvent(QResizeEvent *event) override { QLabel::resizeEvent(event); updateImage(); }
    void mousePressEvent(QMouseEvent *event) override {
        if (original_.isNull() || event->button() != Qt::LeftButton) return;
        pressed_ = event->position(); dragged_ = false;
    }
    void mouseMoveEvent(QMouseEvent *event) override {
        if (original_.isNull()) return;
        if (view_ == "surface" && event->buttons().testFlag(Qt::LeftButton)) {
            const auto delta = event->position() - pressed_; pressed_ = event->position();
            dragged_ = true; yaw_ += delta.x() * .008; pitch_ = qBound(-1.4, pitch_ + delta.y() * .008, 1.4);
            if (onRotate) onRotate(yaw_, pitch_); update(); return;
        }
        const QPointF position = imagePosition(event->position());
        showMagnifier(position); if (onHover) onHover(position);
    }
    void leaveEvent(QEvent *event) override {
        showMagnifier(QPointF(-1, -1)); if (onHover) onHover(QPointF(-1, -1)); QLabel::leaveEvent(event);
    }
    void mouseReleaseEvent(QMouseEvent *event) override {
        if (original_.isNull() || dragged_ || event->button() != Qt::LeftButton) return;
        const QPointF position = imagePosition(event->position());
        if (position.x() < 0 || position.x() > 1 || position.y() < 0 || position.y() > 1) return;
        if (floating_) { floating_->close(); return; }
        floating_ = new NativePixelPreview(displayPixmap(), position, event->globalPosition().toPoint(), this);
        floating_->show();
    }
    void paintEvent(QPaintEvent *event) override {
        if (view_ == "surface" && !surface_.isEmpty()) { paintSurface(); return; }
        QLabel::paintEvent(event);
        if (original_.isNull() || hover_.x() < 0 || hover_.x() > 1 || hover_.y() < 0 || hover_.y() > 1) return;
        const QPixmap display = displayPixmap(); const int zoom = qMin(160, qMin(width() - 16, height() - 16));
        const QPoint center(qRound(hover_.x() * display.width()), qRound(hover_.y() * display.height()));
        QPainter painter(this); const QRect lens(width() - zoom - 8, 8, zoom, zoom);
        painter.fillRect(lens, Qt::black); painter.drawPixmap(lens.topLeft(), display.copy(QRect(center - QPoint(zoom/2, zoom/2), QSize(zoom, zoom))));
        painter.setPen(Qt::yellow); painter.drawRect(lens.adjusted(0, 0, -1, -1)); painter.drawText(lens.adjusted(5, 4, -5, -4), Qt::AlignBottom | Qt::AlignLeft, "1:1 native pixels");
    }
private:
    QPointF imagePosition(const QPointF &point) const {
        const QSize fitted = original_.size().scaled(size(), Qt::KeepAspectRatio);
        if (fitted.isEmpty()) return QPointF(-1, -1);
        const QPointF offset((width() - fitted.width()) / 2., (height() - fitted.height()) / 2.);
        return QPointF((point.x() - offset.x()) / fitted.width(), (point.y() - offset.y()) / fitted.height());
    }
    QPixmap displayPixmap() const {
        if (view_ != "overlay" || source_.isNull()) return original_;
        QPixmap composed = source_; QPainter painter(&composed); painter.setOpacity(.5);
        painter.drawPixmap(composed.rect(), original_); return composed;
    }
    void updateImage() {
        if (!original_.isNull()) setPixmap(displayPixmap().scaled(size(), Qt::KeepAspectRatio, Qt::SmoothTransformation));
    }
    void paintSurface() {
        QPainter painter(this); painter.fillRect(rect(), QColor("#202124")); painter.setRenderHint(QPainter::Antialiasing);
        const int rows = surface_.value("rows").toInt(), cols = surface_.value("columns").toInt();
        const auto heights = surface_.value("heights").toArray(), valid = surface_.value("valid").toArray();
        if (rows < 2 || cols < 2 || heights.size() != rows * cols || valid.size() != heights.size()) return;
        struct Face { QPolygonF polygon; double depth; QColor color; }; QList<Face> faces;
        auto vertex = [&](int x, int y) {
            const double u = double(x)/(cols - 1) - .5, v = double(y)/(rows - 1) - .5, z = heights[y*cols+x].toDouble() * .30;
            const double rx = u*std::cos(yaw_) + z*std::sin(yaw_), rz = -u*std::sin(yaw_) + z*std::cos(yaw_);
            return QVector3D(rx, v*std::cos(pitch_) - rz*std::sin(pitch_), v*std::sin(pitch_) + rz*std::cos(pitch_));
        };
        const double scale = qMin(width(), height()) * .78;
        const QVector3D light = QVector3D(-.4, -.5, 1).normalized();
        for (int y=0; y<rows-1; ++y) for (int x=0; x<cols-1; ++x) {
            if (!valid[y*cols+x].toBool() || !valid[y*cols+x+1].toBool() || !valid[(y+1)*cols+x].toBool() || !valid[(y+1)*cols+x+1].toBool()) continue;
            const auto a=vertex(x,y), b=vertex(x+1,y), c=vertex(x+1,y+1), d=vertex(x,y+1);
            const auto normal=QVector3D::crossProduct(b-a, d-a).normalized();
            const double shade=.2 + .8*qMax(0., double(QVector3D::dotProduct(normal, light)));
            QPolygonF polygon; for (auto point : {a,b,c,d}) polygon << QPointF(width()/2. + point.x()*scale, height()/2. + point.y()*scale);
            faces.append({polygon, (a.z()+b.z()+c.z()+d.z())/4, QColor::fromRgbF(shade*.65, shade*.8, shade)});
        }
        std::sort(faces.begin(), faces.end(), [](const Face &a, const Face &b) { return a.depth < b.depth; });
        painter.setPen(Qt::NoPen); for (const auto &face : faces) { painter.setBrush(face.color); painter.drawPolygon(face.polygon); }
        painter.setPen(Qt::white); painter.drawText(rect().adjusted(8,8,-8,-8), Qt::AlignBottom | Qt::AlignLeft, "Drag to rotate · display relief");
    }
    QPixmap original_, source_; QJsonObject surface_; QString view_ = "depth";
    QPointer<QDialog> floating_; QPointF hover_{-1, -1}, pressed_; bool dragged_ = false;
    double yaw_ = -.25, pitch_ = -.5;
};

class TrainerWindow final : public QMainWindow {
public:
    TrainerWindow() : settings_("IPDE", "RAFTStudio") {
        const auto arguments = QCoreApplication::arguments();
        const int modeArgument = arguments.indexOf("--mode");
        datasetMode_ = modeArgument >= 0 ? arguments.value(modeArgument + 1) == "datasets" : bool(IPDE_DATASET_STUDIO);
        projectRoot_ = IPDE::projectRoot();
        if (!projectRoot_.isEmpty()) projectSettings_ = std::make_unique<QSettings>(QDir(projectRoot_).filePath("project.ini"), QSettings::IniFormat);
        setWindowTitle(datasetMode_ ? "Dataset Studio" : "RAFT Studio");
        if (projectSettings_) setWindowTitle(windowTitle() + " — "
            + projectSettings_->value("name", QFileInfo(projectRoot_).fileName()).toString());
        auto *central = new QWidget(this);
        auto *root = new QVBoxLayout(central);
        root->setContentsMargins(18, 16, 18, 16);
        auto *title = new QLabel(datasetMode_ ? "Dataset Studio" : "RAFT Studio", central);
        QFont font = title->font(); font.setPointSize(font.pointSize() + 7); font.setBold(true); title->setFont(font);
        root->addWidget(title);
        auto *purpose = new QLabel(datasetMode_ ? "Organize photo collections, compare teachers, and keep the predictions that suit your project." : "Combine curated datasets, select validation, train, and compare the result with your original model.", central);
        purpose->setWordWrap(true); root->addWidget(purpose);

        auto *workspaceRow = new QHBoxLayout;
        workspace_ = new QLineEdit(projectRoot_.isEmpty() ? settings_.value("workspace", "/opt/ipde/raft-workspace").toString() : QDir(projectRoot_).filePath("workspace"), central);
        workspaceRow->addWidget(new QLabel("Workspace", central)); workspaceRow->addWidget(workspace_, 1);
        chooseWorkspace_ = new QPushButton("Choose…", central);
        if (!projectRoot_.isEmpty()) { workspace_->setReadOnly(true); chooseWorkspace_->hide(); }
        refresh_ = new QPushButton("Refresh library", central);
        auto *showWorkspace = new QPushButton("Show folder", central);
        workspaceRow->addWidget(chooseWorkspace_); workspaceRow->addWidget(refresh_); workspaceRow->addWidget(showWorkspace);
        root->addLayout(workspaceRow);
        connect(chooseWorkspace_, &QPushButton::clicked, this, [this] {
            const QString chosen = QFileDialog::getExistingDirectory(this, "Choose workspace", workspace_->text());
            if (!chosen.isEmpty()) { workspace_->setText(chosen); refreshLibrary(); }
        });
        connect(workspace_, &QLineEdit::editingFinished, this, [this] { refreshLibrary(); });
        connect(refresh_, &QPushButton::clicked, this, [this] { refreshLibrary(); });
        connect(showWorkspace, &QPushButton::clicked, this, [this] { showFolder(workspace_->text()); });

        auto *library = new QSplitter(Qt::Horizontal, central); library_ = library;
        auto *datasetBox = new QGroupBox("Datasets", library);
        auto *datasetLayout = new QVBoxLayout(datasetBox);
        datasets_ = new QTreeWidget(datasetBox);
        datasets_->setHeaderLabels({"Dataset", "Entries", "Train / validation", "Teacher", "Category", "Storage"});
        datasets_->header()->setSectionResizeMode(0, QHeaderView::Stretch);
        datasets_->setColumnWidth(1, 65); datasets_->setColumnWidth(2, 140); datasets_->setColumnWidth(3, 185);
        datasets_->setRootIsDecorated(false); datasets_->setAlternatingRowColors(true);
        datasetLayout->addWidget(datasets_);
        auto *datasetButtons = new QHBoxLayout;
        auto *inspect = new QPushButton("Inspect selected", datasetBox);
        auto *review = new QPushButton("Review depth maps", datasetBox);
        auto *showDataset = new QPushButton("Show files", datasetBox);
        auto *archive = new QPushButton("Archive selected", datasetBox);
        auto *compact = new QPushButton("Compact copy…", datasetBox);
        review->setIcon(IPDE::appIcon("datasets")); inspect->setIcon(IPDE::appIcon("datasets"));
        datasetButtons->addWidget(review); datasetButtons->addWidget(inspect); datasetButtons->addWidget(showDataset); datasetButtons->addWidget(compact); datasetButtons->addWidget(archive); datasetButtons->addStretch();
        datasetLayout->addLayout(datasetButtons);
        auto *runBox = new QGroupBox("Trained models", library);
        auto *runLayout = new QVBoxLayout(runBox);
        runs_ = new QTreeWidget(runBox);
        runs_->setHeaderLabels({"Run", "Best epoch", "Validation"});
        runs_->header()->setSectionResizeMode(0, QHeaderView::Stretch);
        runs_->setRootIsDecorated(false); runs_->setAlternatingRowColors(true);
        runLayout->addWidget(runs_);
        auto *runButtons = new QHBoxLayout;
        export_ = new QPushButton("Export selected model…", runBox);
        auto *showRun = new QPushButton("Show files", runBox);
        auto *useModel = new QPushButton("Use selected model in project", runBox);
        useModel->setVisible(bool(projectSettings_));
        useModel->setToolTip("Select a trained model after comparing its depth on independent photos. Extraction uses this model until you choose another.");
        runButtons->addWidget(export_); runButtons->addWidget(useModel); runButtons->addWidget(showRun); runButtons->addStretch();
        runLayout->addLayout(runButtons);
        library->addWidget(datasetBox); library->addWidget(runBox);
        root->addWidget(library, 2);
        library->setMaximumHeight(170);
        connect(review, &QPushButton::clicked, this, [this] { reviewDataset(selectedPath(datasets_)); });
        connect(inspect, &QPushButton::clicked, this, [this] {
            const QString path = selectedPath(datasets_);
            if (!path.isEmpty()) startJob("Inspect dataset", {"inspect-dataset", path});
        });
        connect(showDataset, &QPushButton::clicked, this, [this] { showFolder(selectedPath(datasets_)); });
        connect(archive, &QPushButton::clicked, this, [this] { archiveDataset(); });
        connect(compact, &QPushButton::clicked, this, [this] {
            const QString source = selectedPath(datasets_); if (source.isEmpty()) return;
            const QString destination = QDir(workspace_->text()).filePath("datasets/" + QFileInfo(source).fileName() + "-compact-" + QDateTime::currentDateTime().toString("yyyyMMdd-HHmmss"));
            refreshAfter_ = true; startJob("Compact dataset", {"compact-dataset", source, "--output-dir", destination});
        });
        connect(showRun, &QPushButton::clicked, this, [this] {
            const QString path = selectedPath(runs_); if (!path.isEmpty()) showFolder(QFileInfo(path).absolutePath());
        });
        connect(export_, &QPushButton::clicked, this, [this] { exportModel(); });
        connect(useModel, &QPushButton::clicked, this, [this] {
            const QString checkpoint = selectedPath(runs_);
            if (checkpoint.isEmpty() || !projectSettings_) return;
            raftModel_->setText(checkpoint);
            projectSettings_->setValue("raft/member", "");
            saveSharedModelSettings();
            statusBar()->showMessage("Project model selected. Compare on independent photos before relying on its depth.", 8000);
        });

        tabs_ = new QTabWidget(central);
        tabs_->setMinimumHeight(380);
        buildDatasetTab(); buildReviewTab(); buildCollectionTab(); buildTrainingTab();
        connect(tabs_, &QTabWidget::currentChanged, this, [this](int index) { if (index == 1 && previewProcess_) reviewSelectionChanged(); });
        auto *globalGoal = new QHBoxLayout; globalGoal->addWidget(new QLabel("Main purpose", central)); globalGoal->addWidget(goal_, 1); globalGoal->addWidget(advanced_); root->insertLayout(2, globalGoal);
        if (datasetMode_) { tabs_->setTabVisible(2, false); tabs_->setTabVisible(4, false); runBox->hide(); }
        else { tabs_->setTabVisible(0, false); tabs_->setCurrentIndex(2); }
        root->addWidget(tabs_, 3);
        generate_->setIcon(IPDE::appIcon("datasets")); train_->setIcon(IPDE::appIcon("trainer"));
        export_->setIcon(IPDE::appIcon("trainer"));
        auto *notice = new QLabel("Teacher depth is an estimate, not measured ground truth. Capture groups are held out by default; scene validation requires correctly grouped independent scenes.", central);
        notice->setWordWrap(true);
        notice->setToolTip("Dataset checks help catch accidental omissions, altered files, and validation overlap. They do not detect intentional poisoning or establish permission, copyright, or content suitability for externally obtained data.");
        root->addWidget(notice);
        log_ = new QPlainTextEdit(central); log_->setReadOnly(true); log_->setMaximumBlockCount(1500);
        log_->setMaximumHeight(90); log_->setPlaceholderText("Progress and results appear here."); root->addWidget(log_);
        auto *progressRow = new QHBoxLayout;
        progress_ = new QProgressBar(central); progress_->setRange(0, 1); progress_->setValue(0); progress_->setTextVisible(false);
        cancel_ = new QPushButton("Cancel current task", central); cancel_->setEnabled(false);
        progressRow->addWidget(progress_, 1); progressRow->addWidget(cancel_); root->addLayout(progressRow);
        IPDE::setScrollableCentralWidget(this, central, QSize(1280, 900));

        process_ = new QProcess(this); process_->setProcessChannelMode(QProcess::SeparateChannels);
        connect(process_, &QProcess::readyReadStandardOutput, this, [this] { stdout_ += process_->readAllStandardOutput(); });
        connect(process_, &QProcess::readyReadStandardError, this, [this] { appendProgress(process_->readAllStandardError()); });
        connect(process_, qOverload<int, QProcess::ExitStatus>(&QProcess::finished), this,
            [this](int code, QProcess::ExitStatus status) { processFinished(code, status); });
        connect(process_, &QProcess::errorOccurred, this, [this](QProcess::ProcessError error) {
            if (error == QProcess::FailedToStart) {
                log_->appendPlainText("Could not launch Python: " + process_->errorString());
                setBusy(false); groupsFile_.reset(); teachersFile_.reset(); refreshAfter_ = false;
                if (job_ == "Generate dataset") stopGenerationStreaming("Generation could not start. Generate a new dataset to try again.");
                statusBar()->showMessage(job_ + " failed to start; see the progress log.");
                if (job_ == "Preview depth") previewStats_->setText("Preview failed: Python could not be launched.");
            }
        });
        connect(cancel_, &QPushButton::clicked, this, [this] {
            cancelled_ = true; process_->kill(); log_->appendPlainText("Cancellation requested; partial work will not be presented as complete.");
        });
        previewProcess_ = new QProcess(this);
        connect(previewProcess_, &QProcess::readyReadStandardOutput, this, [this] { previewStdout_ += previewProcess_->readAllStandardOutput(); });
        connect(previewProcess_, qOverload<int, QProcess::ExitStatus>(&QProcess::finished), this, [this](int code, QProcess::ExitStatus status) {
            previewStdout_ += previewProcess_->readAllStandardOutput();
            if (!pendingPreviewArgs_.isEmpty()) { const auto args = pendingPreviewArgs_; const auto label = pendingPreviewKind_; pendingPreviewArgs_.clear(); requestPreview(label, args); return; }
            if (previewKind_ == "discarded") return;
            const auto doc = QJsonDocument::fromJson(previewStdout_.trimmed());
            if (status == QProcess::NormalExit && code == 0 && doc.isObject()) {
                if (previewKind_ == "review") populateReview(doc.object()); else populatePreview(doc.object());
            } else if (previewKind_ != "review") previewStats_->setText("This preview could not be generated. Other photos and teachers remain available. " + QString::fromUtf8(previewStdout_).left(800));
        });
        connect(previewProcess_, &QProcess::errorOccurred, this, [this](QProcess::ProcessError error) {
            if (error == QProcess::FailedToStart) previewStats_->setText("Preview process could not start: " + previewProcess_->errorString());
        });
        streamingTimer_ = new QTimer(this); streamingTimer_->setSingleShot(true); streamingTimer_->setInterval(750);
        connect(streamingTimer_, &QTimer::timeout, this, [this] {
            if (!streamingDataset_.isEmpty()) reviewDataset(streamingDataset_, reviewedDataset_.isEmpty());
        });
        statusBar()->showMessage("Ready");
        if (!arguments.contains("--smoke-test") || arguments.contains("--screenshot")) {
            const int reviewArgument = arguments.indexOf("--review-dataset");
            if (reviewArgument >= 0 && reviewArgument + 1 < arguments.size())
                QTimer::singleShot(0, this, [this, arguments, reviewArgument] {
                    pendingReview_ = QFileInfo(arguments.at(reviewArgument + 1)).absoluteFilePath(); refreshLibrary();
                });
            else QTimer::singleShot(0, this, [this] { refreshLibrary(); });
        }
    }

    ~TrainerWindow() override {
        streamingTimer_->stop(); streamingTimer_->disconnect(this);
        process_->disconnect(this); previewProcess_->disconnect(this);
        settings_.setValue("workspace", workspace_->text());
        settings_.setValue("teacher_model", teacher_->currentData());
        settings_.setValue("raft_root", raftRoot_->text()); settings_.setValue("raft_model", raftModel_->text());
        for (auto *process : {process_, previewProcess_}) {
            if (process->state() != QProcess::NotRunning) {
                process->kill(); process->waitForFinished(1500);
            }
        }
    }

    bool taskRunning() const { return process_->state() != QProcess::NotRunning || previewProcess_->state() != QProcess::NotRunning; }
    void projectChanged() {
        if (projectSettings_) {
            projectSettings_->sync(); const QString goal = projectSettings_->value("goal", "effect/map").toString();
            setWindowTitle((datasetMode_ ? "Dataset Studio — " : "RAFT Studio — ")
                + projectSettings_->value("name", QFileInfo(projectRoot_).fileName()).toString());
            if (goal_->currentData().toString() != goal) { QSignalBlocker blocker(goal_); goal_->setCurrentIndex(qMax(0, goal_->findData(goal))); applyGoal(false); }
            if (!raftRoot_->hasFocus()) raftRoot_->setText(projectSettings_->value("raft/root", "/opt/ipde/RAFT-Stereo").toString());
            if (!raftModel_->hasFocus()) raftModel_->setText(projectSettings_->value("raft/model", "/opt/ipde/models/raftstereo-middlebury.pth").toString());
        }
        if (process_->state() == QProcess::NotRunning) refreshLibrary(); else refreshAfter_ = true;
    }

#ifdef IPDE_STUDIO_REGRESSION
public:
#else
private:
#endif
    static QString selectedPath(QTreeWidget *tree) {
        return tree->currentItem() ? tree->currentItem()->data(0, Qt::UserRole).toString() : QString();
    }

    static QComboBox *deviceBox(QWidget *parent) {
        auto *box = new QComboBox(parent); box->addItems({"auto", "mps", "cpu", "cuda"}); return box;
    }

    static QSpinBox *spin(QWidget *parent, int minimum, int maximum, int value) {
        auto *box = new QSpinBox(parent); box->setRange(minimum, maximum); box->setValue(value); return box;
    }

    QWidget *pathRow(QLineEdit *&edit, const QString &value, bool directory, const QString &caption, QWidget *parent) {
        auto *row = new QWidget(parent); auto *layout = new QHBoxLayout(row); layout->setContentsMargins(0, 0, 0, 0);
        edit = new QLineEdit(value, row); auto *browse = new QPushButton("Choose…", row);
        layout->addWidget(edit, 1); layout->addWidget(browse);
        connect(browse, &QPushButton::clicked, this, [this, edit, directory, caption] {
            const QString value = directory ? QFileDialog::getExistingDirectory(this, caption, edit->text())
                : QFileDialog::getOpenFileName(this, caption, edit->text(), "Model checkpoints (*.pth *.pt);;All files (*)");
            if (!value.isEmpty()) { edit->setText(value); if (edit == raftRoot_ || edit == raftModel_) saveSharedModelSettings(); }
        });
        return row;
    }

    void saveSharedModelSettings() {
        if (!projectSettings_ || !raftRoot_ || !raftModel_) return;
        projectSettings_->setValue("raft/root", raftRoot_->text()); projectSettings_->setValue("raft/model", raftModel_->text()); projectSettings_->sync();
    }

    bool addSourcePhoto(const QString &file, const QJsonObject &metadata = {}) {
        const QFileInfo info(file);
        const QString path = info.canonicalFilePath().isEmpty() ? info.absoluteFilePath() : info.canonicalFilePath();
        if (file.isEmpty()) return false;
        QTreeWidgetItem *item = nullptr;
        for (int i = 0; i < sources_->topLevelItemCount(); ++i) {
            auto *existing = sources_->topLevelItem(i);
            const QFileInfo existingInfo(existing->data(0, Qt::UserRole).toString());
            const QString existingPath = existingInfo.canonicalFilePath().isEmpty() ? existingInfo.absoluteFilePath() : existingInfo.canonicalFilePath();
            if (existingPath == path) { item = existing; break; }
        }
        const bool added = item == nullptr;
        if (added) {
            item = new QTreeWidgetItem(sources_, {info.fileName(), "", "", ""});
            item->setData(0, Qt::UserRole, path); item->setToolTip(0, path);
            item->setFlags(item->flags() | Qt::ItemIsEditable);
        }
        if (metadata.contains("camera_model")) item->setText(2, metadata.value("camera_model").toString());
        if (metadata.contains("captured_at")) item->setText(3, metadata.value("captured_at").toString());
        return added;
    }

    void buildDatasetTab() {
        auto *tab = new QWidget; auto *root = new QVBoxLayout(tab);
        auto *scroll = new QScrollArea; scroll->setWidgetResizable(true); scroll->setFrameShape(QFrame::NoFrame); scroll->setWidget(tab);
        auto *intro = new QLabel("A depth teacher is the AI model that generates estimated depth for training. Generate the dataset, then review its depth maps before training.", tab);
        intro->setWordWrap(true); root->addWidget(intro);
        goal_ = new QComboBox(tab);
        goal_->addItem("Effect / displacement map — prioritize detail", "effect/map");
        goal_->addItem("Depth estimation — prioritize distance", "depth-estimation");
        goal_->addItem("Portrait effects / masking — embedded depth and mattes", "photo-effects");
        goal_->addItem("Manual — expose every setting", "manual");
        advanced_ = new QCheckBox("Show advanced settings", tab);
        goalHelp_ = new QLabel(tab); goalHelp_->setWordWrap(true); root->addWidget(goalHelp_);
        auto *row = new QHBoxLayout;
        datasetName_ = new QLineEdit("dataset-" + QDateTime::currentDateTime().toString("yyyyMMdd-HHmmss"), tab);
        row->addWidget(new QLabel("New dataset", tab)); row->addWidget(datasetName_, 1);
        auto *add = new QPushButton("Add photos…", tab); auto *scan = new QPushButton("Find spatial photos in folder…", tab); auto *remove = new QPushButton("Remove selected", tab);
        row->addWidget(add); row->addWidget(scan); row->addWidget(remove); root->addLayout(row);
        auto *categoryRow = new QHBoxLayout; category_ = new QLineEdit(tab); category_->setPlaceholderText("Rooms, landscapes, macro, people…");
        categoryRow->addWidget(new QLabel("Subject / dataset category", tab)); categoryRow->addWidget(category_, 1);
        useGroups_ = new QCheckBox("Use photo groups", tab); categoryRow->addWidget(useGroups_); root->addLayout(categoryRow);
        sources_ = new QTreeWidget(tab); sources_->setHeaderLabels({"Spatial HEIC", "Optional group — double-click to edit", "Camera", "Captured"});
        sources_->setObjectName("datasetSources");
        sources_->setMinimumHeight(100); sources_->setMaximumHeight(110);
        sources_->setSelectionMode(QAbstractItemView::ExtendedSelection); sources_->setRootIsDecorated(false);
        sources_->header()->setSectionResizeMode(0, QHeaderView::Stretch); sources_->header()->setSectionResizeMode(1, QHeaderView::Stretch);
        sources_->setItemDelegate(new GroupDelegate(sources_)); root->addWidget(sources_, 1);
        verifiedScenes_ = new QCheckBox("My groups separate independent scenes", tab);
        verifiedScenes_->setToolTip("This is your declaration, not an automatic scene check. It labels the validation as scene-based. The same group labels keep related photos in one split whether checked or unchecked.");
        root->addWidget(verifiedScenes_); verifiedScenes_->setVisible(false);
        auto *sceneHelp = new QLabel("Give repeat shots of the same room, subject, or setup the same group name. Check this only after grouping every related photo together and confirming the other groups show different scenes. This keeps validation from benefiting from scenes seen during training. If unsure, leave it unchecked; different filenames alone are not evidence.", tab);
        sceneHelp->setWordWrap(true); root->addWidget(sceneHelp); sceneHelp->hide();
        connect(useGroups_, &QCheckBox::toggled, this, [this, sceneHelp](bool grouped) { sources_->setColumnHidden(1, !grouped); verifiedScenes_->setVisible(grouped); sceneHelp->setVisible(grouped); });
        sources_->setColumnHidden(1, true);
        connect(add, &QPushButton::clicked, this, [this] {
            const auto files = QFileDialog::getOpenFileNames(this, "Add spatial HEIC photos", settings_.value("photo_folder").toString(), "HEIC / HEIF photos (*.heic *.HEIC *.heif *.HEIF *.hif *.HIF)");
            for (const QString &file : files) addSourcePhoto(file);
            if (!files.isEmpty()) settings_.setValue("photo_folder", QFileInfo(files.first()).absolutePath());
        });
        connect(scan, &QPushButton::clicked, this, [this] {
            const QString directory = QFileDialog::getExistingDirectory(this, "Choose the folder containing your original HEIC photos; subfolders included", settings_.value("photo_folder").toString());
            if (!directory.isEmpty()) { settings_.setValue("photo_folder", directory); startJob("Scan spatial photos", {"scan-spatial", directory}); }
        });
        connect(remove, &QPushButton::clicked, this, [this] { qDeleteAll(sources_->selectedItems()); });
        auto *teachers = new QHBoxLayout; teachers->addWidget(new QLabel("Generate teacher entries", tab));
        for (const auto &choice : QList<QPair<QString, QString>>{{"DepthPro", "depthpro"}, {"Depth Anything V2", "depth-anything-v2"}, {"Depth Anything 3", "depth-anything-3"}}) {
            auto *check = new QCheckBox(choice.first, tab); check->setProperty("model", choice.second); check->setChecked(choice.second == "depthpro");
            teacherChecks_.append(check); teachers->addWidget(check);
        }
        teachers->addStretch(); root->addLayout(teachers);
        auto *teacherHint = new QLabel("Select up to three teachers. Each prediction becomes its own training entry; reviewing a photo lets you exclude an individual teacher.", tab); teacherHint->setWordWrap(true); root->addWidget(teacherHint);
        datasetAdvanced_ = new QGroupBox("Model locations and processing", tab); auto *advancedLayout = new QVBoxLayout(datasetAdvanced_);
        auto *controls = new QHBoxLayout; auto *left = new QFormLayout; auto *right = new QFormLayout;
        teacher_ = new QComboBox(tab);
        teacher_->addItem("Apple DepthPro — estimated meters", "depthpro");
        teacher_->addItem("Depth Anything V2 Large — relative inverse depth", "depth-anything-v2");
        teacher_->addItem("Depth Anything 3 Giant — relative depth", "depth-anything-3");
        left->addRow("Depth teacher", teacher_);
        left->addRow("Teacher checkpoint", pathRow(teacherPath_, "/opt/ipde/models/depth_pro.pt", false, "Choose teacher checkpoint", tab));
        right->addRow("Teacher source", pathRow(teacherSource_, "/opt/ipde/ml-depth-pro", true, "Choose model source", tab));
        auto *processing = new QWidget(tab); auto *processingLayout = new QHBoxLayout(processing); processingLayout->setContentsMargins(0, 0, 0, 0);
        teacherDevice_ = deviceBox(processing); inputSize_ = spin(processing, 14, 4096, 1036); inputSize_->setSingleStep(14);
        processingLayout->addWidget(teacherDevice_); processingLayout->addWidget(new QLabel("Input size", processing)); processingLayout->addWidget(inputSize_);
        right->addRow("Inference device", processing);
        anchor_ = new QCheckBox("Estimate meters with a second model (DepthPro)", tab);
        anchor_->setChecked(true);
        anchor_->setToolTip("Uses the separately installed Apple DepthPro model to estimate metric scale. Detail remains from the selected teacher. This adds inference and does not create measured ground truth.");
        left->addRow("Depth scale", anchor_);
        controls->addLayout(left, 1); controls->addLayout(right, 1); advancedLayout->addLayout(controls);
        auto *additionalForm = new QFormLayout;
        for (const auto &model : {QString("depthpro"), QString("depth-anything-v2"), QString("depth-anything-3")}) {
            const QString path = model == "depthpro" ? "/opt/ipde/models/depth_pro.pt" : model == "depth-anything-v2" ? "/opt/ipde/models/depth_anything_v2_vitl.pth" : "/opt/ipde/models/DA3-GIANT-1.1";
            const QString source = model == "depthpro" ? "/opt/ipde/ml-depth-pro" : model == "depth-anything-v2" ? "/opt/ipde/Depth-Anything-V2" : "/opt/ipde/Depth-Anything-3";
            QLineEdit *modelEdit = nullptr, *sourceEdit = nullptr;
            additionalForm->addRow(model + " checkpoint", pathRow(modelEdit, path, model == "depth-anything-3", "Choose " + model + " model", datasetAdvanced_));
            additionalForm->addRow(model + " source", pathRow(sourceEdit, source, true, "Choose " + model + " source", datasetAdvanced_));
            modelPaths_.insert(model, modelEdit); modelSources_.insert(model, sourceEdit);
        }
        advancedLayout->addLayout(additionalForm);
        includeDisplayTeacher_ = new QCheckBox("Also generate teachers on the separate full display camera (uses much more storage)", datasetAdvanced_);
        includeDisplayTeacher_->setChecked(false); advancedLayout->addWidget(includeDisplayTeacher_);
        scaleHelp_ = new QLabel(tab); scaleHelp_->setWordWrap(true); advancedLayout->addWidget(scaleHelp_);
        auto *sizeHelp = new QLabel("Input size is the model's processing resolution, not the saved depth precision. A larger size costs time and memory and may still produce incorrect geometry. Auto device selects available hardware; MPS uses Apple GPU, CPU is slower, and CUDA requires an NVIDIA GPU.", tab);
        sizeHelp->setWordWrap(true); advancedLayout->addWidget(sizeHelp); root->addWidget(datasetAdvanced_);
        connect(teacher_, &QComboBox::currentIndexChanged, this, [this] { teacherDefaults(); });
        connect(anchor_, &QCheckBox::toggled, this, [this] { updateScaleHelp(); });
        teacher_->setCurrentIndex(qMax(0, teacher_->findData(settings_.value("teacher_model", "depthpro"))));
        teacherDefaults();
        connect(goal_, &QComboBox::currentIndexChanged, this, [this] { applyGoal(); });
        connect(advanced_, &QCheckBox::toggled, this, [this](bool visible) { datasetAdvanced_->setVisible(visible); if (trainingAdvanced_) trainingAdvanced_->setVisible(visible); });
        const QString configuredGoal = projectSettings_ ? projectSettings_->value("goal", "effect/map").toString() : settings_.value("goal", "effect/map").toString();
        goal_->setCurrentIndex(qMax(0, goal_->findData(configuredGoal))); applyGoal();
        generate_ = new QPushButton("Generate & review dataset", tab); root->addWidget(generate_);
        connect(generate_, &QPushButton::clicked, this, [this] { generateDataset(); });
        tabs_->addTab(scroll, "1. Generate dataset");
    }

    void applyGoal(bool save = true) {
        const QString goal = goal_->currentData().toString();
        if (save) settings_.setValue("goal", goal);
        if (save && projectSettings_) { projectSettings_->setValue("goal", goal); projectSettings_->sync(); }
        if (goal == "manual") { advanced_->setChecked(true); goalHelp_->setText("Every inference, scale, and training setting is available for manual control."); }
        else {
            advanced_->setChecked(false); datasetAdvanced_->hide(); teacherDevice_->setCurrentIndex(0); anchor_->setChecked(true);
            const QString preferred = goal == "effect/map" ? "depth-anything-3" : "depthpro";
            teacher_->setCurrentIndex(teacher_->findData(preferred));
            for (auto *check : teacherChecks_) check->setChecked(check->property("model").toString() == preferred);
            goalHelp_->setText(goal == "effect/map" ? "Detail preset: Depth Anything 3 with DepthPro scale, native-resolution labels, and conservative RAFT fine-tuning. Compare teachers and the original RAFT model to decide which preserves useful detail." : goal == "depth-estimation" ? "Distance preset: DepthPro estimates meters directly. Independent photo groups stay together during validation; model estimates still require visual checking." : "Portrait preset: use Photo Studio for embedded depth and composited portrait mattes. Portraits do not have the calibrated stereo pair required for RAFT datasets; the scanner skips them.");
        }
        if (trainingAdvanced_) trainingAdvanced_->setVisible(advanced_->isChecked());
    }

    void updateScaleHelp() {
        if (teacher_->currentData().toString() == "depthpro") {
            scaleHelp_->setText("DepthPro already estimates distance in meters, so a second scale model is unnecessary. These are AI estimates; meter units do not establish physical accuracy.");
        } else {
            scaleHelp_->setText(QString("Relative depth tells you which surfaces are nearer or farther, with an arbitrary scale for each photo. %1 When enabled, DepthPro supplies an estimated meter scale while the selected teacher supplies detail. This adds inference time, inherits scale errors, and can be rejected when the models disagree. Original relative values are kept separately.")
                .arg(anchor_->isChecked() ? "Keep this enabled to train RAFT with V2 or DA3." : "With this off you can compare relative predictions, but RAFT training requires an accepted meter-scale target."));
        }
    }

    void teacherDefaults() {
        const QString model = teacher_->currentData().toString();
        const bool da3 = model == "depth-anything-3";
        teacherPath_->setText(da3 ? "/opt/ipde/models/DA3-GIANT-1.1" : model == "depthpro" ? "/opt/ipde/models/depth_pro.pt" : "/opt/ipde/models/depth_anything_v2_vitl.pth");
        teacherSource_->setText(da3 ? "/opt/ipde/Depth-Anything-3" : model == "depthpro" ? "/opt/ipde/ml-depth-pro" : "/opt/ipde/Depth-Anything-V2");
        // Replace the checkpoint browse behavior for DA3's directory format.
        auto *button = teacherPath_->parentWidget()->findChild<QPushButton *>();
        disconnect(button, nullptr, this, nullptr);
        connect(button, &QPushButton::clicked, this, [this, da3] {
            const QString path = da3 ? QFileDialog::getExistingDirectory(this, "Choose DA3 model directory", teacherPath_->text())
                : QFileDialog::getOpenFileName(this, "Choose teacher checkpoint", teacherPath_->text(), "Checkpoints (*.pth *.pt);;All files (*)");
            if (!path.isEmpty()) teacherPath_->setText(path);
        });
        inputSize_->setEnabled(model != "depthpro");
        anchor_->setEnabled(model != "depthpro");
        inputSize_->setToolTip(model == "depthpro" ? "DepthPro uses its fixed 1536×1536 native prediction grid." : "V2 uses the shortest side; DA3 uses the longest side. Higher values cost more memory and do not guarantee more accurate geometry.");
        updateScaleHelp();
    }

    void buildReviewTab() {
        auto *tab = new QWidget; auto *root = new QVBoxLayout(tab);
        reviewPath_ = new QLabel("Select a dataset above and click Review depth maps, or generate a new dataset.", tab);
        reviewPath_->setWordWrap(true); reviewPath_->setTextInteractionFlags(Qt::TextSelectableByMouse); root->addWidget(reviewPath_);
        auto *body = new QSplitter(Qt::Horizontal, tab);
        reviewSamples_ = new QTreeWidget(body);
        reviewSamples_->setHeaderLabels({"Include / photo / teacher", "Split", "Group", "Camera", "Captured"});
        reviewSamples_->setRootIsDecorated(true); reviewSamples_->setAlternatingRowColors(true);
        reviewSamples_->header()->setStretchLastSection(false);
        reviewSamples_->header()->setSectionResizeMode(QHeaderView::Interactive);
        reviewSamples_->setColumnWidth(0, 220); reviewSamples_->setColumnWidth(1, 85); reviewSamples_->setColumnWidth(2, 110); reviewSamples_->setColumnWidth(3, 130); reviewSamples_->setColumnWidth(4, 155);
        auto *previewScroll = new QScrollArea(body); previewScroll->setObjectName("reviewPreviewScroll");
        previewScroll->setWidgetResizable(true); previewScroll->setFrameShape(QFrame::NoFrame);
        auto *preview = new QWidget; preview->setObjectName("reviewPreviewContent");
        auto *previewRoot = new QVBoxLayout(preview);
        previewScroll->setWidget(preview);
        auto *labelRow = new QHBoxLayout; labelRow->addWidget(new QLabel("Depth to view", preview));
        reviewLabel_ = new QComboBox(preview); reviewLabel_->setMinimumWidth(0); labelRow->addWidget(reviewLabel_, 1);
        previewRoot->addLayout(labelRow);
        compareTeachers_ = new QCheckBox("Compare teachers for this photo", preview); compareTeachers_->setChecked(true);
        previewRoot->addWidget(compareTeachers_);
        auto *viewRow = new QHBoxLayout; viewRow->addWidget(new QLabel("Visual inspection", preview));
        visualView_ = new QComboBox(preview); visualView_->addItem("Depth map", "depth"); visualView_->addItem("50% overlay on photo", "overlay"); visualView_->addItem("Lit surface — drag to rotate", "surface"); visualView_->addItem("Teacher / baseline disagreement", "difference");
        viewRow->addWidget(visualView_, 1); previewRoot->addLayout(viewRow);
        auto *images = new QHBoxLayout;
        auto *rgbColumn = new QVBoxLayout; auto *depthColumn = new QVBoxLayout;
        auto *sourceTitle = new QLabel("Source view on the same grid", preview); sourceTitle->setWordWrap(true);
        rgbColumn->addWidget(sourceTitle);
        auto *firstDepthTitle = new QLabel("Generated depth", preview); firstDepthTitle->setWordWrap(true); depthTitles_.append(firstDepthTitle); depthColumn->addWidget(firstDepthTitle);
        rgbPreview_ = new DepthPreview(preview); depthPreview_ = new DepthPreview(preview);
        rgbPreview_->setObjectName("reviewSourcePreview"); depthPreview_->setObjectName("reviewDepthPreview");
        depthPreviews_.append(depthPreview_);
        rgbPreview_->reset("Choose a photo to review."); depthPreview_->reset("Depth appears here.");
        rgbColumn->addWidget(rgbPreview_, 1); depthColumn->addWidget(depthPreview_, 1);
        images->addLayout(rgbColumn, 1); images->addLayout(depthColumn, 1);
        for (int i=0; i<2; ++i) {
            auto *columnWidget = new QWidget(preview); auto *column = new QVBoxLayout(columnWidget); column->setContentsMargins(0, 0, 0, 0);
            auto *title = new QLabel("Teacher", columnWidget); title->setWordWrap(true); auto *image = new DepthPreview(columnWidget);
            image->setObjectName(QString("reviewTeacherPreview%1").arg(i + 1));
            image->onVisibility = [columnWidget](bool visible) { columnWidget->setVisible(visible); };
            column->addWidget(title); column->addWidget(image, 1); images->addWidget(columnWidget, 1);
            depthTitles_.append(title); depthPreviews_.append(image); title->hide(); image->hide();
        }
        previewRoot->addLayout(images, 1);
        QList<DepthPreview *> linked{rgbPreview_}; linked.append(depthPreviews_);
        for (auto *image : linked) {
            image->onHover = [linked, image](const QPointF &position) { for (auto *other : linked) if (other != image) other->showMagnifier(position); };
            image->onRotate = [linked, image](double yaw, double pitch) { for (auto *other : linked) if (other != image) other->setAngles(yaw, pitch); };
        }
        previewStats_ = new QLabel(preview); previewStats_->setObjectName("reviewPreviewStats"); previewStats_->setWordWrap(true);
        previewStats_->setSizePolicy(QSizePolicy::Preferred, QSizePolicy::Minimum); previewRoot->addWidget(previewStats_);
        auto *legend = new QLabel("White = nearer · Black = farther · Magenta = invalid. Hover for synchronized 1:1 detail. Click an image to open native pixels centered on that point; drag to pan. Right-click or click outside the popup to close. Metric comparisons share contrast; relative views use separate ranges. Overlay, relief and disagreement are display copies; saved values stay unchanged.", preview);
        legend->setObjectName("reviewPreviewLegend"); legend->setSizePolicy(QSizePolicy::Preferred, QSizePolicy::Minimum);
        legend->setWordWrap(true); previewRoot->addWidget(legend);
        body->addWidget(reviewSamples_); body->addWidget(previewScroll); body->setChildrenCollapsible(false); body->setSizes({330, 860}); root->addWidget(body, 1);
        auto *filterRow = new QHBoxLayout; reviewFilter_ = new QLineEdit(tab); reviewFilter_->setPlaceholderText("Filter photo, teacher, group, or date…");
        reviewCamera_ = new QComboBox(tab); reviewCamera_->addItem("All cameras"); filterRow->addWidget(reviewFilter_, 1); filterRow->addWidget(reviewCamera_); root->addLayout(filterRow);
        auto *navigation = new QHBoxLayout;
        auto *previous = new QPushButton("Previous", tab); auto *next = new QPushButton("Next", tab);
        auto *exclude = new QPushButton("Exclude & next", tab);
        navigation->addWidget(previous); navigation->addWidget(next); navigation->addWidget(exclude);
        reviewCount_ = new QLabel("No dataset loaded", tab); reviewCount_->setWordWrap(true); navigation->addWidget(reviewCount_, 1); root->addLayout(navigation);
        auto *saveRow = new QHBoxLayout; saveRow->addWidget(new QLabel("Reviewed dataset name", tab));
        reviewedName_ = new QLineEdit(tab); saveRow->addWidget(reviewedName_, 1);
        saveReviewed_ = new QPushButton("Save reviewed copy", tab); saveReviewed_->setEnabled(false); saveRow->addWidget(saveReviewed_); root->addLayout(saveRow);
        connect(reviewSamples_, &QTreeWidget::currentItemChanged, this, [this] { reviewSelectionChanged(); });
        connect(reviewSamples_, &QTreeWidget::itemChanged, this, [this] { updateReviewCount(); });
        connect(reviewLabel_, &QComboBox::currentIndexChanged, this, [this] { previewSelectedSample(); });
        connect(compareTeachers_, &QCheckBox::toggled, this, [this] { previewSelectedSample(); });
        connect(visualView_, &QComboBox::currentIndexChanged, this, [this] { updateVisualView(); });
        connect(reviewFilter_, &QLineEdit::textChanged, this, [this] { filterReview(); });
        connect(reviewCamera_, &QComboBox::currentIndexChanged, this, [this] { filterReview(); });
        connect(previous, &QPushButton::clicked, this, [this] { advanceReview(-1); });
        connect(next, &QPushButton::clicked, this, [this] { advanceReview(1); });
        connect(exclude, &QPushButton::clicked, this, [this] {
            if (auto *item = reviewSamples_->currentItem()) { item->setCheckState(0, Qt::Unchecked); advanceReview(1); }
        });
        connect(saveReviewed_, &QPushButton::clicked, this, [this] { saveReviewedDataset(); });
        tabs_->addTab(tab, "2. Review depth maps");
    }

    void reviewDataset(const QString &path, bool openTab = true) {
        if (path.isEmpty()) return;
        requestedReviewPath_ = path;
        requestedReviewOpen_ = openTab;
        requestPreview("review", {"review-dataset", path});
    }

    QList<QTreeWidgetItem *> reviewEntries() const {
        QList<QTreeWidgetItem *> result;
        for (int i=0; i<reviewSamples_->topLevelItemCount(); ++i) {
            auto *photo = reviewSamples_->topLevelItem(i);
            for (int j=0; j<photo->childCount(); ++j) result.append(photo->child(j));
        }
        return result;
    }

    QTreeWidgetItem *selectedReviewEntry() const {
        auto *item = reviewSamples_->currentItem();
        return item && item->childCount() ? item->child(0) : item;
    }

    void filterReview() {
        const QString query = reviewFilter_->text().trimmed(), camera = reviewCamera_->currentIndex() > 0 ? reviewCamera_->currentText() : QString();
        for (int i=0; i<reviewSamples_->topLevelItemCount(); ++i) {
            auto *item = reviewSamples_->topLevelItem(i); QString text;
            for (int c=0; c<item->columnCount(); ++c) text += item->text(c) + " ";
            for (int j=0; j<item->childCount(); ++j) text += item->child(j)->text(0) + " ";
            item->setHidden((!query.isEmpty() && !text.contains(query, Qt::CaseInsensitive)) || (!camera.isEmpty() && item->text(3) != camera));
        }
    }

    void populateReview(const QJsonObject &result) {
        QMap<QString, Qt::CheckState> previousChecks;
        QString selectedId;
        if (auto *item = selectedReviewEntry()) selectedId = item->data(0, Qt::UserRole).toJsonObject().value("id").toString();
        for (auto *item : reviewEntries()) previousChecks.insert(item->data(0, Qt::UserRole).toJsonObject().value("id").toString(), item->checkState(0));
        reviewedDataset_ = result.value("dataset_path").toString(requestedReviewPath_);
        reviewGenerating_ = result.value("generation_state").toString() == "generating" || result.value("splits_provisional").toBool();
        reviewPath_->setText((reviewGenerating_ ? "Generating — review ready entries; save after completion: " : "Reviewing: ") + reviewedDataset_);
        reviewedName_->setText(QFileInfo(reviewedDataset_).fileName() + "-reviewed");
        if (!reviewPreviews_) reviewPreviews_ = std::make_unique<QTemporaryDir>();
        QTreeWidgetItem *selected = nullptr; QMap<QString, QTreeWidgetItem *> photos; QStringList cameras;
        {
            QSignalBlocker blocker(reviewSamples_); reviewSamples_->clear();
            for (const auto &value : result.value("samples").toArray()) {
                const auto sample = value.toObject(); const QString path = sample.value("source_path").toString();
                QString group = sample.value("requested_group").toString();
                if (group.isEmpty()) group = sample.value("group_id").toString().left(12);
                const auto metadata = sample.value("photo_metadata").toObject(); const QString camera = metadata.value("camera_model").toString();
                const QString sourceId = sample.value("source_id").toString(path); QTreeWidgetItem *photo = photos.value(sourceId);
                if (!photo) {
                    photo = new QTreeWidgetItem(reviewSamples_, {QFileInfo(path).fileName(), sample.value("split").toString(), group, camera, metadata.value("captured_at").toString()});
                    photo->setFlags(photo->flags() | Qt::ItemIsAutoTristate); photo->setCheckState(0, Qt::Checked); photo->setToolTip(0, path); photos.insert(sourceId, photo); photo->setExpanded(true);
                    if (!camera.isEmpty() && !cameras.contains(camera)) cameras << camera;
                }
                auto *item = new QTreeWidgetItem(photo, {sample.value("teacher_id").toString("Teacher"), sample.value("split").toString(), group});
                item->setData(0, Qt::UserRole, sample); item->setToolTip(0, path); item->setCheckState(0, previousChecks.value(sample.value("id").toString(), Qt::Checked));
                if (sample.value("id").toString() == selectedId) selected = item;
                QStringList warnings; for (const auto &warning : sample.value("warnings").toArray()) warnings << warning.toString();
                item->setToolTip(1, warnings.join('\n')); item->setToolTip(2, sample.value("group_id").toString());
            }
        }
        { QSignalBlocker blocker(reviewCamera_); const QString camera = reviewCamera_->currentText(); reviewCamera_->clear(); reviewCamera_->addItem("All cameras"); cameras.sort(); reviewCamera_->addItems(cameras); reviewCamera_->setCurrentIndex(qMax(0, reviewCamera_->findText(camera))); }
        filterReview();
        if (requestedReviewOpen_) tabs_->setCurrentIndex(1); updateReviewCount();
        if (reviewSamples_->topLevelItemCount()) {
            QSignalBlocker blocker(reviewSamples_); reviewSamples_->setCurrentItem(selected ? selected : reviewSamples_->topLevelItem(0));
            if (tabs_->currentIndex() == 1) QTimer::singleShot(0, this, [this] { reviewSelectionChanged(); });
        }
        for (const auto &warning : result.value("warnings").toArray()) log_->appendPlainText(warning.toString());
    }

    void reviewSelectionChanged() {
        rgbPreview_->reset("Loading source view…"); depthPreview_->reset("Loading generated depth…"); previewStats_->clear();
        {
            QSignalBlocker blocker(reviewLabel_); reviewLabel_->clear();
            if (auto *item = selectedReviewEntry()) {
                const auto sample = item->data(0, Qt::UserRole).toJsonObject();
                for (const auto &value : sample.value("labels").toArray()) {
                    const auto label = value.toObject(); reviewLabel_->addItem(label.value("title").toString() + " (" + label.value("units").toString() + ")", label.value("key").toString());
                }
            }
        }
        previewSelectedSample();
    }

    void previewSelectedSample() {
        auto *item = selectedReviewEntry();
        if (!item || reviewLabel_->currentIndex() < 0) return;
        rgbPreview_->reset("Loading source view…"); depthPreview_->reset("Loading generated depth…"); previewStats_->clear();
        if (!reviewPreviews_ || !reviewPreviews_->isValid()) {
            previewStats_->setText("Could not create a temporary folder for display previews."); return;
        }
        const QString id = item->data(0, Qt::UserRole).toJsonObject().value("id").toString();
        // Only the Python process constructs file names from manifest IDs.
        auto *photo = item->parent();
        if (compareTeachers_->isChecked() && reviewLabel_->currentData().toString() == "training" && photo && photo->childCount() >= 2) {
            QStringList args{"compare-samples", reviewedDataset_};
            for (int i=0; i<qMin(3, photo->childCount()); ++i) args << "--sample" << photo->child(i)->data(0, Qt::UserRole).toJsonObject().value("id").toString();
            args << "--output-dir" << reviewPreviews_->path() << "--max-dimension" << "0"; requestPreview("compare", args);
        } else requestPreview("preview", {"preview-sample", reviewedDataset_, "--sample", id, "--label", reviewLabel_->currentData().toString(), "--output-dir", reviewPreviews_->path(), "--max-dimension", "0"});
    }

    void requestPreview(const QString &kind, const QStringList &args) {
        if (previewProcess_->state() != QProcess::NotRunning) { pendingPreviewKind_ = kind; pendingPreviewArgs_ = args; previewProcess_->kill(); return; }
        previewKind_ = kind; previewStdout_.clear(); previewProcess_->setProgram(pythonPath()); previewProcess_->setArguments(QStringList{scriptPath(), "--json"} + args); previewProcess_->start();
    }

    void populatePreview(const QJsonObject &result) {
        QJsonArray previews = result.value("samples").toArray();
        if (previews.isEmpty()) previews = result.value("previews").toArray();
        if (previews.isEmpty()) previews.append(result);
        previewRecords_ = previews; differencePath_ = result.value("difference_preview_path").toString();
        const auto first = previews.first().toObject(); const bool rgbLoaded = rgbPreview_->load(first.value("rgb_preview_path").toString());
        bool depthLoaded = true;
        for (int i=0; i<depthPreviews_.size(); ++i) {
            if (i >= previews.size()) { depthPreviews_[i]->hide(); depthTitles_[i]->hide(); continue; }
            const auto record = previews[i].toObject(); depthLoaded &= depthPreviews_[i]->load(record.value("depth_preview_path").toString());
            depthPreviews_[i]->setSource(record.value("rgb_preview_path").toString()); depthPreviews_[i]->setSurface(record.value("surface").toObject());
            depthTitles_[i]->setText(record.value("teacher_id").toString(record.value("label_title").toString("Depth")));
            depthPreviews_[i]->show(); depthTitles_[i]->show();
        }
        QString text = QString("%1 · %2 × %3 samples · %4% valid · range %5 to %6 %7")
            .arg(first.value("label_title").toString()).arg(first.value("width").toInt()).arg(first.value("height").toInt())
            .arg(first.value("valid_fraction").toDouble() * 100, 0, 'f', 1)
            .arg(first.value("min").isDouble() ? QString::number(first.value("min").toDouble(), 'g', 7) : "none")
            .arg(first.value("max").isDouble() ? QString::number(first.value("max").toDouble(), 'g', 7) : "none")
            .arg(first.value("units").toString());
        if (result.value("comparison").isObject()) text += "\n" + result.value("comparison").toObject().value("legend").toString();
        if (!rgbLoaded || !depthLoaded) text += " · Could not load a preview.";
        if (first.value("valid_fraction").toDouble() == 0) text += "\nNo valid depth samples. Exclude this teacher entry before training.";
        if (auto *item = previewKind_ == "baseline" ? nullptr : selectedReviewEntry()) {
            const auto sample = item->data(0, Qt::UserRole).toJsonObject();
            if (!sample.value("training_ready").toBool()) text += "\nNo usable meter-scale training target. Review the anchor or use another teacher before training.";
        }
        previewStats_->setText(text); updateVisualView();
        if (previewKind_ == "baseline") tabs_->setCurrentIndex(1);
    }

    void updateVisualView() {
        const QString view = visualView_->currentData().toString();
        for (int i=0; i<depthPreviews_.size(); ++i) {
            const bool visible = i < previewRecords_.size() && (view != "difference" || i == 0);
            depthPreviews_[i]->setVisible(visible); depthTitles_[i]->setVisible(visible);
            if (!visible) continue;
            const auto record = previewRecords_[i].toObject();
            if (view == "difference") {
                depthPreviews_[i]->setView("depth");
                if (differencePath_.isEmpty()) depthPreviews_[i]->reset("Choose a photo with multiple teachers, or compare a trained model with its baseline.");
                else depthPreviews_[i]->load(differencePath_);
                depthTitles_[i]->setText("Disagreement — first two predictions");
            } else {
                depthPreviews_[i]->load(record.value("depth_preview_path").toString()); depthPreviews_[i]->setView(view);
                depthTitles_[i]->setText(record.value("teacher_id").toString(record.value("label_title").toString("Depth")));
            }
        }
    }

    void advanceReview(int direction) {
        const auto entries = reviewEntries(); const int index = entries.indexOf(selectedReviewEntry()) + direction;
        if (index >= 0 && index < entries.size()) reviewSamples_->setCurrentItem(entries[index]);
    }

    void updateReviewCount() {
        int kept = 0, train = 0, validation = 0;
        const auto entries = reviewEntries();
        for (auto *item : entries) {
            if (item->checkState(0) != Qt::Checked) continue;
            ++kept; train += item->text(1) == "train"; validation += item->text(1) == "validation";
        }
        QString text = QString("%1 of %2 teacher entries included · %3 train / %4 validation").arg(kept).arg(entries.size()).arg(train).arg(validation);
        if (kept && (!train || !validation)) text += " · Training needs samples in both splits; use more held-out groups in a new dataset.";
        reviewCount_->setText(text); saveReviewed_->setEnabled(kept > 0 && !reviewGenerating_ && (!process_ || process_->state() == QProcess::NotRunning));
        updateCollectionReadiness();
    }

    void saveReviewedDataset() {
        const QString name = reviewedName_->text().trimmed();
        if (!validName(name)) { QMessageBox::information(this, "Reviewed dataset name", "Use a folder name without path separators."); return; }
        QStringList args{"curate-dataset", reviewedDataset_, "--output-dir", QDir(workspace_->text()).filePath("datasets/" + name)};
        int kept = 0;
        for (auto *item : reviewEntries()) {
            if (item->checkState(0) != Qt::Checked) continue;
            args << "--keep" << item->data(0, Qt::UserRole).toJsonObject().value("id").toString(); ++kept;
        }
        if (!kept) return;
        refreshAfter_ = true; startJob("Save reviewed dataset", args);
    }

    void buildCollectionTab() {
        auto *tab = new QWidget; auto *root = new QVBoxLayout(tab);
        auto *scroll = new QScrollArea; scroll->setWidgetResizable(true); scroll->setFrameShape(QFrame::NoFrame); scroll->setWidget(tab);
        auto *intro = new QLabel("Build a training set from any number of reviewed datasets. Teacher variants of a photo stay in the same split. Group handling can be overridden for A/B experiments.", tab); intro->setWordWrap(true); root->addWidget(intro);
        collectionSources_ = new QTreeWidget(tab); collectionSources_->setHeaderLabels({"Use for training", "Use as validation", "Category", "Teacher entries"});
        collectionSources_->setRootIsDecorated(false); collectionSources_->header()->setSectionResizeMode(0, QHeaderView::Stretch); collectionSources_->setMinimumHeight(140); root->addWidget(collectionSources_, 1);
        connect(collectionSources_, &QTreeWidget::itemChanged, this, [this](QTreeWidgetItem *item, int column) {
            if (column < 2 && item->checkState(column) == Qt::Checked) { QSignalBlocker blocker(collectionSources_); item->setCheckState(1-column, Qt::Unchecked); }
            updateCollectionReadiness();
        });
        auto *form = new QFormLayout;
        collectionName_ = new QLineEdit("training-set-" + QDateTime::currentDateTime().toString("yyyyMMdd-HHmmss"), tab); form->addRow("Training set name", collectionName_);
        form->setFieldGrowthPolicy(QFormLayout::AllNonFixedFieldsGrow);
        splitMode_ = new QComboBox(tab); splitMode_->addItem("Random across all selected datasets", "global-random"); splitMode_->addItem("Same number chosen from each dataset", "equal-per-dataset"); splitMode_->addItem("Use designated validation datasets", "explicit");
        form->addRow("Validation strategy", splitMode_);
        validationFraction_ = new QDoubleSpinBox(tab); validationFraction_->setRange(.01, .9); validationFraction_->setSingleStep(.05); validationFraction_->setValue(.2); validationFraction_->setSuffix(" of photo groups"); form->addRow("Validation fraction", validationFraction_);
        validationCount_ = spin(tab, 0, 100000, 0); validationCount_->setSpecialValueText("Automatic equal count"); form->addRow("Equal validation groups per dataset", validationCount_);
        groupingPolicy_ = new QComboBox(tab); groupingPolicy_->addItem("Keep photo groups together", "preserve"); groupingPolicy_->addItem("Treat as one pile — ignore authored groups", "ignore"); form->addRow("Groups", groupingPolicy_);
        splitSeed_ = spin(tab, 0, 2147483647, 42); form->addRow("Repeatable random seed", splitSeed_); root->addLayout(form);
        splitHelp_ = new QLabel(tab); splitHelp_->setWordWrap(true); root->addWidget(splitHelp_);
        auto updateSplit = [this] {
            const QString mode = splitMode_->currentData().toString(); validationFraction_->setEnabled(mode != "explicit"); validationCount_->setEnabled(mode == "equal-per-dataset");
            splitHelp_->setText(mode == "explicit" ? "Tick dedicated validation datasets in the second column. They are held out entirely; overlapping source photos and groups are rejected." : mode == "equal-per-dataset" ? "Each dataset contributes the same count of randomly chosen independent photo groups. Zero uses an automatic count based on the smallest dataset. Related captures and all teachers remain together." : "Validation groups are drawn randomly from the combined selected datasets. Larger datasets usually contribute more validation photos. The seed repeats the same selection.");
        };
        connect(splitMode_, &QComboBox::currentIndexChanged, this, [this, updateSplit] { updateSplit(); updateCollectionReadiness(); }); updateSplit();
        collectionStatus_ = new QLabel(tab); collectionStatus_->setWordWrap(true); root->addWidget(collectionStatus_);
        compose_ = new QPushButton("Create training set & continue", tab); root->addWidget(compose_);
        connect(collectionName_, &QLineEdit::textChanged, this, [this] { updateCollectionReadiness(); });
        updateCollectionReadiness();
        connect(compose_, &QPushButton::clicked, this, [this] {
            updateCollectionReadiness(); if (!compose_->isEnabled()) return;
            const QString name = collectionName_->text().trimmed(); if (!validName(name)) { QMessageBox::information(this, "Training set name", "Use a folder name without separators."); return; }
            QStringList inputs, validation;
            for (int i=0; i<collectionSources_->topLevelItemCount(); ++i) { auto *item = collectionSources_->topLevelItem(i); if (item->checkState(0) == Qt::Checked) inputs << item->data(0, Qt::UserRole).toString(); if (item->checkState(1) == Qt::Checked) validation << item->data(0, Qt::UserRole).toString(); }
            if (inputs.isEmpty()) { QMessageBox::information(this, "Select datasets", "Tick one or more datasets for training."); return; }
            const QString mode = splitMode_->currentData().toString();
            if (mode == "explicit" && validation.isEmpty()) { QMessageBox::information(this, "Select validation", "Tick a dedicated validation dataset in the second column."); return; }
            QStringList args{"compose-datasets"}; args << inputs << "--output-dir" << QDir(workspace_->text()).filePath("datasets/" + name) << "--split-mode" << mode << "--validation-fraction" << QString::number(validationFraction_->value()) << "--seed" << QString::number(splitSeed_->value()) << "--grouping" << groupingPolicy_->currentData().toString();
            if (mode == "explicit") for (const QString &path : validation) args << "--validation-dataset" << path;
            if (mode == "equal-per-dataset" && validationCount_->value()) args << "--validation-count-per-dataset" << QString::number(validationCount_->value());
            refreshAfter_ = true; startJob("Create training set", args);
        });
        auto *hfTab = new QWidget; auto *hfRoot = new QVBoxLayout(hfTab); auto *hfScroll = new QScrollArea; hfScroll->setWidgetResizable(true); hfScroll->setFrameShape(QFrame::NoFrame); hfScroll->setWidget(hfTab);
        auto *hf = new QGroupBox("Optional Hugging Face import", hfTab); auto *hfForm = new QFormLayout(hf);
        hfSource_ = new QLineEdit(hf); hfSource_->setPlaceholderText("Hub dataset name or local dataset path"); hfForm->addRow("Dataset", hfSource_);
        hfConfig_ = new QLineEdit(hf); hfForm->addRow("Configuration (optional)", hfConfig_);
        hfSplit_ = new QLineEdit("train", hf); hfForm->addRow("Source split", hfSplit_);
        hfDataFiles_ = new QLineEdit(hf); hfDataFiles_->setPlaceholderText("Optional local JSON, JSONL, CSV or Parquet file for a datasets loader");
        auto *dataRow = new QWidget(hf); auto *dataLayout = new QHBoxLayout(dataRow); dataLayout->setContentsMargins(0,0,0,0); dataLayout->addWidget(hfDataFiles_, 1); auto *dataBrowse = new QPushButton("Choose local data…", hf); dataLayout->addWidget(dataBrowse); hfForm->addRow("Local data file (optional)", dataRow);
        connect(dataBrowse, &QPushButton::clicked, this, [this] { const QString path = QFileDialog::getOpenFileName(this, "Select a local dataset table", {}, "Dataset tables (*.json *.jsonl *.csv *.parquet);;All files (*)"); if (!path.isEmpty()) hfDataFiles_->setText(path); });
        hfForm->addRow("Array asset folder (optional)", pathRow(hfAssetDir_, {}, true, "Select the folder containing NPY/NPZ stereo arrays", hf));
        hfMapping_ = new QLineEdit(hf); auto *mappingRow = new QWidget(hf); auto *mappingLayout = new QHBoxLayout(mappingRow); mappingLayout->setContentsMargins(0,0,0,0); mappingLayout->addWidget(hfMapping_, 1); auto *mappingBrowse = new QPushButton("Choose mapping JSON…", hf); mappingLayout->addWidget(mappingBrowse); hfForm->addRow("Columns and calibration", mappingRow);
        connect(mappingBrowse, &QPushButton::clicked, this, [this] { const QString path = QFileDialog::getOpenFileName(this, "Select stereo dataset mapping", {}, "JSON (*.json)"); if (!path.isEmpty()) hfMapping_->setText(path); });
        auto *hfHelp = new QLabel("Imports calibrated stereo data through the datasets Python library. The mapping must identify left/right RGB, meter depth, focal length, baseline and source identity. Arbitrary single-image datasets cannot train metric RAFT stereo.", hf); hfHelp->setWordWrap(true); hfForm->addRow(hfHelp);
        importHf_ = new QPushButton("Import dataset", hf); hfForm->addRow(importHf_);
        connect(importHf_, &QPushButton::clicked, this, [this] {
            if (hfSource_->text().trimmed().isEmpty() || !QFileInfo::exists(hfMapping_->text())) { QMessageBox::information(this, "Hugging Face import", "Enter a dataset and choose a column mapping JSON file."); return; }
            const QString destination = QDir(workspace_->text()).filePath("datasets/hf-" + QDateTime::currentDateTime().toString("yyyyMMdd-HHmmss"));
            QStringList args{"import-hf", hfSource_->text().trimmed(), "--output-dir", destination, "--mapping-json", hfMapping_->text(), "--split", hfSplit_->text().trimmed()};
            if (!hfConfig_->text().trimmed().isEmpty()) args << "--config" << hfConfig_->text().trimmed();
            if (!hfDataFiles_->text().trimmed().isEmpty()) args << "--data-files" << hfDataFiles_->text().trimmed();
            if (!hfAssetDir_->text().trimmed().isEmpty()) args << "--asset-dir" << hfAssetDir_->text().trimmed();
            refreshAfter_ = true; startJob("Import Hugging Face dataset", args);
        });
        root->addStretch(); hfRoot->addWidget(hf); hfRoot->addStretch(); tabs_->addTab(scroll, "3. Training set"); tabs_->addTab(hfScroll, "Import datasets");
    }

    void buildTrainingTab() {
        auto *tab = new QWidget; auto *root = new QVBoxLayout(tab);
        auto *scroll = new QScrollArea; scroll->setWidgetResizable(true); scroll->setFrameShape(QFrame::NoFrame); scroll->setWidget(tab);
        auto *instruction = new QLabel("Create a training set in the previous step, or select an existing one above. After training, compare an unseen spatial photo against the original RAFT model.", tab);
        instruction->setWordWrap(true); root->addWidget(instruction);
        auto *form = new QFormLayout;
        runName_ = new QLineEdit("run-" + QDateTime::currentDateTime().toString("yyyyMMdd-HHmmss"), tab); form->addRow("New run", runName_);
        form->addRow("RAFT source", pathRow(raftRoot_, projectSettings_ ? projectSettings_->value("raft/root", "/opt/ipde/RAFT-Stereo").toString() : settings_.value("raft_root", "/opt/ipde/RAFT-Stereo").toString(), true, "Choose RAFT-Stereo source", tab));
        form->addRow("Initial RAFT model", pathRow(raftModel_, projectSettings_ ? projectSettings_->value("raft/model", "/opt/ipde/models/raftstereo-middlebury.pth").toString() : settings_.value("raft_model", "/opt/ipde/models/raftstereo-middlebury.pth").toString(), false, "Choose original RAFT checkpoint", tab));
        connect(raftRoot_, &QLineEdit::editingFinished, this, [this] { saveSharedModelSettings(); }); connect(raftModel_, &QLineEdit::editingFinished, this, [this] { saveSharedModelSettings(); });
        root->addLayout(form);
        trainingAdvanced_ = new QGroupBox("Advanced training settings", tab); auto *trainingAdvancedLayout = new QVBoxLayout(trainingAdvanced_); auto *options = new QHBoxLayout;
        epochs_ = spin(tab, 1, 10000, 10); steps_ = spin(tab, 1, 10000, 16); patch_ = spin(tab, 64, 2048, 256); patch_->setSingleStep(32);
        iterations_ = spin(tab, 1, 256, 4); scope_ = new QComboBox(tab); scope_->addItem("Update block", "update"); scope_->addItem("Full network", "full"); trainDevice_ = deviceBox(tab);
        epochs_->setToolTip("Number of rounds of updates and held-out validation. More rounds can overfit the teacher's errors.");
        steps_->setToolTip("Optimizer updates per epoch. Each update draws a photo and crop; an epoch is not necessarily a pass through all photos.");
        patch_->setToolTip("Width and height of each native-resolution training crop. Must be a multiple of 32. Larger crops require more memory.");
        iterations_->setToolTip("RAFT refinement passes per training update and validation prediction. More passes cost computation.");
        scope_->setToolTip("Update block changes the refinement module, a useful starting point. Full network changes all weights and needs more memory and diverse data.");
        trainDevice_->setToolTip("Auto selects available hardware. MPS is the Apple GPU; CUDA needs an NVIDIA GPU; CPU uses the processor.");
        for (auto pair : {qMakePair(QString("Epochs"), epochs_), qMakePair(QString("Steps / epoch"), steps_), qMakePair(QString("Patch pixels"), patch_), qMakePair(QString("RAFT iterations"), iterations_)}) {
            auto *group = new QFormLayout; group->addRow(pair.first, pair.second); options->addLayout(group);
        }
        auto *deviceForm = new QFormLayout; deviceForm->addRow("Train scope", scope_); deviceForm->addRow("Device", trainDevice_); options->addLayout(deviceForm);
        trainingAdvancedLayout->addLayout(options);
        trainingMode_ = new QComboBox(tab); trainingMode_->addItem("Automatic from dataset labels", "auto"); trainingMode_->addItem("Learn teacher estimates (distillation)", "distillation"); trainingMode_->addItem("Learn measured references (supervised)", "supervised");
        auto *trainingModeForm = new QFormLayout; trainingModeForm->addRow("Training labels", trainingMode_); trainingAdvancedLayout->addLayout(trainingModeForm);
        auto *trainingHelp = new QLabel("Epochs × steps sets the number of training updates. Each update uses a native-resolution crop (Patch pixels); RAFT iterations sets how often its prediction is refined. Update block is a useful starting scope; Full network adjusts all weights. More training can also learn the teacher's mistakes. Hover over a control for details.", tab);
        trainingHelp->setWordWrap(true); trainingAdvancedLayout->addWidget(trainingHelp); root->addWidget(trainingAdvanced_); trainingAdvanced_->setVisible(advanced_->isChecked()); root->addStretch();
        train_ = new QPushButton("Train selected dataset", tab); root->addWidget(train_);
        connect(train_, &QPushButton::clicked, this, [this] { trainDataset(); });
        compareBaseline_ = new QPushButton("Compare selected trained model with baseline on a photo…", tab); root->addWidget(compareBaseline_);
        connect(compareBaseline_, &QPushButton::clicked, this, [this] {
            const QString candidate = selectedPath(runs_); if (candidate.isEmpty()) { QMessageBox::information(this, "Select trained model", "Select a run from the trained models library."); return; }
            const QString photo = QFileDialog::getOpenFileName(this, "Compare on a new calibrated spatial photo", {}, "HEIC photos (*.heic *.HEIC *.heif *.HEIF)"); if (photo.isEmpty()) return;
            if (!reviewPreviews_) reviewPreviews_ = std::make_unique<QTemporaryDir>();
            const QString comparisonOutput = QDir(reviewPreviews_->path()).filePath("model-compare-" + QDateTime::currentDateTime().toString("yyyyMMdd-HHmmss-zzz"));
            requestPreview("baseline", {"compare-models", photo, "--baseline-model", raftModel_->text(), "--candidate-model", candidate, "--raft-root", raftRoot_->text(), "--device", trainDevice_->currentText(), "--output-dir", comparisonOutput});
            tabs_->setCurrentIndex(1); previewStats_->setText("Comparing the baseline and trained model in the background…");
        });
        tabs_->addTab(scroll, "4. Train & compare");
    }

    bool validName(const QString &name) {
        return !name.isEmpty() && name != "." && name != ".." && !name.contains('/') && !name.contains('\\');
    }

    void generateDataset() {
        if (sources_->topLevelItemCount() == 0) { QMessageBox::information(this, "Add photos", "Add spatial HEIC photos before generating a dataset."); return; }
        const QString name = datasetName_->text().trimmed();
        if (!validName(name)) { QMessageBox::information(this, "Dataset name", "Use a folder name without path separators."); return; }
        QStringList args{"dataset"}; QJsonObject groups;
        for (int i = 0; i < sources_->topLevelItemCount(); ++i) {
            auto *item = sources_->topLevelItem(i); const QString group = item->text(1).trimmed();
            if (useGroups_->isChecked() && group.isEmpty()) { QMessageBox::information(this, "Photo groups", "Give each photo a group, or turn off Use photo groups to build an ungrouped dataset."); return; }
            const QString path = item->data(0, Qt::UserRole).toString(); args << path; groups.insert(path, group);
        }
        if (useGroups_->isChecked()) {
            groupsFile_ = std::make_unique<QTemporaryFile>();
            if (!groupsFile_->open() || groupsFile_->write(QJsonDocument(groups).toJson()) < 0 || !groupsFile_->flush()) {
                log_->appendPlainText("Could not create the photo-group configuration."); groupsFile_.reset(); return;
            }
            args << "--groups" << groupsFile_->fileName();
        }
        QJsonArray teachers;
        for (auto *check : teacherChecks_) if (check->isChecked()) {
            const QString model = check->property("model").toString(), primary = teacher_->currentData().toString();
            teachers.append(QJsonObject{{"id", model}, {"model", model}, {"model_path", model == primary ? teacherPath_->text() : modelPaths_.value(model)->text()}, {"source_dir", model == primary ? teacherSource_->text() : modelSources_.value(model)->text()}, {"device", teacherDevice_->currentText()}, {"input_size", inputSize_->value()}});
        }
        if (teachers.isEmpty()) { QMessageBox::information(this, "Select teacher", "Enable at least one depth teacher."); return; }
        teachersFile_ = std::make_unique<QTemporaryFile>();
        if (!teachersFile_->open() || teachersFile_->write(QJsonDocument(teachers).toJson()) < 0 || !teachersFile_->flush()) { log_->appendPlainText("Could not create teacher configuration."); teachersFile_.reset(); return; }
        args << "--output-dir" << QDir(workspace_->text()).filePath("datasets/" + name)
             << "--teachers-json" << teachersFile_->fileName() << "--name" << name << "--category" << category_->text().trimmed()
             << "--grouping" << (useGroups_->isChecked() ? (verifiedScenes_->isChecked() ? "scene" : "capture") : "none");
        if (anchor_->isChecked()) args << "--metric-anchor" << "depthpro";
        if (includeDisplayTeacher_->isChecked()) args << "--include-display-teacher";
        refreshAfter_ = true; startJob("Generate dataset", args);
    }

    void trainDataset() {
        const QString dataset = selectedPath(datasets_); const QString name = runName_->text().trimmed();
        if (dataset.isEmpty()) { QMessageBox::information(this, "Choose dataset", "Select a dataset from the library above."); return; }
        if (dataset == reviewedDataset_) {
            for (auto *item : reviewEntries()) {
                if (item->checkState(0) == Qt::Unchecked) {
                    tabs_->setCurrentIndex(1);
                    QMessageBox::information(this, "Save your exclusions", "Unchecking photos changes this review. Click Save reviewed copy to create the dataset with those photos excluded, then train the saved copy selected in the library.");
                    return;
                }
            }
        }
        if (!validName(name)) { QMessageBox::information(this, "Run name", "Use a run folder name without path separators."); return; }
        if (patch_->value() % 32) { QMessageBox::information(this, "Patch size", "RAFT patch size must be a multiple of 32 (for example 256 or 512)."); return; }
        const QString checkpoint = QDir(workspace_->text()).filePath("runs/" + name + "/checkpoint.pth");
        saveSharedModelSettings();
        refreshAfter_ = true;
        QStringList args{"train", dataset, "--checkpoint", checkpoint, "--raft-root", raftRoot_->text(), "--raft-model", raftModel_->text(),
            "--epochs", QString::number(epochs_->value()), "--steps", QString::number(steps_->value()), "--patch-size", QString::number(patch_->value()),
            "--iterations", QString::number(iterations_->value()), "--scope", scope_->currentData().toString(), "--device", trainDevice_->currentText(), "--mode", trainingMode_->currentData().toString()};
        const QString member = projectSettings_ ? projectSettings_->value("raft/member").toString() : QString(); if (!member.isEmpty()) args << "--raft-model-member" << member;
        startJob("Train RAFT-Stereo", args);
    }

    void exportModel() {
        const QString checkpoint = selectedPath(runs_);
        if (checkpoint.isEmpty()) { QMessageBox::information(this, "Choose model", "Select a trained model from the library above."); return; }
        const QString parent = QFileDialog::getExistingDirectory(this, "Choose export destination", workspace_->text());
        if (parent.isEmpty()) return;
        exportDestination_ = QDir(parent).filePath("raft-export-" + QDateTime::currentDateTime().toString("yyyyMMdd-HHmmss"));
        startJob("Export RAFT model", {"export", checkpoint, "--output", exportDestination_});
    }

    void refreshLibrary() {
        if (process_ && process_->state() != QProcess::NotRunning) return;
        settings_.setValue("workspace", workspace_->text());
        QStringList arguments{"workspace", workspace_->text()};
        if (projectSettings_) {
            projectSettings_->sync();
            for (const QString &path : projectSettings_->value("dataset_links").toStringList())
                arguments << "--linked-dataset" << path;
        }
        startJob("Refresh library", arguments);
    }

    void archiveDataset() {
        if (process_->state() != QProcess::NotRunning) return;
        const QString source = selectedPath(datasets_);
        if (source.isEmpty()) return;
        const QString datasetsRoot = QFileInfo(QDir(workspace_->text()).filePath("datasets")).canonicalFilePath();
        if (datasetsRoot.isEmpty() || QFileInfo(source).canonicalPath() != datasetsRoot) {
            log_->appendPlainText("Only datasets directly inside this workspace can be archived."); return;
        }
        const QString archiveRoot = QDir(workspace_->text()).filePath("archived");
        const QString destination = QDir(archiveRoot).filePath("dataset-" + QFileInfo(source).fileName() + "-" + QDateTime::currentDateTime().toString("yyyyMMdd-HHmmss-zzz"));
        if (!QDir().mkpath(archiveRoot) || !QDir().rename(source, destination)) {
            log_->appendPlainText("Could not archive dataset; its files remain at " + source); return;
        }
        log_->appendPlainText("Dataset archived without deleting files: " + destination);
        if (source == reviewedDataset_) {
            reviewedDataset_.clear(); reviewPreviews_.reset();
            QSignalBlocker samplesBlocker(reviewSamples_), labelBlocker(reviewLabel_);
            reviewSamples_->clear(); reviewLabel_->clear();
            rgbPreview_->reset("Dataset archived."); depthPreview_->reset("Select another dataset to review.");
            previewStats_->clear(); reviewPath_->setText("Dataset archived to " + destination); updateReviewCount();
        }
        refreshLibrary();
    }

    void startJob(const QString &label, const QStringList &arguments) {
        if (process_->state() != QProcess::NotRunning) { statusBar()->showMessage("Finish or cancel the current background task first; photo review remains available."); return; }
        if (!QFileInfo::exists(scriptPath())) { log_->appendPlainText("RAFT Studio script is missing: " + scriptPath()); groupsFile_.reset(); teachersFile_.reset(); refreshAfter_ = false; statusBar()->showMessage("Could not start " + label + "; see the progress log."); return; }
        job_ = label; stdout_.clear(); progressBuffer_.clear(); cancelled_ = false;
        if (label == "Scan spatial photos") scanDeliveredPaths_.clear();
        setBusy(true);
        log_->appendPlainText(label + "…");
        process_->setProgram(pythonPath()); process_->setArguments(QStringList{scriptPath(), "--json"} + arguments);
        process_->start();
    }

    void appendProgress(const QByteArray &bytes) {
        progressBuffer_ += bytes;
        while (progressBuffer_.contains('\n')) {
            const int newline = progressBuffer_.indexOf('\n'); QString line = QString::fromUtf8(progressBuffer_.left(newline)); progressBuffer_.remove(0, newline + 1);
            line.remove(QRegularExpression("\\x1b\\[[0-9;]*m"));
            if (line.startsWith("IPDE_EVENT ")) {
                const auto event = QJsonDocument::fromJson(line.mid(11).toUtf8()).object(); const QString type = event.value("event").toString(), phase = event.value("phase").toString();
                if (type == "dataset_started") streamingDataset_ = event.value("dataset_dir").toString();
                else if (type == "sample_ready") { streamingDataset_ = event.value("dataset_dir").toString(streamingDataset_); if (!streamingTimer_->isActive()) streamingTimer_->start(); }
                else if (type == "photo_skipped") log_->appendPlainText("Skipped " + QFileInfo(event.value("source_path").toString()).fileName() + ": " + event.value("reason").toString());
                else if (job_ == "Scan spatial photos" && type == "scan_started") {
                    log_->appendPlainText("Searching " + event.value("directory").toString() + ". Photos appear as they are validated; cancelling keeps photos already found.");
                }
                else if (job_ == "Scan spatial photos" && type == "scan_photo_started") {
                    statusBar()->showMessage(QString("Checking photo %1: %2 — %3 spatial photos found").arg(event.value("candidate_count").toInt()).arg(QFileInfo(event.value("source_path").toString()).fileName()).arg(event.value("accepted_count").toInt()));
                }
                else if (job_ == "Scan spatial photos" && type == "spatial_photo_found") {
                    const QString path = event.value("source_path").toString(); const QFileInfo info(path);
                    scanDeliveredPaths_.insert(info.canonicalFilePath().isEmpty() ? info.absoluteFilePath() : info.canonicalFilePath());
                    addSourcePhoto(path, event.value("photo_metadata").toObject());
                    statusBar()->showMessage(QString("Found %1 spatial photos — %2 photos in the list. Scanning continues…").arg(event.value("accepted_count").toInt()).arg(sources_->topLevelItemCount()));
                }
                else if (type == "dataset_complete") { streamingTimer_->stop(); streamingDataset_.clear(); }
                else if (!phase.isEmpty()) {
                    QString message;
                    const int processed = event.value("processed").toInt(), total = event.value("total").toInt();
                    if (phase == "verifying_dataset") message = QString("Checking source dataset %1 of %2: %3. Large datasets can take several minutes.").arg(processed).arg(total).arg(QFileInfo(event.value("dataset_path").toString()).fileName());
                    else if (phase == "sample_composed") message = QString("Creating training set: %1 of %2 teacher entries copied losslessly (%3 unique arrays).").arg(processed).arg(total).arg(event.value("unique_arrays").toInt());
                    else if (phase == "compressing") message = QString("Compressing arrays losslessly: %1 of %2 records (%3 unique arrays).").arg(processed).arg(total).arg(event.value("unique_arrays").toInt());
                    else if (phase == "verifying_output") message = "Checking the saved arrays before publishing the completed training set. Large datasets can take several minutes.";
                    else if (phase == "publishing_dataset") message = "Publishing the verified training set…";
                    if (!message.isEmpty()) {
                        statusBar()->showMessage(message);
                        if (job_ == "Create training set") collectionStatus_->setText(message + " Use Cancel current task to stop.");
                        if ((phase == "sample_composed" || phase == "compressing") && total > 0) { progress_->setRange(0, total); progress_->setValue(processed); }
                        else progress_->setRange(0, 0);
                    }
                }
            } else if (!line.trimmed().isEmpty()) log_->appendPlainText(line.trimmed());
        }
    }

    void setBusy(bool busy) {
        busy_ = busy;
        workspace_->setEnabled(!busy); chooseWorkspace_->setEnabled(!busy); refresh_->setEnabled(!busy); export_->setEnabled(!busy);
        generate_->setEnabled(!busy); train_->setEnabled(!busy); importHf_->setEnabled(!busy);
        collectionSources_->setEnabled(!busy); collectionName_->setEnabled(!busy); splitMode_->setEnabled(!busy); groupingPolicy_->setEnabled(!busy); splitSeed_->setEnabled(!busy);
        validationFraction_->setEnabled(!busy && splitMode_->currentData().toString() != "explicit"); validationCount_->setEnabled(!busy && splitMode_->currentData().toString() == "equal-per-dataset");
        updateCollectionReadiness();
        if (busy) saveReviewed_->setEnabled(false); else updateReviewCount();
        cancel_->setEnabled(busy); progress_->setRange(0, busy ? 0 : 1); progress_->setValue(0);
        statusBar()->showMessage(busy ? job_ : "Ready");
    }

    bool hasUnsavedReviewExclusions(const QString &path) const {
        if (path.isEmpty() || path != reviewedDataset_) return false;
        for (auto *item : reviewEntries()) if (item->checkState(0) == Qt::Unchecked) return true;
        return false;
    }

    void stopGenerationStreaming(const QString &reason) {
        const QString path = !streamingDataset_.isEmpty() ? streamingDataset_ : reviewGenerating_ ? reviewedDataset_ : QString();
        streamingTimer_->stop(); streamingDataset_.clear();
        if (!path.isEmpty()) {
            if (pendingPreviewArgs_.contains(path)) pendingPreviewArgs_.clear();
            if (previewProcess_->arguments().contains(path)) { previewKind_ = "discarded"; if (previewProcess_->state() != QProcess::NotRunning) previewProcess_->kill(); }
            if (requestedReviewPath_ == path) requestedReviewPath_.clear();
        }
        if (reviewGenerating_ && (path.isEmpty() || reviewedDataset_ == path)) {
            reviewedDataset_.clear(); reviewGenerating_ = false;
            previewRecords_ = {}; differencePath_.clear();
            QSignalBlocker samplesBlocker(reviewSamples_), labelsBlocker(reviewLabel_);
            reviewSamples_->clear(); reviewLabel_->clear();
            rgbPreview_->reset(reason); for (auto *preview : depthPreviews_) preview->reset("Generate or select a complete dataset to review.");
            previewStats_->clear(); reviewPath_->setText(reason);
        }
        updateReviewCount();
    }

    void updateCollectionReadiness() {
        if (!compose_ || !collectionStatus_) return;
        int training = 0, validation = 0, entries = 0; bool unsavedExclusions = false;
        for (int i = 0; i < collectionSources_->topLevelItemCount(); ++i) {
            auto *item = collectionSources_->topLevelItem(i);
            if (item->checkState(0) == Qt::Checked) { ++training; entries += item->text(3).toInt(); }
            if (item->checkState(1) == Qt::Checked) ++validation;
            if (item->checkState(0) == Qt::Checked || (splitMode_->currentData().toString() == "explicit" && item->checkState(1) == Qt::Checked))
                unsavedExclusions |= hasUnsavedReviewExclusions(item->data(0, Qt::UserRole).toString());
        }
        QString reason;
        if (busy_) reason = job_ == "Create training set" ? "Creating the training set. Checking and losslessly copying full-quality arrays may take several minutes. Use Cancel current task to stop." : job_ + " is running. Finish or cancel it before creating a training set.";
        else if (!collectionSources_->topLevelItemCount()) reason = "Import or generate a dataset first; it will appear here and can be used for training.";
        else if (!training) reason = "Tick a dataset in the Use for training column. One dataset is enough; validation photos are held out automatically.";
        else if (!entries) reason = "The selected training datasets have no teacher entries. Finish generating or import a complete dataset first.";
        else if (unsavedExclusions) reason = "This dataset has unsaved photo exclusions. Click Save reviewed copy in Review depth maps, then select that saved copy here so the training set respects your choices.";
        else if (!validName(collectionName_->text().trimmed())) reason = "Enter a training set name without folder separators.";
        else if (QFileInfo::exists(QDir(workspace_->text()).filePath("datasets/" + collectionName_->text().trimmed()))) reason = "A dataset with this name already exists. Enter a new training set name.";
        else if (splitMode_->currentData().toString() == "explicit" && !validation) reason = "Tick a separate dataset in Use as validation, or choose random validation to hold out photos from your training dataset.";
        compose_->setEnabled(reason.isEmpty());
        compose_->setText(busy_ && job_ == "Create training set" ? "Creating training set…" : "Create training set & continue");
        const QString ready = QString("Ready: %1 training dataset(s), %2 teacher entries. %3 The set needs at least two independent photo groups.").arg(training).arg(entries).arg(splitMode_->currentData().toString() == "explicit" ? QString("%1 validation dataset(s) will be held out.").arg(validation) : "Validation photos will be held out automatically.");
        collectionStatus_->setText(reason.isEmpty() ? ready : reason);
        compose_->setToolTip(reason.isEmpty() ? "Create a lossless training set, then open the training step." : reason);
    }

    void processFinished(int code, QProcess::ExitStatus status) {
        stdout_ += process_->readAllStandardOutput(); appendProgress(process_->readAllStandardError());
        if (!progressBuffer_.trimmed().isEmpty()) { log_->appendPlainText(QString::fromUtf8(progressBuffer_).trimmed()); progressBuffer_.clear(); }
        setBusy(false); groupsFile_.reset(); teachersFile_.reset();
        if (cancelled_) {
            if (job_ == "Generate dataset") stopGenerationStreaming("Generation cancelled. Generate a new dataset before saving a reviewed copy.");
            streamingTimer_->stop(); streamingDataset_.clear();
            refreshAfter_ = false; statusBar()->showMessage("Cancelled");
            if (job_ == "Scan spatial photos") { log_->appendPlainText("Scan cancelled; photos already found remain in the list."); statusBar()->showMessage("Scan cancelled; photos already found remain in the list."); }
            if (job_ == "Preview depth") { rgbPreview_->reset("Preview cancelled."); depthPreview_->reset("Choose another photo or depth label to retry."); }
            return;
        }
        QJsonParseError error; const QJsonDocument doc = QJsonDocument::fromJson(stdout_.trimmed(), &error);
        if (status != QProcess::NormalExit || code != 0 || !doc.isObject()) {
            log_->appendPlainText(job_ + " failed (exit " + QString::number(code) + ").");
            statusBar()->showMessage(job_ + " failed; see the progress log. You can adjust the selection and retry.");
            const QString explanation = doc.isObject() ? doc.object().value("error").toString() : QString();
            if (!explanation.isEmpty()) {
                log_->appendPlainText(explanation);
                if (job_ == "Create training set") collectionStatus_->setText("Could not create the training set: " + explanation + " Adjust the selection or settings and retry.");
            } else if (!stdout_.trimmed().isEmpty()) log_->appendPlainText(QString::fromUtf8(stdout_).left(12000));
            if (code == 0 && error.error != QJsonParseError::NoError) log_->appendPlainText("Could not parse the result: " + error.errorString());
            if (job_ == "Preview depth") {
                rgbPreview_->reset("Preview unavailable."); depthPreview_->reset("Preview unavailable.");
                previewStats_->setText("Preview failed. See the error below; this sample has not been automatically excluded.");
            }
            if (job_ == "Generate dataset") stopGenerationStreaming("Generation failed. See the progress log and generate a new dataset to try again.");
            refreshAfter_ = false; return;
        }
        const QJsonObject result = doc.object();
        if (job_ == "Refresh library") {
            populateLibrary(result);
            for (const auto &warning : result.value("warnings").toArray()) log_->appendPlainText(warning.toString());
        }
        else if (job_ == "Review dataset") populateReview(result);
        else if (job_ == "Preview depth") populatePreview(result);
        else if (job_ == "Scan spatial photos") {
            const auto acceptedPhotos = result.value("accepted").toArray();
            for (const auto &value : acceptedPhotos) {
                const auto accepted = value.toObject(); const QString path = accepted.value("source_path").toString(); const QFileInfo info(path);
                const QString canonical = info.canonicalFilePath().isEmpty() ? info.absoluteFilePath() : info.canonicalFilePath();
                if (!scanDeliveredPaths_.contains(canonical)) addSourcePhoto(path, accepted.value("photo_metadata").toObject());
            }
            log_->appendPlainText(QString("Scan complete: %1 valid spatial photos found; %2 skipped. %3 photos are now in the list. %4").arg(acceptedPhotos.size()).arg(result.value("skipped").toArray().size()).arg(sources_->topLevelItemCount()).arg(result.value("authenticity_note").toString()));
            if (acceptedPhotos.isEmpty()) {
                const bool noCandidates = result.value("summary").toObject().value("candidates").toInt(-1) == 0;
                log_->appendPlainText(noCandidates ? "No HEIC / HEIF photos were found. Choose the folder containing your original photos, rather than the workspace or generated dataset folder; subfolders are included."
                    : "No calibrated Apple spatial photos were found. See the skipped-file reasons above. Portrait and ordinary HEIC photos do not contain the stereo pair needed for RAFT training.");
            }
        }
        else if (job_ == "Create training set") {
            pendingTrainingPath_ = result.value("dataset_path").toString(); log_->appendPlainText("Training set saved: " + pendingTrainingPath_);
            for (const auto &warning : result.value("warnings").toArray()) log_->appendPlainText(warning.toString());
        }
        else if (job_ == "Generate dataset" || job_ == "Save reviewed dataset" || job_ == "Import Hugging Face dataset" || job_ == "Compact dataset") {
            pendingReview_ = result.value("dataset_path").toString();
            log_->appendPlainText("Dataset saved: " + pendingReview_ + "\n" + QString::fromUtf8(QJsonDocument(result.value("summary").toObject()).toJson(QJsonDocument::Compact)));
            for (const auto &warning : result.value("warnings").toArray()) log_->appendPlainText(warning.toString());
        }
        else log_->appendPlainText(QString::fromUtf8(QJsonDocument(result).toJson(QJsonDocument::Indented)).left(18000));
        statusBar()->showMessage(job_ + " complete");
        if (job_ == "Export RAFT model") {
            log_->appendPlainText("Export complete: " + exportDestination_ + "/raft-model.pth — choose this file in IPDE's RAFT model control.");
            showFolder(exportDestination_);
        }
        const bool refresh = refreshAfter_; refreshAfter_ = false;
        if (refresh) QTimer::singleShot(0, this, [this] { refreshLibrary(); });
        else if (job_ == "Refresh library" && !pendingReview_.isEmpty()) {
            const QString path = pendingReview_; pendingReview_.clear();
            QTimer::singleShot(0, this, [this, path] { reviewDataset(path); });
        }
        else if (job_ == "Refresh library" && !pendingTrainingPath_.isEmpty()) { pendingTrainingPath_.clear(); tabs_->setCurrentIndex(4); }
    }

    void populateLibrary(const QJsonObject &result) {
        const QString oldDataset = !pendingTrainingPath_.isEmpty() ? pendingTrainingPath_ : pendingReview_.isEmpty() ? selectedPath(datasets_) : pendingReview_, oldRun = selectedPath(runs_);
        QMap<QString, QPair<Qt::CheckState, Qt::CheckState>> collectionChecks;
        for (int i=0; i<collectionSources_->topLevelItemCount(); ++i) { auto *item = collectionSources_->topLevelItem(i); collectionChecks.insert(item->data(0, Qt::UserRole).toString(), {item->checkState(0), item->checkState(1)}); }
        datasets_->clear(); runs_->clear(); QSignalBlocker collectionBlocker(collectionSources_); collectionSources_->clear();
        for (const QJsonValue &value : result.value("datasets").toArray()) {
            const auto obj = value.toObject(); const QString path = obj.value("path").toString();
            QString teacher = obj.value("teacher").toString();
            if (teacher.isEmpty() && obj.value("teacher").isObject()) teacher = obj.value("teacher").toObject().value("model").toString();
            auto *item = new QTreeWidgetItem(datasets_, {obj.value("name").toString(QFileInfo(path).fileName()), QString::number(obj.value("sample_count").toInt()),
                QString("%1 / %2").arg(obj.value("train_count").toInt()).arg(obj.value("validation_count").toInt()), teacher, obj.value("category").toString(), QString::number(obj.value("storage_bytes").toDouble() / (1024*1024), 'f', 1) + " MiB"});
            const bool linked = obj.value("linked").toBool();
            const QString location = path + (linked ? "\nLinked from another location; source files stay there. Reviewed copies are saved in this project." : QString());
            item->setData(0, Qt::UserRole, path); item->setToolTip(0, location);
            if (linked) item->setText(0, item->text(0) + " ↗");
            if (path == oldDataset) datasets_->setCurrentItem(item);
            auto *collection = new QTreeWidgetItem(collectionSources_, {obj.value("name").toString(QFileInfo(path).fileName()), "", obj.value("category").toString(), QString::number(obj.value("sample_count").toInt())});
            collection->setData(0, Qt::UserRole, path); collection->setToolTip(0, location);
            const auto checks = collectionChecks.value(path, {Qt::Unchecked, Qt::Unchecked}); collection->setCheckState(0, checks.first); collection->setCheckState(1, checks.second);
        }
        for (const QJsonValue &value : result.value("runs").toArray()) {
            const auto obj = value.toObject(); const QString path = obj.value("path").toString();
            const QJsonObject metrics = obj.value("validation").toObject();
            const QJsonValue mae = metrics.value("mean_absolute_flow_error_pixels");
            QString validation = mae.isDouble() ? QString::number(mae.toDouble(), 'f', 2) + " px teacher MAE" : "See metrics";
            auto *item = new QTreeWidgetItem(runs_, {obj.value("name").toString(QFileInfo(path).completeBaseName()), QString::number(obj.value("best_epoch").toInt()), validation});
            item->setData(0, Qt::UserRole, path); item->setToolTip(0, path);
            item->setToolTip(2, QString::fromUtf8(QJsonDocument(metrics).toJson(QJsonDocument::Indented)));
            if (path == oldRun) runs_->setCurrentItem(item);
        }
        if (!datasets_->currentItem() && datasets_->topLevelItemCount()) datasets_->setCurrentItem(datasets_->topLevelItem(0));
        if (!runs_->currentItem() && runs_->topLevelItemCount()) runs_->setCurrentItem(runs_->topLevelItem(0));
        if (collectionSources_->topLevelItemCount() == 1) {
            auto *item = collectionSources_->topLevelItem(0);
            if (!collectionChecks.contains(item->data(0, Qt::UserRole).toString())) item->setCheckState(0, Qt::Checked);
        }
        updateCollectionReadiness();
    }

    void showFolder(const QString &path) {
        if (!path.isEmpty()) QDesktopServices::openUrl(QUrl::fromLocalFile(path));
    }

    QSettings settings_;
    std::unique_ptr<QSettings> projectSettings_;
    QString projectRoot_; bool datasetMode_ = false;
    QLineEdit *workspace_ = nullptr, *datasetName_ = nullptr, *runName_ = nullptr;
    QLineEdit *teacherPath_ = nullptr, *teacherSource_ = nullptr, *raftRoot_ = nullptr, *raftModel_ = nullptr;
    QLineEdit *reviewedName_ = nullptr, *category_ = nullptr, *reviewFilter_ = nullptr, *collectionName_ = nullptr;
    QLineEdit *hfSource_ = nullptr, *hfConfig_ = nullptr, *hfSplit_ = nullptr, *hfMapping_ = nullptr, *hfDataFiles_ = nullptr, *hfAssetDir_ = nullptr;
    QMap<QString, QLineEdit *> modelPaths_, modelSources_;
    QTreeWidget *datasets_ = nullptr, *runs_ = nullptr, *sources_ = nullptr;
    QTreeWidget *reviewSamples_ = nullptr, *collectionSources_ = nullptr;
    QComboBox *teacher_ = nullptr, *teacherDevice_ = nullptr, *trainDevice_ = nullptr, *scope_ = nullptr, *trainingMode_ = nullptr;
    QComboBox *reviewLabel_ = nullptr, *goal_ = nullptr, *reviewCamera_ = nullptr, *visualView_ = nullptr, *splitMode_ = nullptr, *groupingPolicy_ = nullptr;
    QLabel *scaleHelp_ = nullptr, *reviewPath_ = nullptr, *reviewCount_ = nullptr, *previewStats_ = nullptr, *goalHelp_ = nullptr, *splitHelp_ = nullptr, *collectionStatus_ = nullptr;
    QList<QLabel *> depthTitles_;
    DepthPreview *rgbPreview_ = nullptr, *depthPreview_ = nullptr;
    QList<DepthPreview *> depthPreviews_;
    QSplitter *library_ = nullptr;
    QCheckBox *verifiedScenes_ = nullptr, *anchor_ = nullptr, *advanced_ = nullptr, *useGroups_ = nullptr, *compareTeachers_ = nullptr, *includeDisplayTeacher_ = nullptr;
    QList<QCheckBox *> teacherChecks_;
    QGroupBox *datasetAdvanced_ = nullptr, *trainingAdvanced_ = nullptr;
    QSpinBox *inputSize_ = nullptr, *epochs_ = nullptr, *steps_ = nullptr, *patch_ = nullptr, *iterations_ = nullptr;
    QSpinBox *splitSeed_ = nullptr, *validationCount_ = nullptr;
    QDoubleSpinBox *validationFraction_ = nullptr;
    QTabWidget *tabs_ = nullptr; QPlainTextEdit *log_ = nullptr; QProgressBar *progress_ = nullptr;
    QPushButton *chooseWorkspace_ = nullptr, *refresh_ = nullptr, *generate_ = nullptr, *train_ = nullptr, *cancel_ = nullptr, *export_ = nullptr;
    QPushButton *saveReviewed_ = nullptr, *compose_ = nullptr, *importHf_ = nullptr, *compareBaseline_ = nullptr;
    QProcess *process_ = nullptr; QByteArray stdout_; QString job_, exportDestination_;
    QString requestedReviewPath_, reviewedDataset_, pendingReview_, pendingTrainingPath_;
    QProcess *previewProcess_ = nullptr; QByteArray previewStdout_, progressBuffer_; QString previewKind_, pendingPreviewKind_, streamingDataset_, differencePath_;
    QStringList pendingPreviewArgs_; QJsonArray previewRecords_; QTimer *streamingTimer_ = nullptr;
    QSet<QString> scanDeliveredPaths_;
    bool cancelled_ = false, refreshAfter_ = false, reviewGenerating_ = false, requestedReviewOpen_ = true, busy_ = false;
    std::unique_ptr<QTemporaryFile> groupsFile_, teachersFile_;
    std::unique_ptr<QTemporaryDir> reviewPreviews_;
};

} // namespace

int main(int argc, char **argv) {
    QApplication application(argc, argv);
    const QStringList args = application.arguments(); const int mode = args.indexOf("--mode");
    const bool datasetMode = mode >= 0 ? args.value(mode + 1) == "datasets" : bool(IPDE_DATASET_STUDIO);
    application.setApplicationName(datasetMode ? "Dataset Studio" : "RAFT Studio"); application.setOrganizationName("IPDE");
    IPDE::ProjectSession session(datasetMode ? "datasets" : "trainer", &application);
    if (!session.start()) return 2;
    TrainerWindow window; window.show();
    const auto icon = IPDE::appIcon(datasetMode ? "datasets" : "trainer");
    application.setWindowIcon(icon); window.setWindowIcon(icon);
    session.setChangedHandler(&window, [&window] { QTimer::singleShot(0, &window, [&window] { window.projectChanged(); }); });
    session.setDisconnectedHandler(&window, [&window] { window.statusBar()->showMessage("Studio disconnected; this project remains locked until this window closes."); });
    QObject::connect(&application, &QCoreApplication::aboutToQuit, &session, [&session] { session.shutdown(); });
    if (args.contains("--smoke-test") && !args.contains("--screenshot")) QTimer::singleShot(100, &application, &QCoreApplication::quit);
    const int screenshot = args.indexOf("--screenshot");
    if (screenshot >= 0 && screenshot + 1 < args.size()) {
        auto *capture = new QTimer(&application); capture->setInterval(1000);
        QObject::connect(capture, &QTimer::timeout, &window, [&application, &window, args, screenshot] {
            if (window.taskRunning()) return;
            const bool saved = window.grab().save(args.at(screenshot + 1));
            application.exit(saved ? 0 : 2);
        });
        capture->start();
    }
    return application.exec();
}
