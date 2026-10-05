#include <QApplication>
#include <QComboBox>
#include <QCloseEvent>
#include <QCryptographicHash>
#include <QDateTime>
#include <QDoubleSpinBox>
#include <QDesktopServices>
#include <QDialog>
#include <QDialogButtonBox>
#include <QDir>
#include <QDirIterator>
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
#include <QListWidget>
#include <QMainWindow>
#include <QMenu>
#include <QToolButton>
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
#include <QSaveFile>
#include <QSet>
#include <QScrollArea>
#include <QScrollBar>
#include <QScreen>
#include <QSignalBlocker>
#include <QSpinBox>
#include <QSlider>
#include <QSplitter>
#include <QStandardPaths>
#include <QStatusBar>
#include <QStyledItemDelegate>
#include <QTabWidget>
#include <QTemporaryFile>
#include <QTemporaryDir>
#include <QTimer>
#include <QThread>
#include <QTreeWidget>
#include <QUrl>
#include <QUuid>
#include <QVector3D>
#include <QVBoxLayout>

#include <memory>
#include <functional>
#include <cmath>
#include <algorithm>

#include "project_session.h"
#include "studio_icons.h"
#include "window_layout.h"
#include "help_support.h"
#include "python_runtime.h"

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
    return IPDE::pythonExecutable();
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

QJsonObject readJson(const QString &path) {
    QFile file(path);
    if (!file.open(QIODevice::ReadOnly)) return {};
    return QJsonDocument::fromJson(file.readAll()).object();
}
QStringList localDatasets(const QString &project) {
    if (project.isEmpty()) return {};
    QStringList paths;
    QDir dir(QDir(project).filePath("workspace/datasets"));
    for (const QFileInfo &info : dir.entryInfoList(QDir::Dirs | QDir::NoDotAndDotDot)) {
        if (QFileInfo::exists(QDir(info.absoluteFilePath()).filePath("dataset.json"))) {
            const QString path = info.canonicalFilePath();
            if (!path.isEmpty() && !paths.contains(path)) paths << path;
        }
    }
    return paths;
}
QStringList linkedDatasets(const QString &project) {
    if (project.isEmpty()) return {};
    return QSettings(QDir(project).filePath("project.ini"), QSettings::IniFormat).value("dataset_links").toStringList();
}
QStringList projectDatasets(const QString &project) {
    if (project.isEmpty()) return {};
    QStringList paths = localDatasets(project);
    for (const QString &link : linkedDatasets(project)) {
        const QString canonical = QFileInfo(link).canonicalFilePath();
        const QString path = canonical.isEmpty() ? QDir::cleanPath(link) : canonical;
        if (!paths.contains(path)) paths << path;
    }
    return paths;
}

class TrainerWindow final : public QMainWindow {
public:
    explicit TrainerWindow(bool datasetMode = bool(IPDE_DATASET_STUDIO)) : settings_(QSettings::defaultFormat(), QSettings::UserScope, "IPDE", "RAFTStudio") {
        const auto arguments = QCoreApplication::arguments();
        const int modeArgument = arguments.indexOf("--mode");
        datasetMode_ = modeArgument >= 0 ? arguments.value(modeArgument + 1) == "datasets" : datasetMode;
        projectRoot_ = IPDE::projectRoot();
        if (!projectRoot_.isEmpty()) projectSettings_ = std::make_unique<QSettings>(QDir(projectRoot_).filePath("project.ini"), QSettings::IniFormat);
        setWindowTitle(datasetMode_ ? "Dataset Studio" : "RAFT Studio");
        IPDE::installHelpMenu(this, datasetMode_ ? "Dataset Studio" : "RAFT Studio", datasetMode_ ? "datasets" : "trainer");
        if (projectSettings_) setWindowTitle(windowTitle() + " — "
            + projectSettings_->value("name", QFileInfo(projectRoot_).fileName()).toString());
        auto *central = new QWidget(this);
        auto *root = new QVBoxLayout(central);
        root->setContentsMargins(18, 16, 18, 16);
        auto *title = new QLabel(datasetMode_ ? "Dataset Studio" : "RAFT Studio", central);
        QFont font = title->font(); font.setPointSize(font.pointSize() + 7); font.setBold(true); title->setFont(font);
        root->addWidget(title);
        auto *purpose = new QLabel(datasetMode_ ? "Choose a dataset, manage its photos, then prepare it for training." : "Train from prepared datasets, manage model runs, and compare with your original model.", central);
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
        datasets_->setObjectName("datasetLibrary");
        datasets_->setSelectionMode(QAbstractItemView::SingleSelection);
        if (datasetMode_) { for (int column : {2, 3, 4, 5}) datasets_->setColumnHidden(column, true); datasets_->setColumnWidth(1, 65); }
        datasetLayout->addWidget(datasets_);
        auto *datasetButtons = new QHBoxLayout;
        auto *inspect = new QPushButton("Inspect selected", datasetBox); inspect->setObjectName("inspectDataset");
        auto *review = new QPushButton("Review depth maps", datasetBox);
        auto *showDataset = new QPushButton("Show files", datasetBox);
        auto *archive = new QPushButton("Archive selected", datasetBox);
        auto *compact = new QPushButton("Compact copy…", datasetBox);
        compact->setObjectName("compactDataset"); archive->setObjectName("archiveDataset");
        compact->setVisible(datasetMode_); archive->setVisible(datasetMode_);
        cleanupDataset_ = new QPushButton("Remove generated dataset…", datasetBox); cleanupDataset_->setVisible(datasetMode_);
        cleanupDataset_->setToolTip("After training, remove this generated dataset to reclaim disk space. Original photos and trained models stay in place. Requires confirmation.");
        auto *manageDatasets = new QPushButton("Manage in Dataset Studio", datasetBox); manageDatasets->setVisible(!datasetMode_);
        review->setText(datasetMode_ ? "Open photos and depth" : "Review in Dataset Studio");
        review->setIcon(IPDE::appIcon("datasets")); inspect->setIcon(IPDE::appIcon("datasets"));
        datasetButtons->addWidget(review); datasetButtons->addWidget(inspect); datasetButtons->addWidget(showDataset); datasetButtons->addWidget(compact); datasetButtons->addWidget(archive); datasetButtons->addWidget(cleanupDataset_); datasetButtons->addWidget(manageDatasets); datasetButtons->addStretch();
        datasetLayout->addLayout(datasetButtons);
        auto *links = new QHBoxLayout; auto *linkProject = new QPushButton("Link datasets from project…", datasetBox); auto *linkDataset = new QPushButton("Link dataset…", datasetBox); auto *manageLinks = new QPushButton("Manage links…", datasetBox);
        for (auto *button : {linkProject, linkDataset, manageLinks}) { links->addWidget(button); button->setVisible(datasetMode_ && bool(projectSettings_)); } links->addStretch(); datasetLayout->addLayout(links);
        connect(linkProject, &QPushButton::clicked, this, [this] { importProjectDatasets(); });
        connect(linkDataset, &QPushButton::clicked, this, [this] { const QString path = QFileDialog::getExistingDirectory(this, "Link an existing dataset folder containing dataset.json", projectRoot_); if (!path.isEmpty()) addDatasetLinks({path}); });
        connect(manageLinks, &QPushButton::clicked, this, [this] { editDatasetLinks(); });
        if (datasetMode_) {
            // Keep everyday actions visible and less frequent file operations in one menu.
            for (auto *button : {inspect, showDataset, compact, archive, cleanupDataset_, manageDatasets, linkProject, linkDataset, manageLinks}) button->hide();
            datasetButtons->removeWidget(review); datasetLayout->insertWidget(1, review);
            auto *more = new QToolButton(datasetBox); more->setText("Dataset actions"); more->setPopupMode(QToolButton::InstantPopup);
            auto *menu = new QMenu(more);
            for (auto *button : {inspect, showDataset, compact, archive, cleanupDataset_, linkDataset, linkProject, manageLinks}) {
                if ((button == linkDataset || button == linkProject || button == manageLinks) && !projectSettings_) continue;
                auto *action = menu->addAction(button->text());
                connect(action, &QAction::triggered, button, &QPushButton::click);
                connect(menu, &QMenu::aboutToShow, action, [button, action] { action->setEnabled(button->isEnabled()); });
            }
            more->setMenu(menu); datasetLayout->addWidget(more);
            auto *hint = new QLabel("Select any dataset to open its photos.\nIn Training split, select a row and choose its use with the buttons.", datasetBox); hint->setWordWrap(true); datasetLayout->addWidget(hint);
        }

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
        cleanupRun_ = new QPushButton("Clean run files…", runBox);
        auto *useModel = new QPushButton("Use selected model in project", runBox);
        useModel->setVisible(bool(projectSettings_));
        useModel->setToolTip("Select a trained model after comparing its depth on independent photos. Extraction uses this model until you choose another.");
        runButtons->addWidget(export_); runButtons->addWidget(useModel); runButtons->addWidget(showRun); runButtons->addWidget(cleanupRun_); runButtons->addStretch();
        runLayout->addLayout(runButtons);
        library->addWidget(datasetBox); library->addWidget(runBox);
        root->addWidget(library, 2);
        if (!datasetMode_) library->setMaximumHeight(170);
        connect(review, &QPushButton::clicked, this, [this] { reviewDataset(selectedPath(datasets_)); });
        connect(inspect, &QPushButton::clicked, this, [this] {
            const QString path = selectedPath(datasets_);
            if (!path.isEmpty()) startJob("Inspect dataset", {"inspect-dataset", path, "--workers", QString::number(workerCount())});
        });
        connect(showDataset, &QPushButton::clicked, this, [this] { showFolder(selectedPath(datasets_)); });
        connect(archive, &QPushButton::clicked, this, [this] { archiveDataset(); });
        connect(manageDatasets, &QPushButton::clicked, this, [this] { openApp("datasets", "prepare", selectedPath(datasets_)); });
        connect(cleanupDataset_, &QPushButton::clicked, this, [this] { cleanupDataset(); });
        connect(cleanupRun_, &QPushButton::clicked, this, [this] { cleanupRun(); });
        connect(compact, &QPushButton::clicked, this, [this] {
            const QString source = selectedPath(datasets_); if (source.isEmpty()) return;
            const QString destination = QDir(workspace_->text()).filePath("datasets/" + QFileInfo(source).fileName() + "-compact-" + QDateTime::currentDateTime().toString("yyyyMMdd-HHmmss"));
            refreshAfter_ = true; startJob("Compact dataset", {"compact-dataset", source, "--output-dir", destination, "--workers", QString::number(workerCount())});
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
        connect(tabs_, &QTabWidget::currentChanged, this, [this](int index) {
            if (datasetMode_ && index == 1 && previewProcess_) {
                if (selectedPath(datasets_) != reviewedDataset_ && selectedPath(datasets_) != requestedReviewPath_) reviewDataset(selectedPath(datasets_), false); else reviewSelectionChanged();
            }
        });
        auto *goalControls = new QWidget(central); auto *globalGoal = new QHBoxLayout(goalControls); globalGoal->setContentsMargins(0, 0, 0, 0); globalGoal->addWidget(new QLabel("Main purpose", central)); globalGoal->addWidget(goal_, 1); globalGoal->addWidget(advanced_); root->insertWidget(2, goalControls);
        connect(tabs_, &QTabWidget::currentChanged, this, [this, goalControls](int index) { goalControls->setVisible(!datasetMode_ || index == 0); });
        if (datasetMode_) { tabs_->setTabVisible(4, false); runBox->hide(); }
        else { for (int index : {0, 2, 3}) tabs_->setTabVisible(index, false); tabs_->setTabText(1, "Compare models"); tabs_->setTabText(4, "Train model and compare"); tabs_->setCurrentIndex(4); }
        connect(datasets_, &QTreeWidget::itemSelectionChanged, this, [this] {
            syncDatasetSelection(datasets_, collectionSources_); updateTrainingSelection(); updateCleanupActions();
            loadSplitControls(selectedPath(datasets_));
            if (datasetMode_ && tabs_->currentIndex() == 1 && process_ && !busy_ && !selectedPath(datasets_).isEmpty() && selectedPath(datasets_) != requestedReviewPath_) reviewDataset(selectedPath(datasets_), false);
        });
        connect(collectionSources_, &QTreeWidget::currentItemChanged, this, [this] { syncDatasetSelection(collectionSources_, datasets_); });
        connect(datasets_, &QTreeWidget::itemDoubleClicked, this, [this] { reviewDataset(selectedPath(datasets_)); });
        connect(runs_, &QTreeWidget::itemSelectionChanged, this, [this] { updateCleanupActions(); });
        if (datasetMode_) {
            root->removeWidget(library);
            auto *manager = new QSplitter(Qt::Horizontal, central); manager->setObjectName("datasetManagerSplit");
            manager->addWidget(library); manager->addWidget(tabs_); manager->setChildrenCollapsible(false);
            library->setMinimumWidth(240); library->setMaximumWidth(380); manager->setStretchFactor(1, 1); manager->setSizes({280, 1000});
            root->addWidget(manager, 1); tabs_->setCurrentIndex(1);
        } else root->addWidget(tabs_, 3);
        generate_->setIcon(IPDE::appIcon("datasets")); train_->setIcon(IPDE::appIcon("trainer"));
        export_->setIcon(IPDE::appIcon("trainer"));
        auto *notice = new QLabel(datasetMode_ ? "Photo and split edits save automatically in this dataset. Image and depth files are never copied or rewritten. Related captures stay together in training or validation." : "Teacher depth is an estimate. Use correctly grouped independent scenes for validation.", central);
        notice->setWordWrap(true);
        notice->setToolTip("Dataset checks help catch accidental omissions, altered files, and validation overlap. They do not detect intentional poisoning or establish permission, copyright, or content suitability for externally obtained data.");
        root->addWidget(notice);
        log_ = new QPlainTextEdit(central); log_->setReadOnly(true); log_->setMaximumBlockCount(1500);
        log_->setMaximumHeight(90); log_->setPlaceholderText("Progress and results appear here.");
        auto *details = new QToolButton(central); details->setText("Show task details"); details->setCheckable(true); root->addWidget(details);
        connect(details, &QToolButton::toggled, this, [this, details](bool visible) { log_->setVisible(visible); details->setText(visible ? "Hide task details" : "Show task details"); });
        root->addWidget(log_); log_->setVisible(!datasetMode_); details->setChecked(!datasetMode_);
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
                setBusy(false); groupsFile_.reset(); teachersFile_.reset(); editsFile_.reset(); refreshAfter_ = false;
                if (job_ == "Generate dataset") stopGenerationStreaming("Generation could not start. Generate a new dataset to try again.");
                statusBar()->showMessage(job_ + " failed to start; see the progress log.");
                if (job_ == "Train RAFT-Stereo") trainingStatus_->setText("Model training failed to start: " + process_->errorString());
                if (job_ == "Preview depth") previewStats_->setText("Preview failed: Python could not be launched.");
            }
        });
        connect(cancel_, &QPushButton::clicked, this, [this] {
            if (job_ == "Train RAFT-Stereo") {
                requestTrainingControl("stop");
                return;
            }
            cancelled_ = true;
            if (activeOperation_ == "edit-dataset") {
                const auto pid = process_->processId(); process_->terminate();
                QTimer::singleShot(5000, this, [this, pid] { if (process_->state() != QProcess::NotRunning && process_->processId() == pid) process_->kill(); });
            } else process_->kill();
            log_->appendPlainText("Cancellation requested; partial work will not be presented as complete.");
        });
        previewProcess_ = new QProcess(this);
        connect(previewProcess_, &QProcess::readyReadStandardOutput, this, [this] { previewStdout_ += previewProcess_->readAllStandardOutput(); });
        connect(previewProcess_, qOverload<int, QProcess::ExitStatus>(&QProcess::finished), this, [this](int code, QProcess::ExitStatus status) {
            previewStdout_ += previewProcess_->readAllStandardOutput();
            if (!pendingPreviewArgs_.isEmpty()) { const auto args = pendingPreviewArgs_; const auto label = pendingPreviewKind_; pendingPreviewArgs_.clear(); requestPreview(label, args); return; }
            if (previewKind_ == "discarded") return;
            const auto doc = QJsonDocument::fromJson(previewStdout_.trimmed());
            if (status == QProcess::NormalExit && code == 0 && doc.isObject()) {
                if (previewKind_ == "review") { if (doc.object().value("dataset_path").toString() == requestedReviewPath_) populateReview(doc.object()); } else populatePreview(doc.object());
            } else if (previewKind_ == "review") {
                const QString error = doc.isObject() ? doc.object().value("error").toString() : QString::fromUtf8(previewStdout_).left(800);
                reviewPath_->setText("Could not open " + QFileInfo(requestedReviewPath_).fileName() + ". Select it and click Open photos and depth to retry.");
                previewStats_->setText(error); statusBar()->showMessage("Dataset could not be opened: " + error);
            } else previewStats_->setText("This preview could not be generated. Other photos and teachers remain available. " + QString::fromUtf8(previewStdout_).left(800));
        });
        connect(previewProcess_, &QProcess::errorOccurred, this, [this](QProcess::ProcessError error) {
            if (error == QProcess::FailedToStart) { previewStats_->setText("Preview process could not start: " + previewProcess_->errorString()); if (previewKind_ == "review") reviewPath_->setText("Dataset could not be opened. Click Open photos and depth to retry."); }
        });
        streamingTimer_ = new QTimer(this); streamingTimer_->setSingleShot(true); streamingTimer_->setInterval(750);
        connect(streamingTimer_, &QTimer::timeout, this, [this] {
            if (!streamingDataset_.isEmpty() && appendBase_.isEmpty()) reviewDataset(streamingDataset_, reviewedDataset_.isEmpty());
        });
        editProcess_ = new QProcess(this);
        connect(editProcess_, &QProcess::readyReadStandardOutput, this, [this] { editStdout_ += editProcess_->readAllStandardOutput(); });
        connect(editProcess_, &QProcess::readyReadStandardError, this, [this] { log_->appendPlainText(QString::fromUtf8(editProcess_->readAllStandardError()).trimmed()); });
        connect(editProcess_, qOverload<int, QProcess::ExitStatus>(&QProcess::finished), this, [this](int code, QProcess::ExitStatus status) { finishDatasetSave(code, status); });
        connect(editProcess_, &QProcess::errorOccurred, this, [this](QProcess::ProcessError error) {
            if (error == QProcess::FailedToStart) { editStdout_ = QJsonDocument(QJsonObject{{"error", "Could not start the dataset save process: " + editProcess_->errorString()}}).toJson(); finishDatasetSave(1, QProcess::CrashExit); }
        });
        autosaveTimer_ = new QTimer(this); autosaveTimer_->setSingleShot(true); autosaveTimer_->setInterval(125);
        connect(autosaveTimer_, &QTimer::timeout, this, [this] { startNextDatasetSave(); });
        loadReviewDrafts();
        statusBar()->showMessage("Ready");
        updateTrainingSelection(); updateCleanupActions();
        if (!arguments.contains("--smoke-test")) {
            const int reviewArgument = arguments.indexOf("--review-dataset");
            if (reviewArgument >= 0 && reviewArgument + 1 < arguments.size())
                QTimer::singleShot(0, this, [this, arguments, reviewArgument] {
                    pendingReview_ = QFileInfo(arguments.at(reviewArgument + 1)).absoluteFilePath(); refreshLibrary();
                });
            else QTimer::singleShot(0, this, [this] { refreshLibrary(); });
        }
    }

    ~TrainerWindow() override {
        persistReviewDrafts();
        // Focus-out signals during QWidget teardown must not access destroyed members.
        for (auto *child : findChildren<QObject *>()) disconnect(child, nullptr, this, nullptr);
        streamingTimer_->stop(); streamingTimer_->disconnect(this);
        process_->disconnect(this); previewProcess_->disconnect(this); editProcess_->disconnect(this); autosaveTimer_->stop();
        settings_.setValue("workspace", workspace_->text());
        settings_.setValue("teacher_model", teacher_->currentData());
        settings_.setValue("raft_root", raftRoot_->text()); settings_.setValue("raft_model", raftModel_->text());
        for (auto *process : {process_, previewProcess_, editProcess_, exportProcess_}) {
            if (process && process->state() != QProcess::NotRunning) {
                process->kill(); process->waitForFinished(1500);
            }
        }
    }

    bool exportRunning() const { return exportProcess_ && exportProcess_->state() != QProcess::NotRunning; }
    bool taskRunning() const { return process_->state() != QProcess::NotRunning || previewProcess_->state() != QProcess::NotRunning || editProcess_->state() != QProcess::NotRunning || exportRunning(); }
    void attachSession(IPDE::ProjectSession *session) {
        session_ = session;
        session_->setActivationHandler(this, [this](const QJsonObject &request) { activateRequested(request); });
        session_->setStateHandler(this, [this](const QJsonObject &state) {
            projectOperations_ = state.value("operations").toObject(); updateCleanupActions(); updateTrainingSelection();
        });
    }
    void activateRequested(const QJsonObject &request) {
        showNormal(); raise(); activateWindow(); if (windowHandle()) windowHandle()->requestActivate();
        const QString path = request.value("dataset").toString();
        const QString section = request.value("section").toString();
        if (!path.isEmpty()) {
            if (datasetMode_ && section == "review") pendingReview_ = path;
            else pendingTrainingPath_ = path;
            if (busy_) refreshAfter_ = true; else refreshLibrary();
        }
        if (datasetMode_ && section == "prepare") tabs_->setCurrentIndex(2);
        if (datasetMode_ && section == "import") tabs_->setCurrentIndex(0);
        if (!datasetMode_) tabs_->setCurrentIndex(4);
    }
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

protected:
    void closeEvent(QCloseEvent *event) override {
        if (busy_ && job_ == "Train RAFT-Stereo") {
            requestTrainingControl("stop"); closeAfterTraining_ = true; event->ignore(); return;
        }
        if (exportRunning()) {
            closeAfterTraining_ = true; statusBar()->showMessage("Finishing checkpoint export before closing…"); event->ignore(); return;
        }
        if (!datasetMode_ || !hasPendingDatasetEdits()) { QMainWindow::closeEvent(event); return; }
        const bool recoverable = persistReviewDrafts();
        QMessageBox dialog(QMessageBox::Warning, "Dataset changes are not saved yet",
            recoverable ? "Some dataset edits are still being saved or could not be applied. They are stored for recovery when you reopen Dataset Studio." : "The pending edits could not be stored for recovery. Keep this window open and retry saving.", QMessageBox::NoButton, this);
        dialog.setObjectName("pendingDatasetEditsWarning");
        auto *save = dialog.addButton("Save and close", QMessageBox::AcceptRole);
        auto *keep = dialog.addButton("Close keeping recoverable edits", QMessageBox::DestructiveRole); keep->setEnabled(recoverable);
        auto *cancel = dialog.addButton("Cancel", QMessageBox::RejectRole); dialog.setDefaultButton(cancel); dialog.setEscapeButton(cancel);
        dialog.exec();
        if (dialog.clickedButton() == keep && recoverable) { event->accept(); return; }
        event->ignore();
        if (dialog.clickedButton() == save) { closeAfterSave_ = true; retryDatasetSaves(); }
    }

#ifdef IPDE_STUDIO_REGRESSION
public:
#else
private:
#endif
    int workerCount() const {
        if (!projectSettings_) return 0;
        projectSettings_->sync(); return qMax(0, projectSettings_->value("performance/workers", 0).toInt());
    }
    void openApp(const QString &role, const QString &section = {}, const QString &dataset = {}) {
        if (datasetMode_ && role == "trainer" && hasUnsavedReviewExclusions(dataset)) {
            if (dataset == reviewedDataset_ && !reviewEntries().isEmpty() && !reviewDrafts_.value(dataset).value("dirty").toBool()) queueReviewSave();
            trainerAfterSave_ = dataset; retryDatasetSaves(); statusBar()->showMessage("Saving your dataset edits before opening Trainer…"); return;
        }
        lastAppRequest_ = QJsonObject{{"role", role}, {"section", section}, {"dataset", dataset}};
        if (session_ && session_->openApp(role, QJsonObject{{"section", section}, {"dataset", dataset}}))
            statusBar()->showMessage("Opening " + (role == "datasets" ? QString("Dataset Studio") : QString("Trainer")) + " for this project…");
        else statusBar()->showMessage("Open this project's " + (role == "datasets" ? QString("Dataset Studio") : QString("Trainer")) + " from IPDE Studio.");
    }
    bool ownsDataset(const QString &path) const {
        const QString container = QFileInfo(QDir(workspace_->text()).filePath("datasets")).canonicalFilePath();
        const QFileInfo info(path);
        return !container.isEmpty() && !info.isSymLink() && info.isDir() && QFileInfo(info.absolutePath()).canonicalFilePath() == container;
    }
    void updateCleanupActions() {
        if (!cleanupDataset_ || !cleanupRun_) return;
        const bool training = !projectOperations_.value("trainer").toString().isEmpty();
        cleanupDataset_->setEnabled(datasetMode_ && !busy_ && !training && ownsDataset(selectedPath(datasets_)));
        cleanupDataset_->setToolTip(training ? "Finish or cancel model training in Trainer before removing a dataset." : "Remove a generated dataset after training; original photos and trained models stay in place. Requires confirmation.");
        cleanupRun_->setEnabled(!datasetMode_ && !busy_ && !selectedPath(runs_).isEmpty());
    }
    void cleanupDataset() {
        if (!datasetMode_) { openApp("datasets", "prepare", selectedPath(datasets_)); return; }
        updateCleanupActions(); if (!cleanupDataset_->isEnabled()) return;
        const QString path = selectedPath(datasets_);
        const QString message = "Permanently remove this generated dataset and its arrays?\n\n" + path + "\n\nOriginal photos and trained models are retained. Training from this dataset requires generating or importing it again. Datasets that depend on these arrays prevent removal.";
        if (QMessageBox::warning(this, "Remove generated dataset", message, QMessageBox::Cancel | QMessageBox::Yes, QMessageBox::Cancel) != QMessageBox::Yes) return;
        refreshAfter_ = true; startJob("Remove generated dataset", {"cleanup-dataset", path, "--workspace", workspace_->text(), "--confirm"});
    }
    void cleanupRun() {
        updateCleanupActions(); if (!cleanupRun_->isEnabled()) return;
        const QString checkpoint = selectedPath(runs_);
        if (QMessageBox::warning(this, "Clean run files", "Remove intermediate files from this training run?\n\n" + QFileInfo(checkpoint).absolutePath() + "\n\nThe output checkpoint and its training report are retained. Original photos and datasets remain available.", QMessageBox::Cancel | QMessageBox::Yes, QMessageBox::Cancel) != QMessageBox::Yes) return;
        refreshAfter_ = true; startJob("Clean run files", {"cleanup-run", checkpoint, "--workspace", workspace_->text(), "--confirm"});
    }
    void syncDatasetSelection(QTreeWidget *source, QTreeWidget *target) {
        const QString path = selectedPath(source);
        if (path.isEmpty() || path == selectedPath(target)) return;
        for (int i = 0; i < target->topLevelItemCount(); ++i) {
            auto *item = target->topLevelItem(i);
            if (item->data(0, Qt::UserRole).toString() == path) { target->setCurrentItem(item); return; }
        }
    }

    static QPushButton *binaryButton(const QString &caption, QWidget *parent) {
        auto *button = new QPushButton(caption + ": Off", parent); button->setCheckable(true);
        button->setAccessibleName(caption); button->setProperty("toggleLabel", caption);
        connect(button, &QPushButton::toggled, button, [button, caption](bool enabled) {
            button->setText(caption + (enabled ? ": On" : ": Off"));
        });
        return button;
    }

    static void disableItemCheckboxes(QTreeWidgetItem *item) {
        item->setFlags(item->flags() & ~(Qt::ItemIsUserCheckable | Qt::ItemIsAutoTristate | Qt::ItemIsUserTristate));
        for (int column = 0; column < item->columnCount(); ++column) item->setData(column, Qt::CheckStateRole, QVariant());
    }

    static bool reviewIncluded(QTreeWidgetItem *item) {
        if (item->childCount()) {
            for (int i=0; i<item->childCount(); ++i) if (reviewIncluded(item->child(i))) return true;
            return false;
        }
        const auto included = item->data(0, Qt::UserRole + 2);
        return !included.isValid() || included.toBool();
    }

    static void setReviewItemIncluded(QTreeWidgetItem *item, bool include) {
        disableItemCheckboxes(item);
        if (item->childCount()) { for (int i=0; i<item->childCount(); ++i) setReviewItemIncluded(item->child(i), include); }
        else item->setData(0, Qt::UserRole + 2, include);
        item->setText(5, include ? "Included" : "Removed");
    }

    static QString collectionRole(QTreeWidgetItem *item) {
        const QString role = item->data(0, Qt::UserRole + 2).toString();
        return role == "train" || role == "validation" ? role : "unused";
    }

    void setCollectionRole(QTreeWidgetItem *item, const QString &role) {
        if (busy_ || !item || !QStringList{"unused", "train", "validation"}.contains(role)) return;
        {
            QSignalBlocker blocker(collectionSources_); disableItemCheckboxes(item);
            item->setData(0, Qt::UserRole + 2, role);
            item->setText(1, role == "train" ? "Training" : role == "validation" ? "Validation" : "Not used");
        }
        collectionSources_->setCurrentItem(item, 0, QItemSelectionModel::NoUpdate);
        if (role == "validation" && splitMode_) splitMode_->setCurrentIndex(splitMode_->findData("explicit"));
        updateCollectionReadiness();
    }

    void setSelectedCollectionRole(const QString &role) {
        if (busy_) return;
        const auto selection = collectionSources_->selectedItems();
        for (auto *item : selection) setCollectionRole(item, role);
        if (!selection.isEmpty()) statusBar()->showMessage(QString("%1 dataset(s) assigned to %2.").arg(selection.size()).arg(role == "train" ? "training" : role == "validation" ? "validation" : "not used"));
    }

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
        addingPhotosHint_ = new QLabel(tab); addingPhotosHint_->setWordWrap(true); addingPhotosHint_->hide(); root->addWidget(addingPhotosHint_);
        cancelAdding_ = new QPushButton("Cancel adding photos", tab); cancelAdding_->hide(); root->addWidget(cancelAdding_);
        connect(cancelAdding_, &QPushButton::clicked, this, [this] { appendBase_.clear(); appendEdits_ = {}; appendVersionName_.clear(); addingPhotosHint_->hide(); cancelAdding_->hide(); generate_->setText("Generate dataset"); tabs_->setCurrentIndex(1); });
        auto *scroll = new QScrollArea; scroll->setWidgetResizable(true); scroll->setFrameShape(QFrame::NoFrame); scroll->setWidget(tab);
        auto *intro = new QLabel("A depth teacher is the AI model that generates estimated depth for training. Generate the dataset, then review its depth maps before training.", tab);
        intro->setWordWrap(true); root->addWidget(intro);
        goal_ = new QComboBox(tab);
        goal_->addItem("Effect / displacement map — prioritize detail", "effect/map");
        goal_->addItem("Depth estimation — prioritize distance", "depth-estimation");
        goal_->addItem("Portrait effects / masking — embedded depth and mattes", "photo-effects");
        goal_->addItem("Manual — expose every setting", "manual");
        advanced_ = binaryButton("Advanced settings", tab);
        goalHelp_ = new QLabel(tab); goalHelp_->setWordWrap(true); root->addWidget(goalHelp_);
        auto *row = new QHBoxLayout;
        datasetName_ = new QLineEdit("dataset-" + QDateTime::currentDateTime().toString("yyyyMMdd-HHmmss"), tab);
        row->addWidget(new QLabel("New dataset", tab)); row->addWidget(datasetName_, 1);
        auto *add = new QPushButton("Add photos…", tab); auto *scan = new QPushButton("Find spatial photos in folder…", tab); auto *remove = new QPushButton("Remove selected", tab);
        row->addWidget(add); row->addWidget(scan); row->addWidget(remove); root->addLayout(row);
        auto *categoryRow = new QHBoxLayout; category_ = new QLineEdit(tab); category_->setPlaceholderText("Rooms, landscapes, macro, people…");
        categoryRow->addWidget(new QLabel("Subject / dataset category", tab)); categoryRow->addWidget(category_, 1);
        useGroups_ = binaryButton("Use photo groups", tab); categoryRow->addWidget(useGroups_); root->addLayout(categoryRow);
        sources_ = new QTreeWidget(tab); sources_->setHeaderLabels({"Spatial HEIC", "Optional group — double-click to edit", "Camera", "Captured"});
        sources_->setObjectName("datasetSources");
        sources_->setMinimumHeight(100); sources_->setMaximumHeight(110);
        sources_->setSelectionMode(QAbstractItemView::ExtendedSelection); sources_->setRootIsDecorated(false);
        sources_->header()->setSectionResizeMode(0, QHeaderView::Stretch); sources_->header()->setSectionResizeMode(1, QHeaderView::Stretch);
        sources_->setItemDelegate(new GroupDelegate(sources_)); root->addWidget(sources_, 1);
        verifiedScenes_ = binaryButton("Independent scene groups", tab);
        verifiedScenes_->setToolTip("This is your declaration, not an automatic scene check. It labels the validation as scene-based. The same group labels keep related photos in one split whether enabled or disabled.");
        root->addWidget(verifiedScenes_); verifiedScenes_->setVisible(false);
        auto *sceneHelp = new QLabel("Give repeat shots of the same room, subject, or setup the same group name. Enable this only after grouping every related photo together and confirming the other groups show different scenes. This keeps validation from benefiting from scenes seen during training. If unsure, leave it off; different filenames alone are not evidence.", tab);
        sceneHelp->setWordWrap(true); root->addWidget(sceneHelp); sceneHelp->hide();
        connect(useGroups_, &QPushButton::toggled, this, [this, sceneHelp](bool grouped) { sources_->setColumnHidden(1, !grouped); verifiedScenes_->setVisible(grouped); sceneHelp->setVisible(grouped); });
        sources_->setColumnHidden(1, true);
        connect(add, &QPushButton::clicked, this, [this] {
            const auto files = QFileDialog::getOpenFileNames(this, "Add spatial HEIC photos", settings_.value("photo_folder").toString(), "HEIC / HEIF photos (*.heic *.HEIC *.heif *.HEIF *.hif *.HIF)");
            for (const QString &file : files) addSourcePhoto(file);
            if (!files.isEmpty()) settings_.setValue("photo_folder", QFileInfo(files.first()).absolutePath());
        });
        connect(scan, &QPushButton::clicked, this, [this] {
            const QString directory = QFileDialog::getExistingDirectory(this, "Choose the folder containing your original HEIC photos; subfolders included", settings_.value("photo_folder").toString());
            if (!directory.isEmpty()) { settings_.setValue("photo_folder", directory); startJob("Scan spatial photos", {"scan-spatial", directory, "--workers", QString::number(workerCount())}); }
        });
        connect(remove, &QPushButton::clicked, this, [this] { qDeleteAll(sources_->selectedItems()); });
        auto *teachers = new QHBoxLayout; teachers->addWidget(new QLabel("Generate teacher entries", tab));
        for (const auto &choice : QList<QPair<QString, QString>>{{"DepthPro", "depthpro"}, {"Depth Anything V2", "depth-anything-v2"}, {"Depth Anything 3", "depth-anything-3"}}) {
            auto *check = binaryButton(choice.first, tab); check->setProperty("model", choice.second); check->setChecked(choice.second == "depthpro");
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
        teacherDevice_ = deviceBox(processing); inputSize_ = spin(processing, 0, 8192, settings_.value("teacher_input_size", 0).toInt()); inputSize_->setSingleStep(14);
        inputSize_->setObjectName("teacherInputSize"); inputSize_->setSpecialValueText("Native source size (V2 / DA3)");
        connect(inputSize_, qOverload<int>(&QSpinBox::valueChanged), this, [this](int value) { settings_.setValue("teacher_input_size", value); });
        processingLayout->addWidget(teacherDevice_); processingLayout->addWidget(new QLabel("V2 / DA3 input size", processing)); processingLayout->addWidget(inputSize_);
        right->addRow("Inference device", processing);
        anchor_ = binaryButton("Estimate meters with DepthPro", tab);
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
        auto *teacherView = new QLabel("Teacher source: full display photo. Stereo left/right images are student inputs only.", datasetAdvanced_);
        teacherView->setWordWrap(true); advancedLayout->addWidget(teacherView);
        scaleHelp_ = new QLabel(tab); scaleHelp_->setWordWrap(true); advancedLayout->addWidget(scaleHelp_);
        auto *sizeHelp = new QLabel("Teachers receive the full display photo, never a stereo image. V2 uses the configured shortest side; DA3 uses the longest side. Set 0 to request native display dimensions on the required 14-pixel grid. The stored display map matches the display photo. DepthPro internally processes a fixed 1536-pixel grid and restores the source size; its model does not support a no-resize mode. Whether DA3 Giant at 5712×4284 fits in 64 GB is unverified: its decoder can hold several very large feature arrays. The experimental student learns these full, unregistered display labels from native stereo RGB. Stock RAFT remains a separate native-left experiment.", tab);
        sizeHelp->setWordWrap(true); advancedLayout->addWidget(sizeHelp); root->addWidget(datasetAdvanced_);
        connect(teacher_, &QComboBox::currentIndexChanged, this, [this] { teacherDefaults(); });
        for (auto *check : teacherChecks_) connect(check, &QPushButton::toggled, this, [this] { updateTeacherSizeControl(); });
        connect(anchor_, &QPushButton::toggled, this, [this] { updateScaleHelp(); });
        teacher_->setCurrentIndex(qMax(0, teacher_->findData(settings_.value("teacher_model", "depthpro"))));
        teacherDefaults();
        connect(goal_, &QComboBox::currentIndexChanged, this, [this] { applyGoal(); });
        connect(advanced_, &QPushButton::toggled, this, [this](bool visible) { datasetAdvanced_->setVisible(visible); if (trainingAdvanced_) trainingAdvanced_->setVisible(visible); });
        const QString configuredGoal = projectSettings_ ? projectSettings_->value("goal", "effect/map").toString() : settings_.value("goal", "effect/map").toString();
        goal_->setCurrentIndex(qMax(0, goal_->findData(configuredGoal))); applyGoal();
        generate_ = new QPushButton("Generate & review dataset", tab); root->addWidget(generate_);
        connect(generate_, &QPushButton::clicked, this, [this] { generateDataset(); });
        tabs_->addTab(scroll, "Add photos / new dataset");
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
            goalHelp_->setText(goal == "effect/map" ? "Detail preset: DA3 teacher on the full display photo, with optional DepthPro meter scale. Review direct display maps before training the experimental stereo-to-display student; output detail and accuracy are not guaranteed." : goal == "depth-estimation" ? "Distance preset: DepthPro estimates meters from the full display photo. Independent photo groups stay together during validation; model estimates still require visual checking." : "Portrait preset: use Photo Studio for embedded depth and composited portrait mattes. Portraits do not have the calibrated stereo pair required for stereo-student datasets; the scanner skips them.");
        }
        if (trainingAdvanced_) trainingAdvanced_->setVisible(advanced_->isChecked());
    }

    void updateScaleHelp() {
        if (teacher_->currentData().toString() == "depthpro") {
            scaleHelp_->setText("DepthPro already estimates distance in meters, so a second scale model is unnecessary. These are AI estimates; meter units do not establish physical accuracy.");
        } else {
            scaleHelp_->setText(QString("Relative depth tells you which surfaces are nearer or farther, with an arbitrary scale for each photo. %1 When enabled, DepthPro supplies an estimated meter scale while the selected teacher supplies detail. This adds inference time, inherits scale errors, and can be rejected when the models disagree. Original relative values are kept separately.")
                .arg(anchor_->isChecked() ? "Enable this when an estimated meter scale is useful; the anchor sees the same full display photo." : "With this off, the display student can learn relative labels directly; keep one unit convention per run. Meter scale is required only for physical native-stereo flow labels."));
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
        updateTeacherSizeControl();
        anchor_->setEnabled(model != "depthpro");
        updateScaleHelp();
    }

    void updateTeacherSizeControl() {
        bool configurable = teacher_->currentData() != "depthpro";
        for (auto *check : teacherChecks_) configurable |= check->isChecked() && check->property("model") != "depthpro";
        inputSize_->setEnabled(configurable);
        inputSize_->setToolTip("V2 shortest side / DA3 longest side. Zero uses the native source grid. DepthPro has a fixed 1536-pixel network regardless of this setting. Requested and actual model dimensions are recorded in the dataset.");
    }

    void buildReviewTab() {
        auto *tab = new QWidget; auto *root = new QVBoxLayout(tab);
        reviewPath_ = new QLabel("Choose a dataset from the library to manage its photos.", tab);
        reviewPath_->setWordWrap(true); reviewPath_->setTextInteractionFlags(Qt::TextSelectableByMouse); root->addWidget(reviewPath_);
        auto *photoActions = new QHBoxLayout;
        addPhotos_ = new QPushButton("Add photos…", tab); addPhotos_->setObjectName("addDatasetPhotos");
        auto *addExisting = new QPushButton("Add from dataset…", tab);
        removePhotos_ = new QPushButton("Remove selected", tab); restorePhotos_ = new QPushButton("Restore selected", tab);
        photoActions->addWidget(addPhotos_); photoActions->addWidget(addExisting); photoActions->addWidget(removePhotos_); photoActions->addWidget(restorePhotos_); photoActions->addStretch();
        if (datasetMode_) root->addLayout(photoActions); else { delete photoActions; addPhotos_->hide(); addExisting->hide(); removePhotos_->hide(); restorePhotos_->hide(); }
        connect(addPhotos_, &QPushButton::clicked, this, [this] { beginAddingPhotos(); });
        connect(addExisting, &QPushButton::clicked, this, [this] { addPreparedDataset(); });
        connect(removePhotos_, &QPushButton::clicked, this, [this] { setReviewIncluded(false); });
        connect(restorePhotos_, &QPushButton::clicked, this, [this] { setReviewIncluded(true); });
        editActions_ << addPhotos_ << addExisting << removePhotos_ << restorePhotos_;
        pendingPhotosAction_ = new QPushButton(tab); pendingPhotosAction_->hide(); root->addWidget(pendingPhotosAction_);
        connect(pendingPhotosAction_, &QPushButton::clicked, this, [this] { prepareAddedPhotoGeneration(pendingPhotoPaths_); });

        auto *body = new QSplitter(Qt::Horizontal, tab);
        reviewSamples_ = new QTreeWidget(body);
        reviewSamples_->setHeaderLabels({"Photo / teacher", "Split", "Group", "Camera", "Captured", "Status"});
        reviewSamples_->setRootIsDecorated(true); reviewSamples_->setAlternatingRowColors(true);
        reviewSamples_->setSelectionMode(QAbstractItemView::ExtendedSelection); reviewSamples_->setObjectName("datasetPhotos");
        reviewSamples_->header()->setStretchLastSection(false);
        reviewSamples_->header()->setSectionResizeMode(QHeaderView::Interactive);
        reviewSamples_->setColumnWidth(0, 170); reviewSamples_->setColumnWidth(1, 90); reviewSamples_->setColumnWidth(2, 90); reviewSamples_->setColumnWidth(3, 130); reviewSamples_->setColumnWidth(4, 155);
        if (datasetMode_) { for (int column : {2, 3, 4}) reviewSamples_->setColumnHidden(column, true); }
        reviewSamples_->setColumnWidth(5, 100);
        auto *previewScroll = new QScrollArea(body); previewScroll->setObjectName("reviewPreviewScroll");
        previewScroll->setWidgetResizable(true); previewScroll->setFrameShape(QFrame::NoFrame);
        auto *preview = new QWidget; preview->setObjectName("reviewPreviewContent");
        auto *previewRoot = new QVBoxLayout(preview);
        previewScroll->setWidget(preview);
        auto *labelRow = new QHBoxLayout; labelRow->addWidget(new QLabel("Depth to view", preview));
        reviewLabel_ = new QComboBox(preview); reviewLabel_->setMinimumWidth(0); labelRow->addWidget(reviewLabel_, 1);
        previewRoot->addLayout(labelRow);
        compareTeachers_ = binaryButton("Compare teachers", preview); compareTeachers_->setChecked(true);
        previewRoot->addWidget(compareTeachers_);
        auto *viewRow = new QHBoxLayout; viewRow->addWidget(new QLabel("Visual inspection", preview));
        visualView_ = new QComboBox(preview); visualView_->addItem("Depth map", "depth"); visualView_->addItem("50% overlay on photo", "overlay"); visualView_->addItem("Lit surface — drag to rotate", "surface"); visualView_->addItem("Teacher / baseline disagreement", "difference");
        viewRow->addWidget(visualView_, 1); previewRoot->addLayout(viewRow);
        auto *images = new QHBoxLayout;
        auto *rgbColumn = new QVBoxLayout; auto *depthColumn = new QVBoxLayout;
        sourceTitle_ = new QLabel("Source view on the same grid", preview); sourceTitle_->setWordWrap(true);
        rgbColumn->addWidget(sourceTitle_);
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
        body->addWidget(reviewSamples_); body->addWidget(previewScroll); body->setChildrenCollapsible(false); body->setSizes({420, 700}); root->addWidget(body, 1);
        auto *filterRow = new QHBoxLayout; reviewFilter_ = new QLineEdit(tab); reviewFilter_->setPlaceholderText("Filter photo, teacher, group, or date…");
        reviewCamera_ = new QComboBox(tab); reviewCamera_->addItem("All cameras"); filterRow->addWidget(reviewFilter_, 1); filterRow->addWidget(reviewCamera_); root->addLayout(filterRow);
        auto *navigation = new QHBoxLayout;
        auto *previous = new QPushButton("Previous", tab); auto *next = new QPushButton("Next", tab);
        auto *exclude = new QPushButton("Remove and next", tab); exclude->setObjectName("removePhotoAndNext"); editActions_ << exclude;
        auto *toTraining = new QPushButton("Move to training", tab); auto *toValidation = new QPushButton("Move to validation", tab);
        auto *undo = new QPushButton("Reload saved state", tab);
        toTraining->hide(); toValidation->hide();
        if (datasetMode_) {
            auto *setSplit = new QToolButton(tab); setSplit->setText("Set split…"); setSplit->setPopupMode(QToolButton::InstantPopup); auto *menu = new QMenu(setSplit);
            auto *trainingAction = menu->addAction("Move selected to training"); auto *validationAction = menu->addAction("Move selected to validation");
            connect(trainingAction, &QAction::triggered, toTraining, &QPushButton::click); connect(validationAction, &QAction::triggered, toValidation, &QPushButton::click);
            setSplit->setMenu(menu); navigation->addWidget(setSplit); navigation->addWidget(undo);
        } else undo->hide();
        editActions_ << toTraining << toValidation << undo;
        connect(toTraining, &QPushButton::clicked, this, [this] { setReviewSplit("train"); });
        connect(toValidation, &QPushButton::clicked, this, [this] { setReviewSplit("validation"); });
        connect(undo, &QPushButton::clicked, this, [this] {
            if (editProcess_ && editProcess_->state() != QProcess::NotRunning) { statusBar()->showMessage("Wait for the current save before reloading."); return; }
            if (reviewDrafts_.value(reviewedDataset_).value("dirty").toBool() && QMessageBox::question(this, "Reload saved dataset?", "Discard pending edits to this dataset and reload its saved state?", QMessageBox::Discard | QMessageBox::Cancel, QMessageBox::Cancel) != QMessageBox::Discard) return;
            reviewDrafts_.remove(reviewedDataset_); reviewAdditions_.remove(reviewedDataset_); persistReviewDrafts(); { QSignalBlocker blocker(reviewSamples_); reviewSamples_->clear(); } reviewDataset(reviewedDataset_);
        });
        navigation->addWidget(previous); navigation->addWidget(next); navigation->addWidget(exclude);
        reviewCount_ = new QLabel("No dataset loaded", tab); reviewCount_->setWordWrap(true); navigation->addWidget(reviewCount_, 1); root->addLayout(navigation);
        auto *saveRow = new QHBoxLayout;
        reviewedName_ = new QLineEdit(tab); reviewedName_->hide();
        reviewSaveStatus_ = new QLabel("All changes saved", tab); reviewSaveStatus_->setObjectName("datasetSaveStatus"); saveRow->addWidget(reviewSaveStatus_, 1);
        saveReviewed_ = new QPushButton("Save changes", tab); saveReviewed_->setObjectName("saveDatasetChanges"); saveReviewed_->setEnabled(false); saveRow->addWidget(saveReviewed_); root->addLayout(saveRow);
        connect(reviewSamples_, &QTreeWidget::currentItemChanged, this, [this] { reviewSelectionChanged(); });
        connect(reviewSamples_, &QTreeWidget::itemChanged, this, [this] { updateReviewCount(); });
        connect(reviewSamples_, &QTreeWidget::itemSelectionChanged, this, [this] { updateReviewCount(); });
        connect(reviewedName_, &QLineEdit::textChanged, this, [this] { updateReviewCount(); });
        connect(reviewLabel_, &QComboBox::currentIndexChanged, this, [this] { previewSelectedSample(); });
        connect(compareTeachers_, &QPushButton::toggled, this, [this] { previewSelectedSample(); });
        connect(visualView_, &QComboBox::currentIndexChanged, this, [this] { updateVisualView(); });
        connect(reviewFilter_, &QLineEdit::textChanged, this, [this] { filterReview(); });
        connect(reviewCamera_, &QComboBox::currentIndexChanged, this, [this] { filterReview(); });
        connect(previous, &QPushButton::clicked, this, [this] { advanceReview(-1); });
        connect(next, &QPushButton::clicked, this, [this] { advanceReview(1); });
        connect(exclude, &QPushButton::clicked, this, [this] {
            if (busy_ || reviewGenerating_ || reviewEntries().isEmpty() || (!requestedReviewPath_.isEmpty() && requestedReviewPath_ != reviewedDataset_)) return;
            if (auto *item = reviewSamples_->currentItem()) {
                if (item->childCount()) {
                    const int index = reviewSamples_->indexOfTopLevelItem(item); setReviewItemIncluded(item, false);
                    for (int nextIndex = index + 1; nextIndex < reviewSamples_->topLevelItemCount(); ++nextIndex) {
                        auto *nextPhoto = reviewSamples_->topLevelItem(nextIndex); if (!nextPhoto->isHidden()) { reviewSamples_->setCurrentItem(nextPhoto); break; }
                    }
                } else { setReviewItemIncluded(item, false); advanceReview(1); }
                queueReviewSave();
            }
        });
        connect(saveReviewed_, &QPushButton::clicked, this, [this] { saveReviewedDataset(); });
        if (!datasetMode_) {
            reviewPath_->setText("Compare a trained model with the original checkpoint using a held-out spatial photo from Train model & compare.");
            reviewSamples_->hide(); compareTeachers_->hide();
            for (auto *layout : {labelRow, filterRow, navigation, saveRow}) for (int index=0; index<layout->count(); ++index)
                if (auto *widget = layout->itemAt(index)->widget()) widget->hide();
        }
        tabs_->addTab(tab, "Photos and depth");
    }

    void reviewDataset(const QString &path, bool openTab = true) {
        if (path.isEmpty()) return;
        if (!datasetMode_) { openApp("datasets", "review", path); return; }
        if (!reviewedDataset_.isEmpty() && !reviewEntries().isEmpty()) rememberReviewDraft();
        requestedReviewPath_ = path;
        requestedReviewOpen_ = openTab;
        { QSignalBlocker blocker(reviewSamples_); reviewSamples_->clear(); }
        reviewSamples_->setEnabled(false); saveReviewed_->setEnabled(false); for (auto *button : editActions_) button->setEnabled(false);
        previewRecords_ = {}; differencePath_.clear(); rgbPreview_->reset("Loading dataset…");
        for (int i=0; i<depthPreviews_.size(); ++i) { depthPreviews_[i]->reset("Choose a photo after it loads."); depthPreviews_[i]->setVisible(i == 0); depthTitles_[i]->setVisible(i == 0); }
        previewStats_->clear();
        reviewPath_->setText("Opening " + QFileInfo(path).fileName() + "…"); reviewCount_->setText("Loading photos…");
        if (openTab) tabs_->setCurrentIndex(1);
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
        QMap<QString, bool> previousInclusion;
        QString selectedId;
        if (auto *item = selectedReviewEntry()) selectedId = item->data(0, Qt::UserRole).toJsonObject().value("id").toString();
        const QString nextDataset = result.value("dataset_path").toString(requestedReviewPath_);
        const QString actualHash = manifestHash(nextDataset), resultHash = result.value("manifest_sha256").toString();
        if (!actualHash.isEmpty() && !resultHash.isEmpty() && actualHash != resultHash && result.value("generation_state").toString() != "generating") {
            requestPreview("review", {"review-dataset", nextDataset}); return;
        }
        if (!reviewedDataset_.isEmpty() && !reviewEntries().isEmpty()) rememberReviewDraft();
        if (nextDataset == reviewedDataset_ && reviewDrafts_.value(nextDataset).value("dirty").toBool()) {
            for (auto *item : reviewEntries()) previousInclusion.insert(item->data(0, Qt::UserRole).toJsonObject().value("id").toString(), reviewIncluded(item));
        } else selectedId.clear();
        const QJsonObject draft = reviewDrafts_.value(nextDataset).value("dirty").toBool() ? reviewDrafts_.value(nextDataset) : QJsonObject{}; const auto savedKeep = draft.value("keep").toArray(), savedKnown = draft.value("known").toArray(); const auto savedSplits = draft.value("all_splits").toObject(draft.value("splits").toObject());
        reviewedDataset_ = nextDataset;
        loadSplitControls(nextDataset);
        const QString hash = result.value("manifest_sha256").toString(actualHash);
        if (!hash.isEmpty() && !draft.value("dirty").toBool()) manifestHashes_.insert(nextDataset, hash);
        reviewGenerating_ = result.value("generation_state").toString() == "generating" || result.value("splits_provisional").toBool();
        reviewPath_->setText((reviewGenerating_ ? "Generating: " : "Dataset: ") + QFileInfo(reviewedDataset_).fileName()); reviewPath_->setToolTip(reviewedDataset_);
        reviewedName_->setText(draft.value("version_name").toString(QFileInfo(reviewedDataset_).fileName() + "-edited-" + QDateTime::currentDateTime().toString("yyyyMMdd-HHmmss")));
        if (!reviewPreviews_) reviewPreviews_ = std::make_unique<QTemporaryDir>();
        QTreeWidgetItem *selected = nullptr; QMap<QString, QTreeWidgetItem *> photos; QStringList cameras;
        {
            QSignalBlocker blocker(reviewSamples_); reviewSamples_->clear();
            for (const auto &value : result.value("samples").toArray()) {
                auto sample = value.toObject(); const QString id = sample.value("id").toString();
                sample.insert("original_split", sample.value("split"));
                sample.insert("original_included", sample.value("included").toBool(!sample.value("excluded").toBool()));
                if (savedSplits.contains(id)) { sample.insert("split", savedSplits.value(id)); sample.insert("split_chosen", true); }
                const QString path = sample.value("source_path").toString();
                QString group = sample.value("requested_group").toString();
                if (group.isEmpty()) group = sample.value("group_id").toString().left(12);
                const auto metadata = sample.value("photo_metadata").toObject(); const QString camera = metadata.value("camera_model").toString();
                const QString sourceId = sample.value("source_id").toString(path); QTreeWidgetItem *photo = photos.value(sourceId);
                if (!photo) {
                    photo = new QTreeWidgetItem(reviewSamples_, {QFileInfo(path).fileName(), sample.value("split").toString(), group, camera, metadata.value("captured_at").toString()});
                    disableItemCheckboxes(photo); photo->setToolTip(0, path); photos.insert(sourceId, photo); photo->setExpanded(true);
                    if (!camera.isEmpty() && !cameras.contains(camera)) cameras << camera;
                }
                auto *item = new QTreeWidgetItem(photo, {sample.value("teacher_id").toString("Teacher"), sample.value("split").toString(), group});
                item->setData(0, Qt::UserRole, sample); item->setToolTip(0, path); setReviewItemIncluded(item, draft.contains("keep") ? (savedKnown.contains(id) ? savedKeep.contains(id) : sample.value("original_included").toBool()) : previousInclusion.value(id, sample.value("original_included").toBool()));
                if (sample.value("id").toString() == selectedId) selected = item;
                QStringList warnings; for (const auto &warning : sample.value("warnings").toArray()) warnings << warning.toString();
                item->setToolTip(1, warnings.join('\n')); item->setToolTip(2, sample.value("group_id").toString());
            }
        }
        { QSignalBlocker blocker(reviewCamera_); const QString camera = reviewCamera_->currentText(); reviewCamera_->clear(); reviewCamera_->addItem("All cameras"); cameras.sort(); reviewCamera_->addItems(cameras); reviewCamera_->setCurrentIndex(qMax(0, reviewCamera_->findText(camera))); }
        filterReview();
        reviewSamples_->setEnabled(true);
        if (requestedReviewOpen_) tabs_->setCurrentIndex(1); updateReviewCount();
        if (reviewSamples_->topLevelItemCount()) {
            QSignalBlocker blocker(reviewSamples_); reviewSamples_->setCurrentItem(selected ? selected : reviewSamples_->topLevelItem(0));
            if (tabs_->currentIndex() == 1) QTimer::singleShot(0, this, [this] { reviewSelectionChanged(); });
        }
        for (const auto &warning : result.value("warnings").toArray()) log_->appendPlainText(warning.toString());
        const auto pending = result.value("pending_photos").toArray();
        pendingPhotosAction_->setVisible(datasetMode_ && !pending.isEmpty());
        pendingPhotosAction_->setText(QString("%1 added photo(s) awaiting depth — generate targets").arg(pending.size()));
        pendingPhotoPaths_.clear(); for (const auto &path : pending) pendingPhotoPaths_ << path.toString();
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
        if (!item || reviewLabel_->currentIndex() < 0 || (!requestedReviewPath_.isEmpty() && requestedReviewPath_ != reviewedDataset_)) return;
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
        QJsonObject selectedSample;
        if (auto *item = previewKind_ == "baseline" ? nullptr : selectedReviewEntry()) {
            selectedSample = item->data(0, Qt::UserRole).toJsonObject();
            if (!selectedSample.value("training_ready").toBool()) text += "\nNo usable meter-scale training target. Review the anchor or use another teacher before training.";
        }
        const QString reference = first.value("rgb_reference").toString(selectedSample.value("rgb_reference").toString());
        const QString source = reference == "spatial_left" ? "Native left stereo grid" : reference == "spatial_right" ? "Native right stereo grid" : reference == "display" ? "Display image on its separate grid" : "Source view on the same grid";
        sourceTitle_->setText(source);
        if (!reference.isEmpty()) text += "\nSource: " + source + ".";
        const QJsonObject registration = first.value("display_registration").isObject() ? first.value("display_registration").toObject() : selectedSample.value("display_registration").toObject();
        if (!registration.isEmpty()) {
            const bool accepted = registration.value("accepted").toBool();
            text += "\nDisplay alignment " + QString(accepted ? "accepted" : "rejected");
            const QString role = registration.value("reference_role").toString();
            if (!role.isEmpty()) text += " · fit reference: " + role;
            if (registration.value("heldout_median_error_pixels").isDouble()) text += " · held-out median " + QString::number(registration.value("heldout_median_error_pixels").toDouble(), 'f', 2) + " px";
            if (registration.value("heldout_p90_error_pixels").isDouble()) text += " / p90 " + QString::number(registration.value("heldout_p90_error_pixels").toDouble(), 'f', 2) + " px";
            text += ".";
            if (!accepted) {
                if (reference == "spatial_left") text += first.value("valid_fraction").toDouble() > 0 && (selectedSample.isEmpty() || selectedSample.value("training_ready").toBool())
                    ? " Native-left targets remain usable; the display image stays separate." : " Native-left alignment is unaffected; the display image stays separate.";
                else text += " The display image cannot supply an aligned stereo target.";
            }
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
        QList<QTreeWidgetItem *> entries;
        for (auto *item : reviewEntries()) if (!item->isHidden() && !item->parent()->isHidden()) entries << item;
        const int index = entries.indexOf(selectedReviewEntry()) + direction;
        if (index >= 0 && index < entries.size()) reviewSamples_->setCurrentItem(entries[index]);
    }

    QString manifestHash(const QString &path) const {
        QFile file(QDir(path).filePath("dataset.json"));
        return file.open(QIODevice::ReadOnly) ? QString::fromLatin1(QCryptographicHash::hash(file.readAll(), QCryptographicHash::Sha256).toHex()) : QString();
    }

    bool hasPendingDatasetEdits() const {
        if (journalRecoveryNeeded_) return true;
        if (editProcess_ && editProcess_->state() != QProcess::NotRunning) return true;
        for (const auto &draft : reviewDrafts_) if (draft.value("dirty").toBool()) return true;
        return false;
    }

    bool persistReviewDrafts() {
        if (!datasetMode_ || !workspace_) return true;
        const QString workspace = draftWorkspace_.isEmpty() ? workspace_->text() : draftWorkspace_;
        QJsonObject drafts;
        for (auto it = reviewDrafts_.cbegin(); it != reviewDrafts_.cend(); ++it) {
            if (!it.value().value("dirty").toBool()) continue;
            auto draft = it.value(); QJsonArray additions;
            for (const auto &path : reviewAdditions_.value(it.key())) additions.append(path);
            draft.insert("additions", additions); drafts.insert(it.key(), draft);
        }
        const QString path = QDir(workspace).filePath(".dataset-studio-edits.json");
        if (journalRecoveryNeeded_) {
            const QString backup = path + ".unreadable-" + QDateTime::currentDateTimeUtc().toString("yyyyMMdd-HHmmss-zzz");
            if (!QFile::copy(path, backup)) { draftStorageError_ = true; statusBar()->showMessage("The previous edit journal could not be backed up. Keep this window open; pending edits have not been discarded."); return false; }
            journalRecoveryNeeded_ = false; log_->appendPlainText("Previous edit journal preserved for recovery: " + backup);
        }
        if (drafts.isEmpty() && !QFileInfo::exists(path)) return true;
        QSaveFile file(path);
        const QByteArray bytes = QJsonDocument(QJsonObject{{"schema", "ipde-dataset-studio-edits-v1"}, {"datasets", drafts}}).toJson();
        const bool saved = QDir().mkpath(workspace) && file.open(QIODevice::WriteOnly) && file.write(bytes) == bytes.size() && file.commit();
        draftStorageError_ = !saved;
        if (!saved && statusBar()) statusBar()->showMessage("Could not store pending dataset edits. Keep this window open and retry Save changes.");
        return saved;
    }

    void loadReviewDrafts() {
        if (!datasetMode_) return;
        const QString workspace = QDir(workspace_->text()).absolutePath();
        if (workspace == draftWorkspace_) return;
        if (!draftWorkspace_.isEmpty() && !persistReviewDrafts()) { workspace_->setText(draftWorkspace_); return; }
        journalRecoveryNeeded_ = false; draftStorageError_ = false;
        draftWorkspace_ = workspace; reviewDrafts_.clear(); reviewAdditions_.clear(); manifestHashes_.clear();
        QFile file(QDir(workspace).filePath(".dataset-studio-edits.json"));
        if (!file.exists()) return;
        if (!file.open(QIODevice::ReadOnly)) { draftStorageError_ = true; journalRecoveryNeeded_ = true; statusBar()->showMessage("Pending dataset edits could not be loaded. Check the workspace folder permissions."); return; }
        QJsonParseError error; const auto document = QJsonDocument::fromJson(file.readAll(), &error);
        if (error.error != QJsonParseError::NoError || document.object().value("schema").toString() != "ipde-dataset-studio-edits-v1") {
            draftStorageError_ = true; journalRecoveryNeeded_ = true; statusBar()->showMessage("The saved edit journal could not be read. Its file has been preserved."); return;
        }
        const auto drafts = document.object().value("datasets").toObject();
        for (auto it = drafts.begin(); it != drafts.end(); ++it) {
            auto draft = it.value().toObject(); if (!draft.value("dirty").toBool()) continue;
            reviewDrafts_.insert(it.key(), draft); manifestHashes_.insert(it.key(), draft.value("manifest_sha256").toString());
            for (const auto &addition : draft.value("additions").toArray()) reviewAdditions_[it.key()] << addition.toString();
            editSerial_ = qMax(editSerial_, draft.value("serial").toInt());
        }
        if (hasPendingDatasetEdits()) { statusBar()->showMessage("Recovered pending dataset edits. Unapplied changes remain available in Photos and depth."); if (autosaveTimer_ && !autosavePaused_) autosaveTimer_->start(); }
    }

    void rememberReviewDraft() {
        if (reviewedDataset_.isEmpty() || reviewEntries().isEmpty()) return;
        QJsonObject draft = selectedReviewEdits(); QJsonArray known; QJsonObject allSplits;
        for (auto *item : reviewEntries()) {
            const auto sample = item->data(0, Qt::UserRole).toJsonObject(); const QString id = sample.value("id").toString(); known.append(id);
            if (sample.contains("original_split") && (sample.value("split_chosen").toBool() || sample.value("split") != sample.value("original_split"))) allSplits.insert(id, sample.value("split"));
        }
        const auto previous = reviewDrafts_.value(reviewedDataset_);
        for (const QString &key : {"dirty", "serial", "manifest_sha256", "save_error", "pending_photos", "validation_fraction", "seed", "temporary_additions", "in_flight"}) if (previous.contains(key)) draft.insert(key, previous.value(key));
        draft.insert("known", known); draft.insert("all_splits", allSplits); draft.insert("version_name", reviewedName_->text());
        if (draft.contains("validation_fraction")) { draft.remove("splits"); draft.remove("all_splits"); }
        reviewDrafts_.insert(reviewedDataset_, draft);
        if (draft.value("dirty").toBool()) persistReviewDrafts();
    }

    void queueReviewSave() {
        if (!datasetMode_ || reviewedDataset_.isEmpty() || reviewGenerating_ || reviewEntries().isEmpty()) return;
        rememberReviewDraft(); auto draft = reviewDrafts_.value(reviewedDataset_);
        draft.insert("dirty", true); draft.insert("serial", ++editSerial_); draft.remove("save_error");
        if (!draft.contains("manifest_sha256")) draft.insert("manifest_sha256", manifestHashes_.value(reviewedDataset_, manifestHash(reviewedDataset_)));
        reviewDrafts_.insert(reviewedDataset_, draft); persistReviewDrafts();
        updateReviewCount(); if (!autosavePaused_) autosaveTimer_->start();
    }

    void retryDatasetSaves() {
        for (auto it = reviewDrafts_.begin(); it != reviewDrafts_.end(); ++it) { auto draft = it.value(); draft.remove("save_error"); it.value() = draft; }
        persistReviewDrafts(); startNextDatasetSave();
    }

    void startNextDatasetSave() {
        if (!datasetMode_ || busy_ || !editProcess_ || editProcess_->state() != QProcess::NotRunning) return;
        for (auto it = reviewDrafts_.cbegin(); it != reviewDrafts_.cend(); ++it) {
            if (!it.value().value("dirty").toBool() || !it.value().value("save_error").toString().isEmpty()) continue;
            editingDataset_ = it.key(); const auto pending = it.value();
            savingDraft_ = pending.value("in_flight").toObject();
            if (savingDraft_.isEmpty()) {
                savingDraft_ = pending; savingDraft_.remove("in_flight");
                QJsonArray additions; for (const auto &path : reviewAdditions_.value(editingDataset_)) additions.append(path); savingDraft_.insert("additions", additions);
                savingDraft_.insert("operation_id", QUuid::createUuid().toString(QUuid::WithoutBraces));
                auto journalDraft = pending; journalDraft.insert("in_flight", savingDraft_); reviewDrafts_.insert(editingDataset_, journalDraft);
            }
            // Journal the complete attempted operation before launching it. A
            // reopened window replays this snapshot first, so an acknowledged
            // commit can safely rebase edits made while that save was running.
            if (!persistReviewDrafts()) {
                editStdout_ = QJsonDocument(QJsonObject{{"error", "Could not store the pending save for recovery. Keep this window open and retry Save changes."}}).toJson();
                finishDatasetSave(1, QProcess::NormalExit); return;
            }
            QJsonObject edits;
            for (const QString &key : {"keep", "splits", "validation_fraction", "seed", "pending_photos"}) if (savingDraft_.contains(key)) edits.insert(key, savingDraft_.value(key));
            editSaveFile_ = std::make_unique<QTemporaryFile>();
            const auto bytes = QJsonDocument(edits).toJson();
            if (!editSaveFile_->open() || editSaveFile_->write(bytes) != bytes.size() || !editSaveFile_->flush()) {
                editStdout_ = QJsonDocument(QJsonObject{{"error", "Could not write the dataset save instructions."}}).toJson(); finishDatasetSave(1, QProcess::CrashExit); return;
            }
            QStringList args{scriptPath(), "--json", "update-dataset", editingDataset_, "--edits-json", editSaveFile_->fileName()};
            const QString hash = savingDraft_.value("manifest_sha256").toString(); if (!hash.isEmpty()) args << "--expected-manifest-sha256" << hash;
            const QString operation = savingDraft_.value("operation_id").toString(); if (!operation.isEmpty()) args << "--operation-id" << operation;
            for (const auto &path : savingDraft_.value("additions").toArray()) args << "--add-dataset" << path.toString();
            editStdout_.clear(); editProcess_->setProgram(pythonPath()); editProcess_->setArguments(args); editProcess_->start();
            reviewSaveStatus_->setText("Saving changes…"); statusBar()->showMessage("Saving dataset metadata; image and depth files stay in place."); return;
        }
        if (!trainerAfterSave_.isEmpty()) { const QString path = trainerAfterSave_; trainerAfterSave_.clear(); openApp("trainer", "train", path); }
        if (closeAfterSave_ && !hasPendingDatasetEdits()) { closeAfterSave_ = false; QTimer::singleShot(0, this, [this] { close(); }); }
    }

    void finishDatasetSave(int code, QProcess::ExitStatus status) {
        if (editingDataset_.isEmpty()) return;
        editStdout_ += editProcess_->readAllStandardOutput(); const auto document = QJsonDocument::fromJson(editStdout_.trimmed());
        const QString path = editingDataset_; const auto savedDraft = savingDraft_;
        editingDataset_.clear(); savingDraft_ = {}; editSaveFile_.reset();
        const bool success = code == 0 && status == QProcess::NormalExit && document.isObject() && document.object().value("dataset_path").toString() == path;
        auto pending = reviewDrafts_.value(path);
        if (!success) {
            const QString error = document.object().value("error").toString("Dataset changes could not be saved.");
            if (status == QProcess::NormalExit && code != 0 && document.object().contains("error")) {
                const QString operation = savedDraft.value("operation_id").toString();
                const QString committedOperation = readJson(QDir(path).filePath("dataset.json")).value("dataset_update").toObject().value("operation_id").toString();
                // Cancellation can report an error just after atomic commit.
                // Retain its snapshot for proof-verified replay in that case.
                if (operation.isEmpty() || committedOperation != operation) pending.remove("in_flight");
            }
            pending.insert("dirty", true); pending.insert("save_error", error); reviewDrafts_.insert(path, pending); persistReviewDrafts();
            reviewSaveStatus_->setText("Not saved — retry Save changes"); statusBar()->showMessage("Dataset changes are pending: " + error); log_->appendPlainText(error);
            closeAfterSave_ = false; trainerAfterSave_.clear(); updateReviewCount(); return;
        }
        const auto result = document.object(); const QString hash = result.value("manifest_sha256").toString(); manifestHashes_.insert(path, hash);
        const bool newer = pending.value("serial").toInt() != savedDraft.value("serial").toInt();
        if (!newer) { reviewDrafts_.remove(path); reviewAdditions_.remove(path); }
        else { pending.remove("in_flight"); pending.insert("manifest_sha256", hash); reviewDrafts_.insert(path, pending); }
        const auto manifest = readJson(QDir(path).filePath("dataset.json")); QMap<QString, QJsonObject> savedSamples;
        for (const auto &value : manifest.value("samples").toArray()) savedSamples.insert(value.toObject().value("id").toString(), value.toObject());
        if (newer && pending.contains("keep")) {
            auto keep = pending.value("keep").toArray(); auto known = pending.value("known").toArray();
            const auto previouslyKnown = savedDraft.value("known").toArray();
            for (auto it = savedSamples.cbegin(); it != savedSamples.cend(); ++it) {
                if (!previouslyKnown.isEmpty() && !previouslyKnown.contains(it.key()) && !known.contains(it.key())) { known.append(it.key()); if (!it.value().value("excluded").toBool()) keep.append(it.key()); }
            }
            pending.insert("keep", keep); pending.insert("known", known); pending.insert("manifest_sha256", hash); reviewDrafts_.insert(path, pending);
        }
        if (reviewedDataset_ == path && !reviewEntries().isEmpty()) {
            QSignalBlocker blocker(reviewSamples_);
            for (auto *item : reviewEntries()) {
                auto sample = item->data(0, Qt::UserRole).toJsonObject(); const QString id = sample.value("id").toString(); const auto record = savedSamples.value(id);
                const bool included = record.isEmpty() ? (savedDraft.contains("keep") ? savedDraft.value("keep").toArray().contains(id) : sample.value("original_included").toBool(true)) : !record.value("excluded").toBool();
                const QString split = record.value("split").toString(savedDraft.value("splits").toObject().value(id).toString(sample.value("original_split").toString(sample.value("split").toString())));
                sample.insert("original_included", included); sample.insert("original_split", split);
                if (!newer || !sample.value("split_chosen").toBool()) { sample.insert("split", split); sample.remove("split_chosen"); item->setText(1, split); }
                if (!newer) setReviewItemIncluded(item, included);
                item->setData(0, Qt::UserRole, sample);
            }
            for (int i=0; i<reviewSamples_->topLevelItemCount(); ++i) {
                auto *photo = reviewSamples_->topLevelItem(i); QSet<QString> splits;
                for (int j=0; j<photo->childCount(); ++j) splits.insert(photo->child(j)->text(1));
                photo->setText(1, splits.size() == 1 ? *splits.begin() : "Mixed");
            }
        }
        const auto summary = result.value("summary").toObject();
        for (int i=0; i<datasets_->topLevelItemCount(); ++i) {
            auto *item = datasets_->topLevelItem(i); if (item->data(0, Qt::UserRole).toString() != path) continue;
            auto metadata = item->data(0, Qt::UserRole + 1).toJsonObject();
            for (const auto &pair : {qMakePair(QString("sample_count"), QString("samples")), qMakePair(QString("train_count"), QString("train_samples")), qMakePair(QString("validation_count"), QString("validation_samples"))}) if (summary.contains(pair.second)) metadata.insert(pair.first, summary.value(pair.second));
            for (const QString &key : {QString("training_eligibility"), QString("raft_training_eligibility"), QString("training_eligibility_by_mode"), QString("raft_training_eligibility_by_mode"), QString("max_native_stereo_pixels")}) {
                if (result.contains(key)) metadata.insert(key, result.value(key)); else metadata.remove(key);
            }
            item->setData(0, Qt::UserRole + 1, metadata); item->setText(1, QString::number(metadata.value("sample_count").toInt())); item->setText(2, QString("%1 / %2").arg(metadata.value("train_count").toInt()).arg(metadata.value("validation_count").toInt()));
        }
        for (int i=0; i<collectionSources_->topLevelItemCount(); ++i) { auto *item = collectionSources_->topLevelItem(i); if (item->data(0, Qt::UserRole).toString() == path && summary.contains("samples")) item->setText(3, QString::number(summary.value("samples").toInt())); }
        persistReviewDrafts(); updateReviewCount(); updateTrainingSelection();
        reviewSaveStatus_->setText(newer ? "Saving latest changes…" : "All changes saved"); statusBar()->showMessage("Dataset changes saved in place.");
        if (!newer) for (const auto &value : savedDraft.value("temporary_additions").toArray()) {
            const QString addition = value.toString(); const QFileInfo info(addition);
            if (info.fileName().startsWith(".added-") && QFileInfo(info.absolutePath()).canonicalFilePath() == QFileInfo(QDir(workspace_->text()).filePath("datasets")).canonicalFilePath() && !info.isSymLink()) QDir(addition).removeRecursively();
        }
        if (!newer && reviewedDataset_ == path && (result.value("added_samples").toInt() || !result.value("pending_photos").toArray().isEmpty())) reviewDataset(path, false);
        if (!autosavePaused_) autosaveTimer_->start();
        else if (!trainerAfterSave_.isEmpty() || closeAfterSave_) startNextDatasetSave();
    }

    QJsonObject selectedReviewEdits() const {
        QJsonArray keep; QJsonObject splits;
        for (auto *item : reviewEntries()) {
            const auto sample = item->data(0, Qt::UserRole).toJsonObject(); const QString id = sample.value("id").toString();
            if (reviewIncluded(item)) keep.append(id);
            if ((sample.value("split_chosen").toBool() || sample.value("split") != sample.value("original_split")) && sample.contains("original_split")) splits.insert(id, sample.value("split"));
        }
        return {{"keep", keep}, {"splits", splits}};
    }

    QList<QTreeWidgetItem *> selectedReviewEntries() const {
        QList<QTreeWidgetItem *> result;
        for (auto *item : reviewSamples_->selectedItems()) {
            if (item->childCount()) { for (int i=0; i<item->childCount(); ++i) if (!result.contains(item->child(i))) result << item->child(i); }
            else if (!result.contains(item)) result << item;
        }
        return result;
    }

    void setReviewIncluded(bool include) {
        if (busy_ || reviewGenerating_ || reviewEntries().isEmpty() || (!requestedReviewPath_.isEmpty() && requestedReviewPath_ != reviewedDataset_)) return;
        { QSignalBlocker blocker(reviewSamples_); for (auto *item : selectedReviewEntries()) setReviewItemIncluded(item, include); }
        queueReviewSave();
    }

    void setReviewSplit(const QString &split) {
        if (busy_ || reviewGenerating_ || (split != "train" && split != "validation") || (!requestedReviewPath_.isEmpty() && requestedReviewPath_ != reviewedDataset_)) return;
        QSet<QString> groups, sources;
        for (auto *item : selectedReviewEntries()) {
            const auto sample = item->data(0, Qt::UserRole).toJsonObject(); const QString group = sample.value("management_group_id").toString(sample.value("group_id").toString());
            if (!group.isEmpty()) groups.insert(group);
            const QString source = sample.value("source_id").toString(sample.value("source_path").toString()); if (!source.isEmpty()) sources.insert(source);
        }
        if (groups.isEmpty() && sources.isEmpty()) return;
        auto draft = reviewDrafts_.value(reviewedDataset_); draft.remove("validation_fraction"); draft.remove("seed"); reviewDrafts_.insert(reviewedDataset_, draft);
        trainerAfterSave_.clear();
        {
            QSignalBlocker blocker(reviewSamples_);
            for (auto *item : reviewEntries()) {
                auto sample = item->data(0, Qt::UserRole).toJsonObject();
                if (!groups.contains(sample.value("management_group_id").toString(sample.value("group_id").toString())) && !sources.contains(sample.value("source_id").toString(sample.value("source_path").toString()))) continue;
                if (!sample.contains("original_split")) sample.insert("original_split", sample.value("split"));
                sample.insert("split", split); sample.insert("split_chosen", true); item->setData(0, Qt::UserRole, sample); item->setText(1, split); item->parent()->setText(1, split);
            }
        }
        statusBar()->showMessage("Split changed for the selected photos and their related capture group."); queueReviewSave();
    }

    void updateReviewCount() {
        if (!saveReviewed_ || !reviewSamples_) return;
        QSignalBlocker treeBlocker(reviewSamples_);
        int kept = 0, train = 0, validation = 0, photos = 0, changed = 0;
        const auto entries = reviewEntries();
        for (auto *item : entries) {
            const bool included = reviewIncluded(item); item->setText(5, included ? "Included" : "Removed");
            QFont font = item->font(0); font.setStrikeOut(!included); item->setFont(0, font);
            if (included) { ++kept; train += item->text(1) == "train"; validation += item->text(1) == "validation"; }
            const auto sample = item->data(0, Qt::UserRole).toJsonObject();
            changed += included != sample.value("original_included").toBool(true) || (sample.contains("original_split") && (sample.value("split_chosen").toBool() || sample.value("split") != sample.value("original_split")));
        }
        for (int i=0; i<reviewSamples_->topLevelItemCount(); ++i) {
            auto *photo = reviewSamples_->topLevelItem(i); const bool included = reviewIncluded(photo); int includedTargets = 0;
            for (int j=0; j<photo->childCount(); ++j) includedTargets += reviewIncluded(photo->child(j));
            photo->setText(5, !includedTargets ? "Removed" : includedTargets == photo->childCount() ? "Included" : "Some removed");
            photos += included; QFont font = photo->font(0); font.setStrikeOut(!included); photo->setFont(0, font);
        }
        const auto draft = reviewDrafts_.value(reviewedDataset_); const bool pending = draft.value("dirty").toBool();
        QString text = entries.isEmpty() ? "Choose a dataset to view its photos." : QString("%1 photos · %2 targets included · %3 train / %4 validation%5").arg(photos).arg(kept).arg(train).arg(validation).arg(changed || pending ? " · Saving changes" : " · Saved");
        if (kept && (!train || !validation)) text += " · Set aside a related photo group for validation before training.";
        if (!reviewAdditions_.value(reviewedDataset_).isEmpty()) text += " · Adding prepared photos";
        reviewCount_->setText(text);
        const bool editable = datasetMode_ && !entries.isEmpty() && !reviewGenerating_ && !busy_ && (requestedReviewPath_.isEmpty() || requestedReviewPath_ == reviewedDataset_);
        saveReviewed_->setEnabled(datasetMode_ && !busy_ && !reviewGenerating_ && pending && editProcess_ && editProcess_->state() == QProcess::NotRunning);
        if (reviewSaveStatus_) reviewSaveStatus_->setText(!draft.value("save_error").toString().isEmpty() ? "Not saved — retry Save changes" : pending ? "Saving changes…" : "All changes saved");
        for (auto *button : editActions_) button->setEnabled(editable);
        addPhotos_->setEnabled(datasetMode_ && !reviewedDataset_.isEmpty() && !reviewGenerating_ && !busy_ && (requestedReviewPath_.isEmpty() || requestedReviewPath_ == reviewedDataset_));
        removePhotos_->setEnabled(editable && !reviewSamples_->selectedItems().isEmpty()); restorePhotos_->setEnabled(removePhotos_->isEnabled());
        updateCollectionReadiness();
    }

    bool writeReviewEdits(const QJsonObject &edits) {
        editsFile_ = std::make_unique<QTemporaryFile>();
        if (!editsFile_->open() || editsFile_->write(QJsonDocument(edits).toJson()) < 0 || !editsFile_->flush()) {
            log_->appendPlainText("Could not save the edit instructions."); editsFile_.reset(); return false;
        }
        return true;
    }

    void saveDatasetVersion(const QString &source, const QJsonObject &edits, const QStringList &additions = {}, const QStringList &temporaryAdditions = {}) {
        if (source.isEmpty()) return;
        if (source == reviewedDataset_ && !reviewEntries().isEmpty()) rememberReviewDraft();
        auto draft = reviewDrafts_.value(source);
        if (!draft.contains("known")) { QJsonArray known; for (const auto &sample : readJson(QDir(source).filePath("dataset.json")).value("samples").toArray()) known.append(sample.toObject().value("id")); draft.insert("known", known); }
        for (auto it = edits.begin(); it != edits.end(); ++it) draft.insert(it.key(), it.value());
        if (!draft.contains("manifest_sha256")) draft.insert("manifest_sha256", manifestHashes_.value(source, manifestHash(source)));
        draft.insert("dirty", true); draft.insert("serial", ++editSerial_); draft.remove("save_error");
        for (const auto &addition : additions) if (!reviewAdditions_[source].contains(addition)) reviewAdditions_[source] << addition;
        auto temporary = draft.value("temporary_additions").toArray(); for (const auto &addition : temporaryAdditions) if (!temporary.contains(addition)) temporary.append(addition); if (!temporary.isEmpty()) draft.insert("temporary_additions", temporary);
        reviewDrafts_.insert(source, draft); persistReviewDrafts(); updateReviewCount(); startNextDatasetSave();
    }

    void saveReviewedDataset() {
        if (!datasetMode_) { openApp("datasets", "review", reviewedDataset_); return; }
        if (reviewDrafts_.value(reviewedDataset_).value("dirty").toBool()) retryDatasetSaves();
        else if (!reviewedDataset_.isEmpty() && !reviewEntries().isEmpty()) { queueReviewSave(); startNextDatasetSave(); }
    }

    void loadSplitControls(const QString &path) {
        if (!datasetMode_ || !validationFraction_ || path.isEmpty() || path == splitControlsDataset_) return;
        splitControlsDataset_ = path; const auto manifest = readJson(QDir(path).filePath("dataset.json")), draft = reviewDrafts_.value(path);
        const auto fraction = draft.value("validation_fraction").isDouble() ? draft.value("validation_fraction") : manifest.value("validation_fraction");
        const auto seed = draft.contains("seed") ? draft.value("seed") : manifest.value("split_seed");
        QSignalBlocker fractionSignals(validationFraction_), seedSignals(splitSeed_);
        validationFraction_->setValue(fraction.isDouble() ? fraction.toDouble() * 100.0 : 20.0);
        splitSeed_->setValue(seed.isDouble() ? seed.toInt() : 42);
    }

    void applySplitAndOpenTrainer() { queueAutomaticSplit(true); }

    void queueAutomaticSplit(bool openTrainer) {
        const QString path = selectedPath(datasets_);
        if (!datasetMode_ || path.isEmpty() || busy_) return;
        if (splitMode_->currentData().toString() == "explicit") {
            if (openTrainer) statusBar()->showMessage("For designated validation datasets, create a combined training set. To split this dataset in place, choose random validation."); return;
        }
        if (path == reviewedDataset_ && !reviewEntries().isEmpty()) rememberReviewDraft();
        auto draft = reviewDrafts_.value(path); draft.remove("splits"); draft.remove("all_splits");
        draft.insert("validation_fraction", validationFraction_->value() / 100.0); draft.insert("seed", splitSeed_->value());
        draft.insert("dirty", true); draft.insert("serial", ++editSerial_); draft.remove("save_error");
        if (!draft.contains("manifest_sha256")) draft.insert("manifest_sha256", manifestHashes_.value(path, manifestHash(path)));
        reviewDrafts_.insert(path, draft); if (openTrainer) trainerAfterSave_ = path; persistReviewDrafts(); updateReviewCount();
        if (openTrainer) startNextDatasetSave(); else if (!autosavePaused_) autosaveTimer_->start();
    }

    void prepareAddedPhotoGeneration(const QStringList &files) {
        if (files.isEmpty()) return;
        appendBase_ = reviewedDataset_; appendEdits_ = {}; appendVersionName_.clear();
        sources_->clear(); for (const auto &file : files) addSourcePhoto(file);
        settings_.setValue("photo_folder", QFileInfo(files.first()).absolutePath());
        datasetName_->setText(".added-" + QDateTime::currentDateTime().toString("yyyyMMdd-HHmmss-zzz"));
        addingPhotosHint_->setText("These photos are registered in " + QFileInfo(appendBase_).fileName() + ". Generate only their new depth targets; existing photos and depth files stay in place.");
        addingPhotosHint_->show(); cancelAdding_->show(); generate_->setText("Generate depth and add to dataset"); tabs_->setCurrentIndex(0);
    }

    void beginAddingPhotos() {
        if (reviewedDataset_.isEmpty() || busy_ || reviewGenerating_) return;
        const auto files = QFileDialog::getOpenFileNames(this, "Add photos to " + QFileInfo(reviewedDataset_).fileName(), settings_.value("photo_folder").toString(), "HEIC / HEIF photos (*.heic *.HEIC *.heif *.HEIF *.hif *.HIF)");
        if (files.isEmpty()) return;
        QJsonArray pending = reviewDrafts_.value(reviewedDataset_).value("pending_photos").toArray();
        if (pending.isEmpty()) pending = readJson(QDir(reviewedDataset_).filePath("dataset.json")).value("pending_photos").toArray();
        for (const auto &file : files) if (!pending.contains(file)) pending.append(file);
        saveDatasetVersion(reviewedDataset_, {{"pending_photos", pending}});
        prepareAddedPhotoGeneration(files);
    }

    void addPreparedDataset() {
        if (reviewedDataset_.isEmpty() || busy_) return;
        QDialog dialog(this); dialog.setWindowTitle("Add photos from another dataset"); auto *layout = new QVBoxLayout(&dialog);
        auto *hint = new QLabel("Add prepared photos to this dataset. Their full-quality files are linked in place without reading or copying existing arrays.", &dialog); hint->setWordWrap(true); layout->addWidget(hint);
        auto *choice = new QComboBox(&dialog);
        for (int i=0; i<datasets_->topLevelItemCount(); ++i) { auto *item = datasets_->topLevelItem(i); const QString path = item->data(0, Qt::UserRole).toString(); if (path != reviewedDataset_) choice->addItem(item->text(0), path); }
        layout->addWidget(choice); auto *buttons = new QDialogButtonBox(QDialogButtonBox::Cancel | QDialogButtonBox::Ok, &dialog); buttons->button(QDialogButtonBox::Ok)->setText("Add photos"); buttons->button(QDialogButtonBox::Ok)->setEnabled(choice->count() > 0); layout->addWidget(buttons);
        connect(buttons, &QDialogButtonBox::accepted, &dialog, &QDialog::accept); connect(buttons, &QDialogButtonBox::rejected, &dialog, &QDialog::reject);
        if (dialog.exec() == QDialog::Accepted) saveDatasetVersion(reviewedDataset_, selectedReviewEdits(), {choice->currentData().toString()});
    }

    void buildCollectionTab() {
        auto *tab = new QWidget; auto *root = new QVBoxLayout(tab);
        auto *scroll = new QScrollArea; scroll->setWidgetResizable(true); scroll->setFrameShape(QFrame::NoFrame); scroll->setWidget(tab);
        auto *intro = new QLabel("Set Validation size, then apply the split to the selected dataset and open Trainer. This saves only metadata, with no new dataset or array copy. For individual assignments, use Set split in Photos and depth. Combining several datasets is optional.", tab); intro->setWordWrap(true); root->addWidget(intro);
        collectionSources_ = new QTreeWidget(tab); collectionSources_->setHeaderLabels({"Dataset", "Use", "Category", "Depth targets"});
        collectionSources_->setObjectName("collectionDatasets"); collectionSources_->setSelectionMode(QAbstractItemView::ExtendedSelection);
        collectionSources_->setRootIsDecorated(false); collectionSources_->header()->setSectionResizeMode(0, QHeaderView::Stretch); collectionSources_->setColumnWidth(1, 130); collectionSources_->setMinimumHeight(140); root->addWidget(collectionSources_, 1);
        auto *roleActions = new QHBoxLayout;
        useDatasetForTraining_ = new QPushButton("Use for training", tab); useDatasetForValidation_ = new QPushButton("Use for validation", tab); skipDataset_ = new QPushButton("Do not use", tab);
        useDatasetForTraining_->setObjectName("useDatasetForTraining"); useDatasetForValidation_->setObjectName("useDatasetForValidation"); skipDataset_->setObjectName("skipDataset");
        useDatasetForValidation_->setToolTip("Hold out every photo in the selected datasets. This chooses the designated validation datasets strategy.");
        for (auto *button : {useDatasetForTraining_, useDatasetForValidation_, skipDataset_}) roleActions->addWidget(button);
        roleActions->addStretch(); root->addLayout(roleActions);
        connect(useDatasetForTraining_, &QPushButton::clicked, this, [this] { setSelectedCollectionRole("train"); });
        connect(useDatasetForValidation_, &QPushButton::clicked, this, [this] { setSelectedCollectionRole("validation"); });
        connect(skipDataset_, &QPushButton::clicked, this, [this] { setSelectedCollectionRole("unused"); });
        connect(collectionSources_, &QTreeWidget::itemSelectionChanged, this, [this] { updateCollectionReadiness(); });
        auto *form = new QFormLayout;
        collectionForm_ = form;
        collectionName_ = new QLineEdit("training-set-" + QDateTime::currentDateTime().toString("yyyyMMdd-HHmmss"), tab); form->addRow("Training set name", collectionName_);
        form->setFieldGrowthPolicy(QFormLayout::AllNonFixedFieldsGrow);
        splitMode_ = new QComboBox(tab); splitMode_->addItem("Random across all selected datasets", "global-random"); splitMode_->addItem("Same number chosen from each dataset", "equal-per-dataset"); splitMode_->addItem("Use designated validation datasets", "explicit");
        form->addRow("Validation strategy", splitMode_);
        validationFraction_ = new QDoubleSpinBox(tab); validationFraction_->setRange(0, 100); validationFraction_->setDecimals(0); validationFraction_->setSingleStep(5); validationFraction_->setValue(20); validationFraction_->setSuffix("% for validation"); form->addRow("Validation size", validationFraction_);
        validationCount_ = spin(tab, 0, 100000, 0); validationCount_->setSpecialValueText("Automatic equal count"); form->addRow("Equal validation groups per dataset", validationCount_);
        groupingPolicy_ = new QComboBox(tab); groupingPolicy_->addItem("Keep photo groups together", "preserve"); groupingPolicy_->addItem("Treat as one pile — ignore authored groups", "ignore"); form->addRow("Groups", groupingPolicy_);
        splitSeed_ = spin(tab, 0, 2147483647, 42); form->addRow("Repeatable random seed", splitSeed_); root->addLayout(form);
        form->setRowVisible(groupingPolicy_, false); form->setRowVisible(splitSeed_, false);
        auto *advancedSplit = new QToolButton(tab); advancedSplit->setText("Advanced split options"); advancedSplit->setCheckable(true); root->addWidget(advancedSplit);
        connect(advancedSplit, &QToolButton::toggled, this, [this, form](bool visible) { form->setRowVisible(groupingPolicy_, visible); form->setRowVisible(splitSeed_, visible); });
        splitHelp_ = new QLabel(tab); splitHelp_->setWordWrap(true); root->addWidget(splitHelp_);
        auto updateSplit = [this] {
            const QString mode = splitMode_->currentData().toString(); validationFraction_->setEnabled(mode != "explicit"); validationCount_->setEnabled(mode == "equal-per-dataset");
            collectionSources_->setColumnHidden(1, false);
            collectionForm_->setRowVisible(validationFraction_, mode != "explicit"); collectionForm_->setRowVisible(validationCount_, mode == "equal-per-dataset");
            splitHelp_->setText(mode == "explicit" ? "Select the validation dataset rows and click Use for validation. They are held out entirely; overlapping source photos and groups are rejected." : mode == "equal-per-dataset" ? "Each dataset contributes the same count of randomly chosen independent photo groups. Zero uses an automatic count based on the smallest dataset. Related captures and all teachers remain together." : "Validation groups are drawn randomly from the combined selected datasets. Larger datasets usually contribute more validation photos. The seed repeats the same selection.");
        };
        connect(splitMode_, &QComboBox::currentIndexChanged, this, [this, updateSplit] { updateSplit(); updateCollectionReadiness(); }); updateSplit();
        collectionStatus_ = new QLabel(tab); collectionStatus_->setWordWrap(true); root->addWidget(collectionStatus_);
        openDatasetTrainer_ = new QPushButton("Apply split and open selected dataset in Trainer", tab); openDatasetTrainer_->setObjectName("openDatasetInTrainer"); root->addWidget(openDatasetTrainer_);
        connect(openDatasetTrainer_, &QPushButton::clicked, this, [this] { applySplitAndOpenTrainer(); });
        connect(validationFraction_, &QDoubleSpinBox::valueChanged, this, [this] { queueAutomaticSplit(false); updateCollectionReadiness(); });
        connect(splitSeed_, &QSpinBox::valueChanged, this, [this] { queueAutomaticSplit(false); });
        compose_ = new QPushButton("Create training set and continue", tab); root->addWidget(compose_);
        connect(collectionName_, &QLineEdit::textChanged, this, [this] { updateCollectionReadiness(); });
        updateCollectionReadiness();
        connect(compose_, &QPushButton::clicked, this, [this] {
            updateCollectionReadiness(); if (!compose_->isEnabled()) return;
            const QString name = collectionName_->text().trimmed(); if (!validName(name)) { QMessageBox::information(this, "Training set name", "Use a folder name without separators."); return; }
            QStringList inputs, validation;
            for (int i=0; i<collectionSources_->topLevelItemCount(); ++i) { auto *item = collectionSources_->topLevelItem(i); if (collectionRole(item) == "train") inputs << item->data(0, Qt::UserRole).toString(); if (collectionRole(item) == "validation") validation << item->data(0, Qt::UserRole).toString(); }
            if (inputs.isEmpty()) { QMessageBox::information(this, "Select datasets", "Select one or more dataset rows and click Use for training."); return; }
            const QString mode = splitMode_->currentData().toString();
            if (mode == "explicit" && validation.isEmpty()) { QMessageBox::information(this, "Select validation", "Select a dedicated validation dataset and click Use for validation."); return; }
            QStringList args{"compose-datasets"}; args << inputs << "--output-dir" << QDir(workspace_->text()).filePath("datasets/" + name) << "--split-mode" << mode << "--validation-fraction" << QString::number(validationFraction_->value() / 100.0) << "--seed" << QString::number(splitSeed_->value()) << "--grouping" << groupingPolicy_->currentData().toString();
            if (mode == "explicit") for (const QString &path : validation) args << "--validation-dataset" << path;
            if (mode == "equal-per-dataset" && validationCount_->value()) args << "--validation-count-per-dataset" << QString::number(validationCount_->value());
            args << "--workers" << QString::number(workerCount()); continueToTrainer_ = true; refreshAfter_ = true; startJob("Create training set", args);
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
            args << "--workers" << QString::number(workerCount()); refreshAfter_ = true; startJob("Import Hugging Face dataset", args);
        });
        root->addStretch(); hfRoot->addWidget(hf); hfRoot->addStretch(); tabs_->addTab(scroll, "Training split"); tabs_->addTab(hfScroll, "Import datasets");
    }

    void buildTrainingTab() {
        auto *tab = new QWidget; auto *root = new QVBoxLayout(tab);
        auto *scroll = new QScrollArea; scroll->setWidgetResizable(true); scroll->setFrameShape(QFrame::NoFrame); scroll->setWidget(tab);
        auto *instruction = new QLabel("Select an existing dataset above and choose the student model, then Start model training. The experimental display student uses native stereo inputs and learns full, unregistered display targets. Stock RAFT is a separate native-left experiment. Preparing another training set is optional.", tab);
        instruction->setWordWrap(true); root->addWidget(instruction);
        trainingDataset_ = new QLabel("Select a dataset from the library above.", tab); trainingDataset_->setObjectName("trainingDataset");
        trainingDataset_->setWordWrap(true); trainingDataset_->setTextFormat(Qt::PlainText); root->addWidget(trainingDataset_);
        trainingStatus_ = new QLabel("Ready to start model training.", tab); trainingStatus_->setObjectName("trainingStatus");
        trainingStatus_->setWordWrap(true); trainingStatus_->setTextFormat(Qt::PlainText); trainingStatus_->setTextInteractionFlags(Qt::TextSelectableByMouse); root->addWidget(trainingStatus_);
        auto *form = new QFormLayout;
        form->setFieldGrowthPolicy(QFormLayout::AllNonFixedFieldsGrow);
        runName_ = new QLineEdit("run-" + QDateTime::currentDateTime().toString("yyyyMMdd-HHmmss"), tab); form->addRow("New run", runName_);
        student_ = new QComboBox(tab); student_->setObjectName("trainingStudent");
        student_->addItem("Experimental stereo → display depth", "display");
        student_->addItem("Stock RAFT native-left disparity", "raft");
        student_->setToolTip("Display: native left/right inputs, full unregistered display target and learned display-grid output. Stock RAFT: calibrated native-left flow/disparity task. Their targets and error units differ.");
        form->addRow("Student model", student_);
        form->addRow("RAFT source", pathRow(raftRoot_, projectSettings_ ? projectSettings_->value("raft/root", "/opt/ipde/RAFT-Stereo").toString() : settings_.value("raft_root", "/opt/ipde/RAFT-Stereo").toString(), true, "Choose RAFT-Stereo source", tab));
        form->addRow("Initial RAFT model", pathRow(raftModel_, projectSettings_ ? projectSettings_->value("raft/model", "/opt/ipde/models/raftstereo-middlebury.pth").toString() : settings_.value("raft_model", "/opt/ipde/models/raftstereo-middlebury.pth").toString(), false, "Choose original RAFT checkpoint", tab));
        connect(raftRoot_, &QLineEdit::editingFinished, this, [this] { saveSharedModelSettings(); }); connect(raftModel_, &QLineEdit::editingFinished, this, [this] { saveSharedModelSettings(); });
        root->addLayout(form);
        trainingSimple_ = new QWidget(tab); auto *simple = new QFormLayout(trainingSimple_);
        quality_ = new QSlider(Qt::Horizontal, tab); quality_->setObjectName("trainingQuality"); quality_->setRange(0, 2); quality_->setValue(1); quality_->setTickPosition(QSlider::TicksBelow); quality_->setTickInterval(1);
        length_ = new QSlider(Qt::Horizontal, tab); length_->setObjectName("trainingLength"); length_->setRange(0, 2); length_->setValue(1); length_->setTickPosition(QSlider::TicksBelow); length_->setTickInterval(1);
        simple->addRow("Quality · Low → High", quality_); simple->addRow("Length · Fast → Slow", length_);
        quality_->setToolTip("Larger native pixel crops and more RAFT refinement passes. This costs GPU memory and time; it cannot add detail missing from the teacher.");
        length_->setToolTip("Dataset-aware training duration and validation error goal. The safety cap limits overfitting; inspect independent scenes before using a model.");
        root->addWidget(trainingSimple_);
        trainingAdvanced_ = new QGroupBox("Advanced training settings", tab); auto *trainingAdvancedLayout = new QFormLayout(trainingAdvanced_);
        limitMode_ = new QComboBox(tab); limitMode_->setObjectName("trainingLimitMode"); limitMode_->addItem("Number of training epochs", "epochs"); limitMode_->addItem("Total number of training steps", "steps");
        epochs_ = spin(tab, 1, 10000, 10); epochs_->setObjectName("trainingEpochs");
        steps_ = spin(tab, 1, 100000000, 1000); steps_->setObjectName("trainingTotalSteps");
        stepsPerUpdate_ = spin(tab, 1, 128, 1); stepsPerUpdate_->setObjectName("trainingStepsPerUpdate");
        patch_ = spin(tab, 64, 2048, 512); patch_->setSingleStep(32);
        iterations_ = spin(tab, 1, 256, 16); scope_ = new QComboBox(tab); scope_->addItem("Update block", "update"); scope_->addItem("Full network", "full"); trainDevice_ = deviceBox(tab);
        epochs_->setToolTip("One epoch visits every eligible training image once in shuffled order, drawing one native-resolution crop per image.");
        steps_->setToolTip("Total Steps counts optimizer updates. Step mode stops at exactly this cumulative update, saving the final checkpoint.");
        stepsPerUpdate_->setToolTip("Accumulate this many image/crop gradients before one optimizer update. The last update of an epoch can contain fewer images. Total Steps changes immediately in epoch mode.");
        patch_->setToolTip("Width and height of each native-resolution crop; a multiple of 32. Cropping preserves pixel scale. Larger crops give spatial context and require more memory.");
        iterations_->setToolTip("RAFT refinement passes per crop and validation prediction. More passes cost computation.");
        scope_->setToolTip("Update block changes the refinement module. Full network changes all weights and needs diverse data and more memory.");
        trainingAdvancedLayout->addRow("Stop training by", limitMode_); trainingAdvancedLayout->addRow("Number of training epochs", epochs_);
        trainingAdvancedLayout->addRow("Total Steps", steps_); trainingAdvancedLayout->addRow("Steps Per Update", stepsPerUpdate_);
        patchLabel_ = new QLabel("Display decoder tile pixels", tab);
        trainingAdvancedLayout->addRow(patchLabel_, patch_); trainingAdvancedLayout->addRow("RAFT iterations", iterations_);
        trainingAdvancedLayout->addRow("Train scope", scope_); trainingAdvancedLayout->addRow("Device", trainDevice_);
        learningRate_ = new QDoubleSpinBox(tab); learningRate_->setDecimals(9); learningRate_->setRange(0.000000001, 0.1); learningRate_->setValue(0.00001);
        learningRate_->setToolTip("AdamW learning rate per optimizer update. Start conservatively; a larger rate can damage pretrained features or destabilize training.");
        trainingAdvancedLayout->addRow("Learning rate", learningRate_);
        trainingMode_ = new QComboBox(tab); trainingMode_->addItem("Automatic from dataset labels", "auto"); trainingMode_->addItem("Learn teacher estimates (distillation)", "distillation"); trainingMode_->addItem("Learn measured references (supervised)", "supervised"); trainingMode_->addItem("Use both eligible label kinds", "mixed");
        trainingAdvancedLayout->addRow("Training labels", trainingMode_);
        trainingLabelHelp_ = new QLabel(tab); trainingLabelHelp_->setWordWrap(true); trainingAdvancedLayout->addRow(trainingLabelHelp_);
        validationSchedule_ = new QComboBox(tab); validationSchedule_->addItem("Every epoch", "epoch"); validationSchedule_->addItem("Only when saving a checkpoint", "checkpoint");
        validationSamples_ = spin(tab, 0, 1000000, 0); validationSamples_->setSpecialValueText("All validation images"); validationSamples_->setToolTip("N selects fresh random validation images on each check. A passing early-stop threshold is confirmed with the full held-out set before stopping.");
        checkpointSchedule_ = new QComboBox(tab); checkpointSchedule_->addItem("Each epoch", "epoch"); checkpointSchedule_->addItem("Every N epochs", "epochs"); checkpointSchedule_->addItem("Every N Total Steps", "steps"); checkpointSchedule_->addItem("Only at the end", "end");
        checkpointEvery_ = spin(tab, 1, 1000000, 1);
        earlyStop_ = binaryButton("Stop when validation error reaches goal", tab);
        earlyStopError_ = new QDoubleSpinBox(tab); earlyStopError_->setDecimals(4); earlyStopError_->setRange(0.0001, 1000); earlyStopError_->setValue(1.0); earlyStopError_->setSuffix(" px MAE");
        earlyStopError_->setToolTip("Mean absolute horizontal flow error against held-out labels, in native stereo pixels. This is teacher/reference agreement, not a percentage or proof of depth accuracy.");
        maxLoss_ = new QDoubleSpinBox(tab); maxLoss_->setRange(1, 1000000); maxLoss_->setValue(1000); maxLoss_->setSuffix(" px"); maxLoss_->setToolTip("Stop if the weighted mean absolute flow error across refinement iterations reaches this limit, is zero, or is nonfinite. The optimizer's summed sequence loss is kept separately. The checkpoint retains finite weights and resume state.");
        resumePath_ = new QLineEdit(tab); resumePath_->setPlaceholderText("Optional resumable .pth from an earlier run");
        auto *resumeRow = new QWidget(tab); auto *resumeLayout = new QHBoxLayout(resumeRow); resumeLayout->setContentsMargins(0,0,0,0); resumeLayout->addWidget(resumePath_, 1);
        auto *resumeBrowse = new QPushButton("Choose…", tab); resumeLayout->addWidget(resumeBrowse);
        connect(resumeBrowse, &QPushButton::clicked, this, [this] { const QString path = QFileDialog::getOpenFileName(this, "Resume model training", workspace_->text(), "Checkpoints (*.pth *.pt)"); if (!path.isEmpty()) resumePath_->setText(path); });
        trainingAdvancedLayout->addRow("Validation timing", validationSchedule_); trainingAdvancedLayout->addRow("Validation images per check", validationSamples_);
        trainingAdvancedLayout->addRow("Save intermediate checkpoints", checkpointSchedule_); trainingAdvancedLayout->addRow("Checkpoint interval N", checkpointEvery_);
        trainingAdvancedLayout->addRow(earlyStop_); trainingAdvancedLayout->addRow("Validation error goal", earlyStopError_); trainingAdvancedLayout->addRow("Maximum safe training error", maxLoss_); trainingAdvancedLayout->addRow("Resume checkpoint", resumeRow);
        root->addWidget(trainingAdvanced_); trainingAdvanced_->setVisible(advanced_->isChecked()); trainingSimple_->setVisible(!advanced_->isChecked());
        trainingHelp_ = new QLabel(tab); trainingHelp_->setWordWrap(true); root->addWidget(trainingHelp_);
        train_ = new QPushButton("Start model training", tab); root->addWidget(train_);
        saveCheckpoint_ = new QPushButton("Save checkpoint now", tab); saveCheckpoint_->setObjectName("saveCheckpointNow"); saveCheckpoint_->setEnabled(false); root->addWidget(saveCheckpoint_);
        connect(saveCheckpoint_, &QPushButton::clicked, this, [this] { requestTrainingControl("save"); });
        for (auto *control : {epochs_, steps_, stepsPerUpdate_}) connect(control, qOverload<int>(&QSpinBox::valueChanged), this, [this] { updateTrainingSelection(); });
        connect(limitMode_, &QComboBox::currentIndexChanged, this, [this] { updateTrainingSelection(); });
        connect(student_, &QComboBox::currentIndexChanged, this, [this] { updateStudentControls(true); updateTrainingSelection(); });
        connect(trainingMode_, &QComboBox::currentIndexChanged, this, [this] { updateTrainingSelection(); });
        connect(checkpointSchedule_, &QComboBox::currentIndexChanged, this, [this] { checkpointEvery_->setEnabled(!busy_ && (checkpointSchedule_->currentData() == "epochs" || checkpointSchedule_->currentData() == "steps")); });
        for (auto *slider : {quality_, length_}) connect(slider, &QSlider::valueChanged, this, [this] { applySimpleTrainingSettings(); updateTrainingSelection(); });
        connect(advanced_, &QPushButton::toggled, this, [this](bool visible) { trainingSimple_->setVisible(!visible); if (!visible) applySimpleTrainingSettings(); updateTrainingSelection(); });
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
        updateStudentControls(true);
        tabs_->addTab(scroll, "4. Train model & compare");
    }

    bool displayStudent() const { return !student_ || student_->currentData() == "display"; }
    QJsonObject trainingEligibility(const QJsonObject &record) const {
        const QString base = displayStudent() ? "training_eligibility" : "raft_training_eligibility";
        const auto modes = record.value(base + "_by_mode").toObject();
        const QString mode = trainingMode_ ? trainingMode_->currentData().toString() : "auto";
        return modes.value(mode).isObject() ? modes.value(mode).toObject() : record.value(base).toObject();
    }
    static QJsonValue relativeDepthError(const QJsonObject &metrics) {
        return metrics.contains("mean_relative_depth_error") ? metrics.value("mean_relative_depth_error") : metrics.value("mean_absolute_fractional_depth_error");
    }
    static QString validationDescription(const QJsonObject &metrics, const QString &units = {}) {
        if (relativeDepthError(metrics).isDouble()) {
            QString result = QString::number(relativeDepthError(metrics).toDouble() * 100, 'f', 2) + "% mean relative depth error";
            if (metrics.value("mean_absolute_depth_error").isDouble()) result += " · MAE " + QString::number(metrics.value("mean_absolute_depth_error").toDouble(), 'g', 5) + " " + metrics.value("units").toString(units);
            return result;
        }
        if (metrics.value("mean_absolute_flow_error_pixels").isDouble()) return QString::number(metrics.value("mean_absolute_flow_error_pixels").toDouble(), 'f', 3) + " px flow MAE";
        return "See metrics";
    }
    void updateStudentControls(bool resetDefaults) {
        if (!patchLabel_ || !trainingHelp_) return;
        const bool display = displayStudent();
        patchLabel_->setText(display ? "Display decoder tile pixels" : "Native stereo patch pixels");
        patch_->setToolTip(display ? "Output-query tile side. Each image uses every supported full-display target pixel; native stereo RGB is encoded whole. Larger tiles reduce dispatch overhead and increase decoder memory, without adding source information." : "Width and height of a native stereo training crop; a multiple of 32. A crop preserves source pixel scale. Larger crops give more context and use more memory.");
        epochs_->setToolTip(display ? "An epoch visits every eligible image/teacher entry once, using every supported display target pixel in bounded tiles." : "An epoch visits every eligible image/teacher entry once, drawing one native-resolution crop per image.");
        stepsPerUpdate_->setToolTip("Accumulate this many image gradients before one optimizer update. The last update of an epoch may contain fewer images. Total Steps updates immediately in epoch mode.");
        iterations_->setToolTip(display ? "RAFT refinement passes for whole-native-pair correspondence context; more passes cost time and memory." : "RAFT refinement passes for each native crop and validation prediction.");
        scope_->setItemText(scope_->findData("update"), display ? "Added encoders + decoder (RAFT frozen)" : "RAFT update block");
        scope_->setToolTip(display ? "Limited scope trains the added encoders and query decoder while freezing RAFT. Full network also adapts RAFT and costs more memory. Full-native encoding memory is not guaranteed by dataset size." : "Update block changes RAFT's refinement module. Full network changes all weights and needs diverse data and more memory.");
        quality_->setToolTip(display ? "Larger output tiles, wider added student features/decoder, and more native RAFT iterations. All display target pixels are still used. Experimental: higher quality is not a guarantee of better geometry or 64 GB fit." : "Larger native pixel crops and more RAFT refinement passes; more memory and context, without repairing bad labels.");
        earlyStopError_->setSuffix(display ? " fraction" : " px MAE");
        earlyStopError_->setToolTip(display ? "Mean abs(prediction-target)/target over valid held-out display pixels. 0.10 is a 10% average relative difference. Subsample success is confirmed on the full validation set; teacher agreement is not physical accuracy." : "Mean absolute horizontal flow error against held-out labels, in native stereo pixels; not a percent or proof of accuracy.");
        maxLoss_->setSuffix(display ? " fraction" : " px");
        maxLoss_->setToolTip(display ? "Stop if mean fractional display error abs(prediction-target)/target reaches this limit, is zero or nonfinite. Saves finite model and resume state." : "Stop if weighted mean absolute flow error across refinement iterations reaches this pixel limit, is zero or nonfinite. Saves finite model and resume state.");
        trainingLabelHelp_->setText(display ? "Distillation learns the full display teacher, including its blur, scale and mistakes. Relative depth or inverse depth can train directly; one run must retain one unit convention. Supervised requires an independently measured display-grid reference. The teacher and optional anchor see full display RGB; the student sees only native left/right RGB. No teacher warp or stereo-teacher fallback is used." : "Distillation learns compatible native-left teacher labels, including blur, scale and mistakes; physical flow labels require meter scale and calibration. Supervised uses measured references aligned with the native left camera. Display-grid teachers cannot directly train stock RAFT. Teacher agreement does not prove physical accuracy.");
        trainingHelp_->setText(display ? "Total Steps = epochs × ceil(training entries ÷ Steps Per Update). Each entry visits all supported full-display target pixels in bounded decoder tiles. Native left/right RGB remains full size; output is predicted directly on the display grid. Stop saves resumable state at a safe update boundary. This student is experimental." : "Total Steps = epochs × ceil(training entries ÷ Steps Per Update). Each entry supplies a native stereo crop and the output stays on the left grid. Stop saves resumable state at a safe update boundary.");
        if (resetDefaults) learningRate_->setValue(display ? .0001 : .00001);
        if (resetDefaults) { earlyStopError_->setValue(display ? .10 : 1.0); maxLoss_->setValue(1000); }
    }

    int trainingImageCount() const {
        auto *item = datasets_->currentItem(); if (!item) return 0;
        const auto record = item->data(0, Qt::UserRole + 1).toJsonObject();
        const auto eligibility = trainingEligibility(record);
        return eligibility.value("train_count").toInt(record.value("train_count").toInt());
    }
    qint64 plannedTrainingSteps() const {
        if (!limitMode_ || !stepsPerUpdate_) return 0;
        if (limitMode_->currentData() == "steps") return steps_->value();
        return qint64(epochs_->value()) * ((qint64(trainingImageCount()) + stepsPerUpdate_->value() - 1) / stepsPerUpdate_->value());
    }
    void applySimpleTrainingSettings() {
        if (!quality_ || advanced_->isChecked() || busy_) return;
        const int count = qMax(1, trainingImageCount()), quality = quality_->value(), length = length_->value();
        QSignalBlocker e(epochs_), t(steps_), u(stepsPerUpdate_), m(limitMode_);
        patch_->setValue(quality == 0 ? 256 : quality == 1 ? 512 : 768);
        iterations_->setValue(quality == 0 ? 8 : quality == 1 ? 16 : 24);
        const int budget = length == 0 ? 1000 : length == 1 ? 5000 : 15000;
        const int cap = length == 0 ? 5 : length == 1 ? 20 : 50;
        epochs_->setValue(qBound(1, (budget + count - 1) / count, cap)); limitMode_->setCurrentIndex(0); stepsPerUpdate_->setValue(1);
        const auto record = datasets_->currentItem() ? datasets_->currentItem()->data(0, Qt::UserRole + 1).toJsonObject() : QJsonObject{};
        const double nativePixels = record.value("max_native_stereo_pixels").toDouble();
        const bool fullScope = count >= 500 && quality >= 1 && (!displayStudent() || (nativePixels > 0 && nativePixels <= 512 * 512));
        scope_->setCurrentIndex(scope_->findData(fullScope ? "full" : "update"));
        checkpointSchedule_->setCurrentIndex(checkpointSchedule_->findData("epochs")); checkpointEvery_->setValue(length == 0 ? 1 : length == 1 ? 2 : 5);
        validationSchedule_->setCurrentIndex(validationSchedule_->findData("checkpoint")); validationSamples_->setValue(count >= 100 ? 16 : 0);
        earlyStop_->setChecked(true); earlyStopError_->setValue(displayStudent() ? (length == 0 ? .20 : length == 1 ? .10 : .05) : (length == 0 ? 2.0 : length == 1 ? 1.0 : 0.5));
        trainingMode_->setCurrentIndex(0); maxLoss_->setValue(1000);
        learningRate_->setValue(displayStudent() ? .0001 : .00001);
    }
    void updateTrainingSelection() {
        if (!trainingDataset_ || !train_) return;
        if (busy_ && job_ == "Train RAFT-Stereo") return;
        applySimpleTrainingSettings();
        auto *item = datasets_->currentItem();
        train_->setEnabled(!datasetMode_ && !busy_ && item && projectOperations_.value("datasets").toString() != "cleanup-dataset");
        const bool epochMode = limitMode_->currentData() == "epochs";
        epochs_->setEnabled(!busy_ && epochMode); steps_->setEnabled(!busy_ && !epochMode);
        const qint64 planned = plannedTrainingSteps();
        if (epochMode) { QSignalBlocker blocker(steps_); steps_->setValue(int(qBound<qint64>(qint64(1), planned, qint64(steps_->maximum())))); }
        steps_->setToolTip(planned > steps_->maximum() ? "This epoch plan exceeds the GUI's 100,000,000-update limit. Reduce epochs or increase Steps Per Update. The exact plan is shown above." : "Total Steps counts optimizer updates. Step mode stops at exactly this cumulative update, saving the final checkpoint.");
        if (!item) { trainingDataset_->setText("Select a dataset from the library above."); return; }
        const auto eligibility = trainingEligibility(item->data(0, Qt::UserRole + 1).toJsonObject());
        if ((eligibility.contains("trainable") && !eligibility.value("trainable").toBool()) || hasUnsavedReviewExclusions(item->data(0, Qt::UserRole).toString())) train_->setEnabled(false);
        if (planned > steps_->maximum()) train_->setEnabled(false);
        QString details = QString("Dataset: %1 · %2 entries · %3 train / validation\nTotal Steps: %4 · %5 images per epoch · %6 image gradients per update.")
            .arg(item->text(0), item->text(1), item->text(2)).arg(plannedTrainingSteps()).arg(trainingImageCount()).arg(stepsPerUpdate_->value());
        details += epochMode ? QString(" Stop after %1 epochs.").arg(epochs_->value()) : " Stop at the exact Total Steps limit.";
        details += QString(displayStudent() ? "\nDisplay tiles: %1 × %1 (all supported target pixels) · %2 RAFT iterations · %3. Validation: %4; %5." : "\nNative crops: %1 × %1 · %2 RAFT iterations · %3. Validation: %4; %5.")
            .arg(patch_->value()).arg(iterations_->value()).arg(scope_->currentText(), validationSchedule_->currentText(), validationSamples_->value() == 0 ? QString("all held-out images") : QString("%1 random held-out images").arg(validationSamples_->value()));
        if (earlyStop_->isChecked()) details += displayStudent() ? QString(" Full-set early-stop goal: %1% mean relative depth error.").arg(earlyStopError_->value() * 100) : QString(" Full-set early-stop goal: %1 px flow MAE.").arg(earlyStopError_->value());
        if (displayStudent() && eligibility.value("units").isString()) {
            const QString units = eligibility.value("units").toString();
            details += "\nTarget/output units: " + units + (units == "meters" ? ". Learned meter estimates; units do not establish physical accuracy." : ". Relative values have no measured distance scale; inverse-depth remains inverse-depth. No automatic meter conversion.");
        }
        if (!eligibility.isEmpty()) details += QString("\n%1 usable targets; %2 unusable targets will be skipped automatically.")
            .arg(eligibility.value("eligible_count").toInt()).arg(eligibility.value("excluded_count").toInt());
        if (eligibility.contains("trainable") && !eligibility.value("trainable").toBool()) details += "\n" + eligibility.value("reason").toString();
        if (planned > steps_->maximum()) details += "\nPlan exceeds the GUI's 100,000,000-update limit; reduce epochs or increase Steps Per Update.";
        trainingDataset_->setText(details); trainingDataset_->setToolTip(item->data(0, Qt::UserRole).toString());
    }
    void requestTrainingControl(const QString &command) {
        if (trainingControlPath_.isEmpty() || process_->state() == QProcess::NotRunning) return;
        QSaveFile file(trainingControlPath_);
        const QByteArray request = QJsonDocument(QJsonObject{{"command", command}, {"request_id", QUuid::createUuid().toString(QUuid::WithoutBraces)}}).toJson();
        if (!file.open(QIODevice::WriteOnly) || file.write(request) != request.size() || !file.commit()) {
            log_->appendPlainText("Could not write training request: " + file.errorString()); return;
        }
        if (command == "stop") { trainingStopping_ = true; cancel_->setEnabled(false); saveCheckpoint_->setEnabled(false); }
        trainingStatus_->setText(command == "stop" ? "Stop requested. Finishing the current update and saving a resumable checkpoint…" : "Checkpoint requested. Saving after the current update…");
    }

    void updateTrainingProgress(const QJsonObject &event) {
        const QString stage = event.value("stage").toString();
        const int epoch = event.value("epoch").toInt(), epochs = event.value("epochs").toInt();
        const int step = event.value("step").toInt(), steps = event.value("steps_per_epoch").toInt();
        const int completed = event.value("completed_steps").toInt(), total = event.value("total_steps").toInt();
        const int processed = event.value("processed").toInt(), samples = event.value("total").toInt();
        QString message;
        if (stage == "checking_dataset" || stage == "dataset_preflight") message = "Checking dataset metadata; native dataset arrays are verified when used.";
        else if (stage == "filtering_targets") message = QString("Selecting usable targets; %1 invalid targets skipped. Dataset files remain unchanged.").arg(event.value("excluded_count").toInt());
        else if (stage == "skipped_sample") message = "Skipping unusable target: " + QFileInfo(event.value("source_path").toString(event.value("sample_id").toString())).fileName() + " · " + event.value("reason").toString();
        else if (stage == "preparing_targets") message = QString("Preparing %1 targets · %2 / %3").arg(event.value("role").toString()).arg(processed).arg(samples);
        else if (stage == "model_setup") message = displayStudent() ? "Loading the experimental display student model and native RAFT context." : "Loading the RAFT model and preparing the training device.";
        else if (stage == "baseline_validation") message = QString("Checking baseline validation · %1 / %2 photos").arg(processed).arg(samples);
        else if (stage == "epoch_step") message = QString("Epoch %1 / %2 · step %3 / %4").arg(epoch).arg(epochs).arg(step).arg(steps);
        else if (stage == "display_tile") message = event.value("validation").toBool()
            ? QString("Validation · display tile %1 / %2").arg(event.value("tile").toInt()).arg(event.value("tiles").toInt())
            : QString("Epoch %1 / %2 · image step %3 / %4 · display tile %5 / %6").arg(epoch).arg(epochs).arg(step + 1).arg(steps).arg(event.value("tile").toInt()).arg(event.value("tiles").toInt());
        else if (stage == "epoch_validation") message = QString("Epoch %1 / %2 · validation %3 / %4 photos").arg(epoch).arg(epochs).arg(processed).arg(samples);
        else if (stage == "checkpoint_validation") message = QString("Checkpoint validation · %1 / %2 photos").arg(processed).arg(samples);
        else if (stage == "resumed") message = QString("Resumed training at epoch %1 · Total Step %2").arg(epoch).arg(completed);
        else if (stage == "final_validation") message = QString("Final validation of the selected checkpoint · %1 / %2 photos").arg(processed).arg(samples);
        else if (stage == "writing_checkpoint") message = "Saving and verifying a resumable model checkpoint.";
        else if (stage == "checkpoint_saved") message = "Checkpoint ready: " + event.value("checkpoint_path").toString();
        else if (stage == "early_stop_confirmation") message = "Confirming the error goal against the full validation set.";
        else if (stage == "stopped" || stage == "training_stopped") message = "Training stopped: " + event.value("stop_reason").toString();
        else if (stage == "completed") message = "Model checkpoint saved. Finishing the training run.";
        if (message.isEmpty()) return;
        if (stage == "checkpoint_saved") {
            const QString path = event.value("checkpoint_path").toString();
            bool listed = false;
            for (int i = 0; i < runs_->topLevelItemCount(); ++i) listed |= runs_->topLevelItem(i)->data(0, Qt::UserRole).toString() == path;
            if (!path.isEmpty() && !listed) { auto *item = new QTreeWidgetItem(runs_, {QFileInfo(path).fileName(), QString::number(epoch), "Saved during training"}); item->setData(0, Qt::UserRole, path); item->setToolTip(0, path); }
        }
        if (event.value("sample_patches").toInt() > 0) message += QString(" · crop %1 / %2").arg(event.value("sample_patch").toInt()).arg(event.value("sample_patches").toInt());
        message += QString(" · Total Steps %1 / %2").arg(completed).arg(total);
        if (event.value("loss").isDouble()) message += " · loss " + QString::number(event.value("loss").toDouble(), 'g', 6);
        if (event.value("guard_error_pixels").isDouble()) message += " · training error " + QString::number(event.value("guard_error_pixels").toDouble(), 'f', 2) + " px";
        if (event.value("mean_absolute_flow_error_pixels").isDouble()) message += " · held-out error " + QString::number(event.value("mean_absolute_flow_error_pixels").toDouble(), 'f', 3) + " px";
        if (relativeDepthError(event).isDouble()) message += " · mean relative depth error " + QString::number(relativeDepthError(event).toDouble() * 100, 'f', 2) + "%";
        if (event.value("mean_absolute_depth_error").isDouble()) message += " · depth MAE " + QString::number(event.value("mean_absolute_depth_error").toDouble(), 'g', 5) + " " + event.value("units").toString();
        trainingStatus_->setText(message); statusBar()->showMessage(message);
        progress_->setTextVisible(true); progress_->setFormat("%v / %m Total Steps");
        if (total > 0) { progress_->setRange(0, total); progress_->setValue(qBound(0, completed, total)); }
        if (stage != "epoch_step" && stage != "display_tile" && stage != "preparing_targets" && stage != "skipped_sample"
            && (event.value("status").toString() == "finished" || stage == "model_setup" || stage == "writing_checkpoint" || stage == "completed")) log_->appendPlainText(message);
    }

    bool validName(const QString &name) {
        return !name.isEmpty() && name != "." && name != ".." && !name.contains('/') && !name.contains('\\');
    }

    void generateDataset() {
        if (!datasetMode_) { openApp("datasets", "import"); return; }
        if (sources_->topLevelItemCount() == 0) { QMessageBox::information(this, "Add photos", "Add spatial HEIC photos before generating a dataset."); return; }
        const QString name = datasetName_->text().trimmed();
        if (!validName(name)) { QMessageBox::information(this, "Dataset name", "Use a folder name without path separators."); return; }
        if (inputSize_->value() > 0 && inputSize_->value() < 14) { QMessageBox::information(this, "Teacher input size", "Use 0 for native source dimensions, or a custom input size of at least 14 pixels."); return; }
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
             << "--teacher-view" << "display"
             << "--grouping" << (useGroups_->isChecked() ? (verifiedScenes_->isChecked() ? "scene" : "capture") : "none");
        if (anchor_->isChecked()) args << "--metric-anchor" << "depthpro";
        args << "--workers" << QString::number(workerCount()); refreshAfter_ = true; startJob("Generate dataset", args);
    }

    void trainDataset() {
        if (datasetMode_) { openApp("trainer", "train", selectedPath(datasets_)); return; }
        if (projectOperations_.value("datasets").toString() == "cleanup-dataset") { statusBar()->showMessage("Finish dataset cleanup in Dataset Studio before training."); return; }
        const QString dataset = selectedPath(datasets_); const QString name = runName_->text().trimmed();
        if (dataset.isEmpty()) { QMessageBox::information(this, "Choose dataset", "Select a dataset from the library above."); return; }
        if (hasUnsavedReviewExclusions(dataset)) { statusBar()->showMessage("Finish saving the pending dataset edits before training."); return; }
        const auto eligibility = trainingEligibility(datasets_->currentItem()->data(0, Qt::UserRole + 1).toJsonObject());
        if (eligibility.contains("trainable") && !eligibility.value("trainable").toBool()) { statusBar()->showMessage(eligibility.value("reason").toString()); return; }
        if (!validName(name)) { QMessageBox::information(this, "Run name", "Use a run folder name without path separators."); return; }
        if (!displayStudent() && patch_->value() % 32) { QMessageBox::information(this, "Patch size", "Stock RAFT patch size must be a multiple of 32 (for example 256 or 512)."); return; }
        const QString checkpoint = QDir(workspace_->text()).filePath("runs/" + name + "/checkpoint.pth");
        saveSharedModelSettings();
        refreshAfter_ = true;
        QStringList args{"train", dataset, "--checkpoint", checkpoint, "--raft-root", raftRoot_->text(), "--raft-model", raftModel_->text(),
            "--student", student_->currentData().toString(),
            "--limit-mode", limitMode_->currentData().toString(), "--epochs", QString::number(epochs_->value()), "--total-steps", QString::number(steps_->value()),
            "--steps-per-update", QString::number(stepsPerUpdate_->value()), "--patch-size", QString::number(patch_->value()),
            "--validation-schedule", validationSchedule_->currentData().toString(), "--validation-samples", QString::number(validationSamples_->value()),
            "--checkpoint-schedule", checkpointSchedule_->currentData().toString(), "--checkpoint-every", QString::number(checkpointEvery_->value()), "--max-loss", QString::number(maxLoss_->value()),
            "--learning-rate", QString::number(learningRate_->value(), 'g', 10),
            "--iterations", QString::number(iterations_->value()), "--scope", scope_->currentData().toString(), "--device", trainDevice_->currentText(), "--mode", trainingMode_->currentData().toString(), "--workers", QString::number(workerCount())};
        const QString member = projectSettings_ ? projectSettings_->value("raft/member").toString() : QString(); if (!member.isEmpty()) args << "--raft-model-member" << member;
        trainingControlPath_ = QFileInfo(checkpoint).absolutePath() + "/training-control.json";
        QDir().mkpath(QFileInfo(checkpoint).absolutePath());
        if (QFileInfo::exists(trainingControlPath_)) { QMessageBox::information(this, "Run already exists", "Choose a new run name; this run already has training control state."); return; }
        args << "--control-file" << trainingControlPath_;
        if (earlyStop_->isChecked()) args << "--early-stop-error" << QString::number(earlyStopError_->value());
        if (!resumePath_->text().trimmed().isEmpty()) args << "--resume" << resumePath_->text().trimmed();
        trainingStopping_ = false;
        startJob("Train RAFT-Stereo", args);
    }

    void updateSessionBusy() {
        if (session_) session_->setBusy(busy_ && job_ != "Refresh library" ? activeOperation_ : exportRunning() ? QString("export") : QString());
    }

    void configureExportProcess() {
        if (exportProcess_) return;
        exportProcess_ = new QProcess(this);
        auto finished = [this] {
            updateSessionBusy();
            export_->setEnabled(!busy_ || job_ == "Train RAFT-Stereo");
            if (closeAfterTraining_ && !(busy_ && job_ == "Train RAFT-Stereo")) {
                closeAfterTraining_ = false; QTimer::singleShot(0, this, &QWidget::close);
            }
        };
        connect(exportProcess_, qOverload<int, QProcess::ExitStatus>(&QProcess::finished), this, [this, finished](int code, QProcess::ExitStatus status) {
            const auto result = QJsonDocument::fromJson(exportProcess_->readAllStandardOutput()).object();
            log_->appendPlainText(status == QProcess::NormalExit && code == 0 ? "Checkpoint export ready: " + exportedCheckpointPath(result) : "Checkpoint export failed: " + result.value("error").toString(QString::fromUtf8(exportProcess_->readAllStandardError())));
            finished();
        });
        connect(exportProcess_, &QProcess::errorOccurred, this, [this, finished](QProcess::ProcessError error) {
            if (error == QProcess::FailedToStart) { log_->appendPlainText("Could not start checkpoint export: " + exportProcess_->errorString()); finished(); }
        });
    }

    QString exportedCheckpointPath(const QJsonObject &result) const {
        const QString path = result.value("checkpoint_path").toString();
        if (!path.isEmpty()) return path;
        return QDir(exportDestination_).filePath(result.value("checkpoint").toString(result.value("schema").toString().contains("display") ? "display-model.pth" : "raft-model.pth"));
    }

    void exportModel() {
        const QString checkpoint = selectedPath(runs_);
        if (checkpoint.isEmpty()) { QMessageBox::information(this, "Choose model", "Select a trained model from the library above."); return; }
        const QString parent = QFileDialog::getExistingDirectory(this, "Choose export destination", workspace_->text());
        if (parent.isEmpty()) return;
        exportDestination_ = QDir(parent).filePath("raft-export-" + QDateTime::currentDateTime().toString("yyyyMMdd-HHmmss"));
        const QStringList arguments{"export", checkpoint, "--output", exportDestination_, "--raft-root", raftRoot_->text()};
        if (busy_ && job_ == "Train RAFT-Stereo") {
            if (exportRunning()) return;
            configureExportProcess();
            exportProcess_->setProgram(pythonPath()); exportProcess_->setArguments(QStringList{scriptPath(), "--json"} + arguments); exportProcess_->start();
            updateSessionBusy();
            export_->setEnabled(false); log_->appendPlainText("Exporting the saved checkpoint while training continues…");
            return;
        }
        startJob("Export RAFT model", arguments);
    }

    void refreshLibrary() {
        if (process_ && process_->state() != QProcess::NotRunning) return;
        if (!draftWorkspace_.isEmpty() && QDir(workspace_->text()).absolutePath() != draftWorkspace_ && hasPendingDatasetEdits()) {
            workspace_->setText(draftWorkspace_); statusBar()->showMessage("Finish saving pending dataset edits before changing the workspace."); return;
        }
        loadReviewDrafts();
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
        if (!datasetMode_) { openApp("datasets", "prepare", selectedPath(datasets_)); return; }
        if (process_->state() != QProcess::NotRunning) return;
        const QString source = selectedPath(datasets_);
        if (source.isEmpty()) return;
        if (!ownsDataset(source)) {
            log_->appendPlainText("Only datasets directly inside this workspace can be archived."); return;
        }
        refreshAfter_ = true; startJob("Archive dataset", {"archive-dataset", source, "--workspace", workspace_->text()});
    }

    void clearUnavailableReview(const QString &path, const QString &message) {
        if (path.isEmpty() || path != reviewedDataset_) return;
        pendingPreviewArgs_.clear(); previewKind_ = "discarded"; previewProcess_->kill();
        reviewedDataset_.clear(); requestedReviewPath_.clear(); reviewPreviews_.reset(); reviewGenerating_ = false;
        { QSignalBlocker blocker(reviewSamples_); reviewSamples_->clear(); }
        { QSignalBlocker blocker(reviewLabel_); reviewLabel_->clear(); }
        rgbPreview_->reset(message); depthPreview_->reset("Select another dataset to review.");
        previewStats_->clear(); reviewPath_->setText(message); updateReviewCount();
    }

    void startJob(const QString &label, const QStringList &arguments) {
        const QString operation = arguments.value(0);
        const bool mutatesDataset = QStringList{"dataset", "compose-datasets", "compact-dataset", "curate-dataset", "edit-dataset", "import-hf", "cleanup-dataset", "archive-dataset"}.contains(operation);
        if ((!datasetMode_ && mutatesDataset) || (datasetMode_ && QStringList{"train", "export", "cleanup-run"}.contains(operation))) {
            openApp(mutatesDataset ? "datasets" : "trainer", mutatesDataset ? "prepare" : "train", selectedPath(datasets_));
            groupsFile_.reset(); teachersFile_.reset(); refreshAfter_ = false; return;
        }
        if (operation == "cleanup-dataset" && !projectOperations_.value("trainer").toString().isEmpty()) {
            statusBar()->showMessage("Finish or cancel model training in Trainer before removing a dataset."); refreshAfter_ = false; return;
        }
        if (process_->state() != QProcess::NotRunning) { statusBar()->showMessage("Finish or cancel the current background task first; photo review remains available."); return; }
        if (!QFileInfo::exists(scriptPath())) { log_->appendPlainText("RAFT Studio script is missing: " + scriptPath()); groupsFile_.reset(); teachersFile_.reset(); refreshAfter_ = false; statusBar()->showMessage("Could not start " + label + "; see the progress log."); return; }
        job_ = label; stdout_.clear(); progressBuffer_.clear(); cancelled_ = false;
        activeOperation_ = operation;
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
                if (type == "dataset_started" && appendBase_.isEmpty()) streamingDataset_ = event.value("dataset_dir").toString();
                else if (type == "sample_ready" && appendBase_.isEmpty()) { streamingDataset_ = event.value("dataset_dir").toString(streamingDataset_); if (!streamingTimer_->isActive()) streamingTimer_->start(); }
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
                else if (job_ == "Train RAFT-Stereo" && phase == "training_progress") updateTrainingProgress(event);
                else if (!phase.isEmpty()) {
                    QString message;
                    const int processed = event.value("processed").toInt(), total = event.value("total").toInt();
                    if (phase == "verifying_dataset") message = QString("Checking source dataset %1 of %2: %3. Large datasets can take several minutes.").arg(processed).arg(total).arg(QFileInfo(event.value("dataset_path").toString()).fileName());
                    else if (phase == "sample_composed") message = QString("Preparing training set: %1 of %2 entries · %3 arrays %4.").arg(processed).arg(total).arg(event.value("unique_arrays").toInt()).arg(event.value("storage_mode").toString() == "copy" ? "copied losslessly" : "reused without a full data copy");
                    else if (phase == "editing_dataset") message = QString("Saving dataset version: %1 of %2 arrays preserved.").arg(processed).arg(total);
                    else if (phase == "compressing") message = QString("Compressing arrays losslessly: %1 of %2 records (%3 unique arrays).").arg(processed).arg(total).arg(event.value("unique_arrays").toInt());
                    else if (phase == "verifying_output") message = "Checking the saved arrays before publishing the completed training set. Large datasets can take several minutes.";
                    else if (phase == "publishing_dataset") message = "Publishing the verified training set…";
                    if (!message.isEmpty()) {
                        statusBar()->showMessage(message);
                        if (job_ == "Create training set") collectionStatus_->setText(message + " Use Cancel current task to stop.");
                        if ((phase == "sample_composed" || phase == "compressing" || phase == "editing_dataset") && total > 0) { progress_->setRange(0, total); progress_->setValue(processed); }
                        else progress_->setRange(0, 0);
                    }
                }
            } else if (!line.trimmed().isEmpty()) log_->appendPlainText(line.trimmed());
        }
    }

    void setBusy(bool busy) {
        busy_ = busy;
        updateSessionBusy();
        workspace_->setEnabled(!busy); chooseWorkspace_->setEnabled(!busy); refresh_->setEnabled(!busy); export_->setEnabled((!busy || job_ == "Train RAFT-Stereo") && (!exportProcess_ || exportProcess_->state() == QProcess::NotRunning));
        generate_->setEnabled(!busy); train_->setEnabled(!busy); importHf_->setEnabled(!busy); cancelAdding_->setEnabled(!busy);
        for (auto *button : editActions_) button->setEnabled(!busy && !reviewGenerating_ && !reviewEntries().isEmpty());
        reviewSamples_->setEnabled(!busy || job_ == "Generate dataset"); reviewedName_->setEnabled(!busy);
        train_->setText(busy && job_ == "Train RAFT-Stereo" ? "Training model…" : "Start model training");
        for (auto *control : QList<QWidget *>{runName_, student_, raftRoot_, raftModel_, epochs_, steps_, patch_, iterations_, scope_, trainDevice_, trainingMode_, stepsPerUpdate_, limitMode_, validationSchedule_, validationSamples_, checkpointSchedule_, checkpointEvery_, earlyStop_, earlyStopError_, maxLoss_, learningRate_, resumePath_, quality_, length_, advanced_}) control->setEnabled(!busy);
        collectionSources_->setEnabled(!busy); collectionName_->setEnabled(!busy); splitMode_->setEnabled(!busy); groupingPolicy_->setEnabled(!busy); splitSeed_->setEnabled(!busy);
        validationFraction_->setEnabled(!busy && splitMode_->currentData().toString() != "explicit"); validationCount_->setEnabled(!busy && splitMode_->currentData().toString() == "equal-per-dataset");
        updateCollectionReadiness();
        updateReviewCount();
        cancel_->setEnabled(busy && !trainingStopping_);
        cancel_->setText(busy && job_ == "Train RAFT-Stereo" ? "Stop and save training" : "Cancel current task");
        saveCheckpoint_->setEnabled(busy && job_ == "Train RAFT-Stereo" && !trainingStopping_);
        if (!busy) { trainingStopping_ = false; trainingControlPath_.clear(); }

        if (job_ == "Train RAFT-Stereo") {
            if (busy) {
                trainingStatus_->setText("Starting model training. Checking the selected dataset before the first epoch.");
                progress_->setRange(0, int(qBound<qint64>(qint64(1), plannedTrainingSteps(), qint64(steps_->maximum())))); progress_->setValue(0);
            }
            progress_->setTextVisible(true); progress_->setFormat("%v / %m Total Steps");
        } else { progress_->setTextVisible(false); progress_->setRange(0, busy ? 0 : 1); progress_->setValue(0); }
        updateTrainingSelection();
        updateCleanupActions();
        statusBar()->showMessage(busy ? job_ : "Ready");
        if (!busy && autosaveTimer_ && !autosavePaused_ && hasPendingDatasetEdits()) autosaveTimer_->start();
    }

    bool hasUnsavedReviewExclusions(const QString &path) const {
        if (path.isEmpty()) return false;
        if (reviewDrafts_.value(path).value("dirty").toBool() || editingDataset_ == path) return true;
        if (!reviewAdditions_.value(path).isEmpty()) return true;
        if (path == reviewedDataset_ && !reviewEntries().isEmpty()) {
            for (auto *item : reviewEntries()) {
                const auto sample = item->data(0, Qt::UserRole).toJsonObject();
                if (reviewIncluded(item) != sample.value("original_included").toBool(true) || (sample.contains("original_split") && (sample.value("split_chosen").toBool() || sample.value("split") != sample.value("original_split")))) return true;
            }
        } else if (reviewDrafts_.contains(path)) {
            const auto draft = reviewDrafts_.value(path);
            if (draft.value("dirty").toBool()) return true;
        }
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
            if (collectionRole(item) == "train") { ++training; entries += item->text(3).toInt(); }
            if (collectionRole(item) == "validation") ++validation;
            if (collectionRole(item) == "train" || (splitMode_->currentData().toString() == "explicit" && collectionRole(item) == "validation"))
                unsavedExclusions |= hasUnsavedReviewExclusions(item->data(0, Qt::UserRole).toString());
        }
        QString reason;
        if (!datasetMode_) reason = "Open Dataset Studio to prepare or manage training datasets.";
        else if (busy_) reason = job_ == "Create training set" ? "Preparing the training set using existing array storage. Checking full-quality arrays may take several minutes. Use Cancel current task to stop." : job_ + " is running. Finish or cancel it before creating a training set.";
        else if (!collectionSources_->topLevelItemCount()) reason = "Import or generate a dataset first; it will appear here and can be used for training.";
        else if (!training) reason = "Select a dataset row and click Use for training. One dataset is enough; validation photos are held out automatically.";
        else if (!entries) reason = "The selected training datasets have no teacher entries. Finish generating or import a complete dataset first.";
        else if (unsavedExclusions) reason = "Photo or split edits are saving. If a save failed, use Save changes in Photos and depth to retry before creating a combined set.";
        else if (!validName(collectionName_->text().trimmed())) reason = "Enter a training set name without folder separators.";
        else if (QFileInfo::exists(QDir(workspace_->text()).filePath("datasets/" + collectionName_->text().trimmed()))) reason = "A dataset with this name already exists. Enter a new training set name.";
        else if (splitMode_->currentData().toString() == "explicit" && !validation) reason = "Select a separate dataset and click Use for validation, or choose random validation to hold out photos from your training dataset.";
        if (reason.isEmpty() && splitMode_->currentData().toString() != "explicit" && validation) reason = "A dataset is assigned as validation. Choose designated validation datasets, or change that dataset to Training or Not used.";
        const bool canAssign = datasetMode_ && !busy_ && !collectionSources_->selectedItems().isEmpty();
        for (auto *button : {useDatasetForTraining_, useDatasetForValidation_, skipDataset_}) if (button) button->setEnabled(canAssign);
        if (openDatasetTrainer_) openDatasetTrainer_->setEnabled(datasetMode_ && !busy_ && !selectedPath(datasets_).isEmpty());
        compose_->setEnabled(reason.isEmpty());
        compose_->setText(busy_ && job_ == "Create training set" ? "Creating training set…" : "Create training set and continue");
        const QString ready = QString("Ready: %1 training dataset(s), %2 teacher entries. %3 The set needs at least two independent photo groups.").arg(training).arg(entries).arg(splitMode_->currentData().toString() == "explicit" ? QString("%1 validation dataset(s) will be held out.").arg(validation) : "Validation photos will be held out automatically.");
        collectionStatus_->setText(reason.isEmpty() ? ready : reason);
        compose_->setToolTip(reason.isEmpty() ? "Prepare validation splits using shared array storage, then open the model training step." : reason);
    }

    void processFinished(int code, QProcess::ExitStatus status) {
        stdout_ += process_->readAllStandardOutput(); appendProgress(process_->readAllStandardError());
        if (!progressBuffer_.trimmed().isEmpty()) { log_->appendPlainText(QString::fromUtf8(progressBuffer_).trimmed()); progressBuffer_.clear(); }
        setBusy(false); groupsFile_.reset(); teachersFile_.reset(); editsFile_.reset();
        if (cancelled_) {
            if (job_ == "Create training set") continueToTrainer_ = false;
            if (job_ == "Generate dataset") stopGenerationStreaming("Generation cancelled. Generate a new dataset before saving a reviewed copy.");
            if (job_ == "Train RAFT-Stereo") trainingStatus_->setText("Model training cancelled. The dataset is unchanged; this run did not finish.");
            streamingTimer_->stop(); streamingDataset_.clear();
            refreshAfter_ = false; statusBar()->showMessage("Cancelled");
            if (job_ == "Scan spatial photos") { log_->appendPlainText("Scan cancelled; photos already found remain in the list."); statusBar()->showMessage("Scan cancelled; photos already found remain in the list."); }
            if (job_ == "Preview depth") { rgbPreview_->reset("Preview cancelled."); depthPreview_->reset("Choose another photo or depth label to retry."); }
            return;
        }
        QJsonParseError error; const QJsonDocument doc = QJsonDocument::fromJson(stdout_.trimmed(), &error);
        if (status != QProcess::NormalExit || code != 0 || !doc.isObject()) {
            if (job_ == "Create training set") continueToTrainer_ = false;
            log_->appendPlainText(job_ + " failed (exit " + QString::number(code) + ").");
            statusBar()->showMessage(job_ + " failed; see the progress log. You can adjust the selection and retry.");
            const QString explanation = doc.isObject() ? doc.object().value("error").toString() : QString();
            if (!explanation.isEmpty()) {
                log_->appendPlainText(explanation);
                if (job_ == "Create training set") collectionStatus_->setText("Could not create the training set: " + explanation + " Adjust the selection or settings and retry.");
            } else if (!stdout_.trimmed().isEmpty()) log_->appendPlainText(QString::fromUtf8(stdout_).left(12000));
            if (job_ == "Train RAFT-Stereo") trainingStatus_->setText("Model training failed: " + (explanation.isEmpty() ? "see the progress log." : explanation) + " Dataset files remain unchanged.");
            if (code == 0 && error.error != QJsonParseError::NoError) log_->appendPlainText("Could not parse the result: " + error.errorString());
            if (job_ == "Preview depth") {
                rgbPreview_->reset("Preview unavailable."); depthPreview_->reset("Preview unavailable.");
                previewStats_->setText("Preview failed. See the error below; this sample has not been automatically excluded.");
            }
            if (job_ == "Generate dataset") stopGenerationStreaming("Generation failed. See the progress log and generate a new dataset to try again.");
            closeAfterTraining_ = false;
            refreshAfter_ = false; return;
        }
        const QJsonObject result = doc.object();
        if (job_ == "Generate dataset" && !appendBase_.isEmpty()) {
            const QString source = appendBase_, addition = result.value("dataset_path").toString(); const QJsonObject edits = appendEdits_;
            appendBase_.clear(); appendEdits_ = {}; appendVersionName_.clear();
            addingPhotosHint_->hide(); cancelAdding_->hide(); generate_->setText("Generate dataset");
            log_->appendPlainText("New photo targets generated. Linking them into the existing dataset…");
            saveDatasetVersion(source, edits, {addition}, {addition}); return;
        }
        if (job_ == "Save dataset version") {
            reviewAdditions_.remove(savingVersionSource_);
            // The saved version contains this draft; returning to the source starts from its original membership.
            reviewDrafts_.remove(savingVersionSource_);
            if (reviewedDataset_ == savingVersionSource_) { QSignalBlocker blocker(reviewSamples_); reviewSamples_->clear(); }
        }
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
            trainingStatus_->setText("Training set prepared. Click Start model training to run epochs and save a model checkpoint.");
            for (const auto &warning : result.value("warnings").toArray()) log_->appendPlainText(warning.toString());
        }
        else if (job_ == "Archive dataset") {
            const QString destination = result.value("archived_dataset").toString();
            clearUnavailableReview(result.value("source_dataset").toString(), "Dataset archived to " + destination);
            log_->appendPlainText("Dataset archived without deleting files: " + destination);
        }
        else if (job_ == "Remove generated dataset") {
            clearUnavailableReview(result.value("cleaned_dataset").toString(), "Generated dataset removed. Original photos and trained models retained.");
            log_->appendPlainText(QString("Removed %1 generated files (%2 MiB logical size). Shared storage may remain in other datasets.").arg(result.value("removed_files").toInt()).arg(result.value("removed_logical_bytes").toDouble() / (1024*1024), 0, 'f', 1));
        }
        else if (job_ == "Generate dataset" || job_ == "Save reviewed dataset" || job_ == "Save dataset version" || job_ == "Import Hugging Face dataset" || job_ == "Compact dataset") {
            pendingReview_ = result.value("dataset_path").toString();
            log_->appendPlainText("Dataset saved: " + pendingReview_ + "\n" + QString::fromUtf8(QJsonDocument(result.value("summary").toObject()).toJson(QJsonDocument::Compact)));
            for (const auto &warning : result.value("warnings").toArray()) log_->appendPlainText(warning.toString());
        }
        else if (job_ == "Train RAFT-Stereo") {
            const auto metrics = result.value("validation").toObject();
            QString summary = QString("Model training finished · %1 epochs · %2 Total Steps · best epoch %3.")
                .arg(result.value("epochs_completed").toInt(result.value("history").toArray().size()))
                .arg(result.value("total_steps").toInt()).arg(result.value("best_epoch").toInt());
            if (relativeDepthError(metrics).isDouble() || metrics.value("mean_absolute_flow_error_pixels").isDouble()) summary += " Held-out error: " + validationDescription(metrics, result.value("units").toString()) + ".";
            if (result.contains("stop_reason")) summary += " Reason: " + result.value("stop_reason").toString() + ".";
            summary += " Checkpoint: " + result.value("checkpoint_path").toString();
            const auto excluded = result.value("excluded_samples").toArray();
            if (!excluded.isEmpty()) summary += QString(" %1 unusable targets skipped; dataset unchanged.").arg(excluded.size());
            trainingStatus_->setText(summary); log_->appendPlainText(summary);
            for (const auto &warning : result.value("warnings").toArray()) log_->appendPlainText(warning.toString());
            if (closeAfterTraining_) { refreshAfter_ = false; if (!exportRunning()) { closeAfterTraining_ = false; QTimer::singleShot(0, this, &QWidget::close); } return; }
        }
        else log_->appendPlainText(QString::fromUtf8(QJsonDocument(result).toJson(QJsonDocument::Indented)).left(18000));
        statusBar()->showMessage(job_ + " complete");
        if (job_ == "Export RAFT model") {
            const QString file = exportedCheckpointPath(result);
            log_->appendPlainText("Export complete: " + file + " — choose this checkpoint in the model control; its architecture determines the available output grid.");
            showFolder(exportDestination_);
        }
        const bool refresh = refreshAfter_; refreshAfter_ = false;
        if (refresh) QTimer::singleShot(0, this, [this] { refreshLibrary(); });
        else if (job_ == "Refresh library" && !pendingReview_.isEmpty()) {
            const QString path = pendingReview_; pendingReview_.clear();
            QTimer::singleShot(0, this, [this, path] { reviewDataset(path); });
        }
        else if (job_ == "Refresh library" && !pendingTrainingPath_.isEmpty()) {
            const QString path = pendingTrainingPath_; pendingTrainingPath_.clear();
            if (datasetMode_ && continueToTrainer_) { continueToTrainer_ = false; openApp("trainer", "train", path); }
            else if (!datasetMode_) tabs_->setCurrentIndex(4);
        }
    }

    void populateLibrary(const QJsonObject &result) {
        const QString oldDataset = !pendingTrainingPath_.isEmpty() ? pendingTrainingPath_ : pendingReview_.isEmpty() ? selectedPath(datasets_) : pendingReview_, oldRun = selectedPath(runs_);
        QMap<QString, QString> collectionRoles;
        for (int i=0; i<collectionSources_->topLevelItemCount(); ++i) { auto *item = collectionSources_->topLevelItem(i); collectionRoles.insert(item->data(0, Qt::UserRole).toString(), collectionRole(item)); }
        QSignalBlocker datasetBlocker(datasets_); datasets_->clear(); runs_->clear(); QSignalBlocker collectionBlocker(collectionSources_); collectionSources_->clear();
        for (const QJsonValue &value : result.value("datasets").toArray()) {
            const auto obj = value.toObject(); const QString path = obj.value("path").toString();
            QString teacher = obj.value("teacher").toString();
            if (teacher.isEmpty() && obj.value("teacher").isObject()) teacher = obj.value("teacher").toObject().value("model").toString();
            auto *item = new QTreeWidgetItem(datasets_, {obj.value("name").toString(QFileInfo(path).fileName()), QString::number(obj.value("sample_count").toInt()),
                QString("%1 / %2").arg(obj.value("train_count").toInt()).arg(obj.value("validation_count").toInt()), teacher, obj.value("category").toString(), QString::number(obj.value("storage_bytes").toDouble() / (1024*1024), 'f', 1) + " MiB"});
            const bool linked = obj.value("linked").toBool();
            const QString location = path + (linked ? "\nLinked from another location. Photo and split edits update this dataset's manifest in its original location." : QString());
            item->setData(0, Qt::UserRole, path); item->setToolTip(0, location);
            item->setData(0, Qt::UserRole + 1, obj);
            if (obj.contains("storage_bytes_known") && !obj.value("storage_bytes_known").toBool()) item->setText(5, "Size not measured");
            const auto storage = obj.value("storage").toObject();
            if (storage.value("storage_mode").toString() == "shared") {
                item->setText(5, QString("Shared arrays · %1 MiB added").arg(storage.value("added_storage_bytes").toDouble() / (1024*1024), 'f', 2));
                item->setToolTip(5, "Reuses existing array storage without another full data copy. " + QString::number(storage.value("reused_array_storage_bytes").toDouble() / (1024*1024), 'f', 1) + " MiB of arrays reused.");
            }
            if (linked) item->setText(0, item->text(0) + " ↗");
            if (path == oldDataset) datasets_->setCurrentItem(item);
            auto *collection = new QTreeWidgetItem(collectionSources_, {obj.value("name").toString(QFileInfo(path).fileName()), "", obj.value("category").toString(), QString::number(obj.value("sample_count").toInt())});
            collection->setData(0, Qt::UserRole, path); collection->setToolTip(0, location);
            disableItemCheckboxes(collection); const QString role = collectionRoles.value(path, "unused");
            collection->setData(0, Qt::UserRole + 2, role); collection->setText(1, role == "train" ? "Training" : role == "validation" ? "Validation" : "Not used");
        }
        for (const QJsonValue &value : result.value("runs").toArray()) {
            const auto obj = value.toObject(); const QString path = obj.value("path").toString();
            const QJsonObject metrics = obj.value("validation").toObject();
            QString validation = validationDescription(metrics, obj.value("units").toString());
            auto *item = new QTreeWidgetItem(runs_, {obj.value("name").toString(QFileInfo(path).completeBaseName()), QString::number(obj.value("best_epoch").toInt()), validation});
            item->setData(0, Qt::UserRole, path); item->setToolTip(0, path);
            item->setToolTip(2, QString::fromUtf8(QJsonDocument(metrics).toJson(QJsonDocument::Indented)));
            if (path == oldRun) runs_->setCurrentItem(item);
        }
        if (!datasets_->currentItem() && datasets_->topLevelItemCount()) datasets_->setCurrentItem(datasets_->topLevelItem(0));
        if (!runs_->currentItem() && runs_->topLevelItemCount()) runs_->setCurrentItem(runs_->topLevelItem(0));
        if (collectionSources_->topLevelItemCount() == 1) {
            auto *item = collectionSources_->topLevelItem(0);
            if (!collectionRoles.contains(item->data(0, Qt::UserRole).toString())) { item->setData(0, Qt::UserRole + 2, "train"); item->setText(1, "Training"); }
        }
        syncDatasetSelection(datasets_, collectionSources_);
        datasetBlocker.unblock(); collectionBlocker.unblock(); loadSplitControls(selectedPath(datasets_)); updateCollectionReadiness(); updateTrainingSelection(); updateCleanupActions();
        if (datasetMode_ && tabs_->currentIndex() == 1 && pendingReview_.isEmpty() && !selectedPath(datasets_).isEmpty() && selectedPath(datasets_) != reviewedDataset_) reviewDataset(selectedPath(datasets_), false);
    }

    void importProjectDatasets() {
        if (!datasetMode_) { openApp("datasets", "prepare"); return; }
        QDialog dialog(this); dialog.setObjectName("linkProjectDatasetsDialog"); dialog.setWindowTitle("Link datasets from another project"); dialog.resize(640, 430); auto *layout = new QVBoxLayout(&dialog);
        auto *hint = new QLabel("Choose a source project, then select datasets to reference in this project. Their files stay in the source location.", &dialog); hint->setWordWrap(true); layout->addWidget(hint);
        auto *row = new QHBoxLayout; auto *source = new QComboBox(&dialog); source->setSizeAdjustPolicy(QComboBox::AdjustToMinimumContentsLengthWithIcon); source->setMinimumContentsLength(20); row->addWidget(source, 1); auto *browse = new QPushButton("Other project…", &dialog); row->addWidget(browse); layout->addLayout(row);
        auto *datasets = new QListWidget(&dialog); datasets->setObjectName("linkProjectDatasets"); datasets->setSelectionMode(QAbstractItemView::ExtendedSelection); layout->addWidget(datasets, 1); auto *notice = new QLabel(&dialog); notice->setWordWrap(true); layout->addWidget(notice);
        auto populate = [&] {
            datasets->clear(); for (const QString &path : projectDatasets(source->currentData().toString())) {
                auto *item = new QListWidgetItem(QFileInfo(path).fileName(), datasets); item->setToolTip(path); item->setData(Qt::UserRole, path); item->setFlags(item->flags() & ~Qt::ItemIsUserCheckable); item->setSelected(true);
                if (readJson(QDir(path).filePath("dataset.json")).value("schema").toString() != "ipde-depth-dataset-v1") { item->setText(item->text() + " (unavailable)"); item->setSelected(false); item->setFlags(item->flags() & ~Qt::ItemIsEnabled); }
            }
            notice->setText(datasets->count() ? "Linked datasets are available in this project's Dataset Manager and Trainer." : "This project does not contain any datasets yet.");
        };
        connect(source, &QComboBox::currentIndexChanged, &dialog, populate);
        QSettings recent("IPDE", "Studio");
        for (const QString &path : recent.value("projects").toStringList()) if (path != projectRoot_)
            source->addItem(QSettings(QDir(path).filePath("project.ini"), QSettings::IniFormat).value("name", QFileInfo(path).fileName()).toString(), path);
        connect(browse, &QPushButton::clicked, &dialog, [&] {
            const QString path = QFileDialog::getExistingDirectory(&dialog, "Choose source project", QStandardPaths::writableLocation(QStandardPaths::DocumentsLocation)); if (path.isEmpty()) return;
            const QString canonical = QFileInfo(path).canonicalFilePath(); if (canonical == projectRoot_) { notice->setText("Choose a different project; this project's own datasets are already available."); return; }
            int index = source->findData(canonical); if (index < 0) { source->addItem(QFileInfo(path).fileName(), canonical); index = source->count()-1; } source->setCurrentIndex(index); populate();
        });
        auto *buttons = new QDialogButtonBox(QDialogButtonBox::Cancel | QDialogButtonBox::Ok, &dialog); buttons->button(QDialogButtonBox::Ok)->setText("Link selected datasets"); layout->addWidget(buttons);
        connect(buttons, &QDialogButtonBox::rejected, &dialog, &QDialog::reject);
        connect(buttons, &QDialogButtonBox::accepted, &dialog, [&] {
            QStringList paths; for (int i=0; i<datasets->count(); ++i) if (datasets->item(i)->isSelected()) paths << datasets->item(i)->data(Qt::UserRole).toString();
            if (paths.isEmpty()) { notice->setText("Select at least one available dataset."); return; }
            addDatasetLinks(paths); dialog.accept();
        });
        populate(); dialog.exec();
    }
    void addDatasetLinks(const QStringList &paths) {
        if (!datasetMode_) { openApp("datasets", "prepare"); return; }
        const QString project = projectRoot_; if (project.isEmpty() || !projectSettings_) return;
        QStringList links = linkedDatasets(project); const QStringList existing = projectDatasets(project); int added = 0; QStringList invalid;
        for (const QString &path : paths) {
            const QString canonical = QFileInfo(path).canonicalFilePath();
            if (canonical.isEmpty() || readJson(QDir(canonical).filePath("dataset.json")).value("schema").toString() != "ipde-depth-dataset-v1") { invalid << path; continue; }
            if (!existing.contains(canonical) && !links.contains(canonical)) { links << canonical; ++added; }
        }
        QSettings settings(QDir(project).filePath("project.ini"), QSettings::IniFormat); settings.setValue("dataset_links", links); settings.sync();
        statusBar()->showMessage(settings.status() != QSettings::NoError ? "Could not save dataset links." : QString("Linked %1 datasets without copying files.%2").arg(added).arg(invalid.isEmpty() ? QString() : " Unavailable folders: " + invalid.join(", ")));
        refreshLibrary();
    }
    void editDatasetLinks() {
        if (!datasetMode_) { openApp("datasets", "prepare"); return; }
        const QString project = projectRoot_; if (project.isEmpty() || !projectSettings_) return;
        QDialog dialog(this); dialog.setObjectName("datasetLinksDialog"); dialog.setWindowTitle("Dataset links"); dialog.resize(650, 330); auto *layout = new QVBoxLayout(&dialog);
        auto *hint = new QLabel("Select links and click Remove selected links, then Save. The original dataset files remain in place.", &dialog); hint->setWordWrap(true); layout->addWidget(hint);
        auto *list = new QListWidget(&dialog); list->setObjectName("datasetLinks"); list->setSelectionMode(QAbstractItemView::ExtendedSelection); layout->addWidget(list, 1);
        for (const QString &path : linkedDatasets(project)) { auto *item = new QListWidgetItem(path, list); item->setData(Qt::UserRole, path); item->setFlags(item->flags() & ~Qt::ItemIsUserCheckable); }
        auto *remove = new QPushButton("Remove selected links", &dialog); layout->addWidget(remove);
        connect(remove, &QPushButton::clicked, &dialog, [list] { qDeleteAll(list->selectedItems()); });
        auto *buttons = new QDialogButtonBox(QDialogButtonBox::Cancel | QDialogButtonBox::Save, &dialog); layout->addWidget(buttons);
        connect(buttons, &QDialogButtonBox::rejected, &dialog, &QDialog::reject);
        connect(buttons, &QDialogButtonBox::accepted, &dialog, [&] {
            QStringList links; for (int i=0; i<list->count(); ++i) links << list->item(i)->data(Qt::UserRole).toString();
            QSettings settings(QDir(project).filePath("project.ini"), QSettings::IniFormat); settings.setValue("dataset_links", links); settings.sync(); refreshLibrary(); dialog.accept();
        }); dialog.exec();
    }

    void showFolder(const QString &path) {
        if (!path.isEmpty()) QDesktopServices::openUrl(QUrl::fromLocalFile(path));
    }

    QSettings settings_;
    std::unique_ptr<QSettings> projectSettings_;
    QString projectRoot_; bool datasetMode_ = false;
    IPDE::ProjectSession *session_ = nullptr;
    QJsonObject projectOperations_, lastAppRequest_;
    QString activeOperation_;
    bool continueToTrainer_ = false;
    QLineEdit *workspace_ = nullptr, *datasetName_ = nullptr, *runName_ = nullptr;
    QLineEdit *teacherPath_ = nullptr, *teacherSource_ = nullptr, *raftRoot_ = nullptr, *raftModel_ = nullptr;
    QLineEdit *reviewedName_ = nullptr, *category_ = nullptr, *reviewFilter_ = nullptr, *collectionName_ = nullptr;
    QLineEdit *hfSource_ = nullptr, *hfConfig_ = nullptr, *hfSplit_ = nullptr, *hfMapping_ = nullptr, *hfDataFiles_ = nullptr, *hfAssetDir_ = nullptr;
    QMap<QString, QLineEdit *> modelPaths_, modelSources_;
    QTreeWidget *datasets_ = nullptr, *runs_ = nullptr, *sources_ = nullptr;
    QTreeWidget *reviewSamples_ = nullptr, *collectionSources_ = nullptr;
    QComboBox *teacher_ = nullptr, *teacherDevice_ = nullptr, *trainDevice_ = nullptr, *scope_ = nullptr, *trainingMode_ = nullptr, *student_ = nullptr;
    QComboBox *reviewLabel_ = nullptr, *goal_ = nullptr, *reviewCamera_ = nullptr, *visualView_ = nullptr, *splitMode_ = nullptr, *groupingPolicy_ = nullptr;
    QLabel *scaleHelp_ = nullptr, *reviewPath_ = nullptr, *reviewCount_ = nullptr, *previewStats_ = nullptr, *goalHelp_ = nullptr, *splitHelp_ = nullptr, *collectionStatus_ = nullptr;
    QLabel *trainingStatus_ = nullptr, *trainingDataset_ = nullptr, *sourceTitle_ = nullptr, *patchLabel_ = nullptr, *trainingHelp_ = nullptr, *trainingLabelHelp_ = nullptr;
    QList<QLabel *> depthTitles_;
    DepthPreview *rgbPreview_ = nullptr, *depthPreview_ = nullptr;
    QList<DepthPreview *> depthPreviews_;
    QSplitter *library_ = nullptr;
    QPushButton *verifiedScenes_ = nullptr, *anchor_ = nullptr, *advanced_ = nullptr, *useGroups_ = nullptr, *compareTeachers_ = nullptr;
    QList<QPushButton *> teacherChecks_;
    QGroupBox *datasetAdvanced_ = nullptr, *trainingAdvanced_ = nullptr;
    QSpinBox *inputSize_ = nullptr, *epochs_ = nullptr, *steps_ = nullptr, *patch_ = nullptr, *iterations_ = nullptr;
    QWidget *trainingSimple_ = nullptr;
    QSlider *quality_ = nullptr, *length_ = nullptr;
    QComboBox *limitMode_ = nullptr, *validationSchedule_ = nullptr, *checkpointSchedule_ = nullptr;
    QSpinBox *stepsPerUpdate_ = nullptr, *validationSamples_ = nullptr, *checkpointEvery_ = nullptr;
    QDoubleSpinBox *earlyStopError_ = nullptr, *maxLoss_ = nullptr, *learningRate_ = nullptr;
    QPushButton *earlyStop_ = nullptr, *saveCheckpoint_ = nullptr;
    QLineEdit *resumePath_ = nullptr; QString trainingControlPath_; bool trainingStopping_ = false, closeAfterTraining_ = false;
    QSpinBox *splitSeed_ = nullptr, *validationCount_ = nullptr;
    QDoubleSpinBox *validationFraction_ = nullptr; QFormLayout *collectionForm_ = nullptr;
    QTabWidget *tabs_ = nullptr; QPlainTextEdit *log_ = nullptr; QProgressBar *progress_ = nullptr;
    QPushButton *chooseWorkspace_ = nullptr, *refresh_ = nullptr, *generate_ = nullptr, *train_ = nullptr, *cancel_ = nullptr, *export_ = nullptr;
    QPushButton *useDatasetForTraining_ = nullptr, *useDatasetForValidation_ = nullptr, *skipDataset_ = nullptr;
    QPushButton *saveReviewed_ = nullptr, *compose_ = nullptr, *importHf_ = nullptr, *compareBaseline_ = nullptr;
    QPushButton *cleanupDataset_ = nullptr, *cleanupRun_ = nullptr;
    QProcess *process_ = nullptr; QByteArray stdout_; QString job_, exportDestination_;
    QProcess *exportProcess_ = nullptr;
    QProcess *editProcess_ = nullptr; QTimer *autosaveTimer_ = nullptr; QByteArray editStdout_;
    std::unique_ptr<QTemporaryFile> editSaveFile_;
    QString draftWorkspace_, editingDataset_, trainerAfterSave_; QJsonObject savingDraft_;
    QString splitControlsDataset_;
    QMap<QString, QString> manifestHashes_; int editSerial_ = 0;
    bool closeAfterSave_ = false, draftStorageError_ = false, journalRecoveryNeeded_ = false, autosavePaused_ = false;
    QString requestedReviewPath_, reviewedDataset_, pendingReview_, pendingTrainingPath_;
    QProcess *previewProcess_ = nullptr; QByteArray previewStdout_, progressBuffer_; QString previewKind_, pendingPreviewKind_, streamingDataset_, differencePath_;
    QStringList pendingPreviewArgs_; QJsonArray previewRecords_; QTimer *streamingTimer_ = nullptr;
    QSet<QString> scanDeliveredPaths_;
    bool cancelled_ = false, refreshAfter_ = false, reviewGenerating_ = false, requestedReviewOpen_ = true, busy_ = false;
    std::unique_ptr<QTemporaryFile> groupsFile_, teachersFile_, editsFile_;
    QMap<QString, QJsonObject> reviewDrafts_;
    QMap<QString, QStringList> reviewAdditions_; QString savingVersionSource_;
    QString appendBase_, appendVersionName_; QJsonObject appendEdits_;
    QLabel *addingPhotosHint_ = nullptr; QPushButton *cancelAdding_ = nullptr;
    QPushButton *addPhotos_ = nullptr, *removePhotos_ = nullptr, *restorePhotos_ = nullptr; QList<QPushButton *> editActions_;
    QPushButton *openDatasetTrainer_ = nullptr, *pendingPhotosAction_ = nullptr; QStringList pendingPhotoPaths_;
    QLabel *reviewSaveStatus_ = nullptr;
    std::unique_ptr<QTemporaryDir> reviewPreviews_;
};

} // namespace

int main(int argc, char **argv) {
    QApplication application(argc, argv);
    const QStringList args = application.arguments(); const int mode = args.indexOf("--mode");
    std::unique_ptr<QTemporaryDir> smokeSettings;
    if (args.contains("--smoke-test")) {
        smokeSettings = std::make_unique<QTemporaryDir>();
        if (!smokeSettings->isValid()) return 2;
        QSettings::setDefaultFormat(QSettings::IniFormat);
        QSettings::setPath(QSettings::IniFormat, QSettings::UserScope, smokeSettings->path());
    }
    const bool datasetMode = mode >= 0 ? args.value(mode + 1) == "datasets" : bool(IPDE_DATASET_STUDIO);
    application.setApplicationName(datasetMode ? "Dataset Studio" : "RAFT Studio"); application.setOrganizationName("IPDE");
    IPDE::ProjectSession session(datasetMode ? "datasets" : "trainer", &application);
    if (!session.start()) return 2;
    TrainerWindow window; window.attachSession(&session); window.show();
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
            if (args.contains("--smoke-test")) window.resize(1440, 1000);
            const bool saved = window.grab().save(args.at(screenshot + 1));
            application.exit(saved ? 0 : 2);
        });
        capture->start();
    }
    return application.exec();
}
