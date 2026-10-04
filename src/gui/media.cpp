#include "project_session.h"
#include "studio_icons.h"
#include <QApplication>
#include <QCheckBox>
#include <QClipboard>
#include <QCloseEvent>
#include <QComboBox>
#include <QDesktopServices>
#include <QDragEnterEvent>
#include <QDialog>
#include <QDropEvent>
#include <QFile>
#include <QFileDialog>
#include <QFormLayout>
#include <QGroupBox>
#include <QJsonArray>
#include <QLabel>
#include <QLineEdit>
#include <QMimeData>
#include <QPlainTextEdit>
#include <QPointer>
#include <QProcess>
#include <QProcessEnvironment>
#include <QPushButton>
#include <QSettings>
#include <QSignalBlocker>
#include <QScrollArea>
#include <QSplitter>
#include <QStandardPaths>
#include <QStandardItemModel>
#include <QStatusBar>
#include <QTabWidget>
#include <QTemporaryDir>
#include <QTimer>
#include <QTreeWidget>
#include <QVBoxLayout>
#include <QMainWindow>
#include <memory>

#ifndef IPDE_PYTHON_EXECUTABLE
#define IPDE_PYTHON_EXECUTABLE "python3"
#endif
#ifndef MEDIA_STUDIO_SOURCE_SCRIPT
#define MEDIA_STUDIO_SOURCE_SCRIPT "media_studio.py"
#endif

namespace {
#ifdef PHOTO_STUDIO
constexpr bool photo = true;
#else
constexpr bool photo = false;
#endif
QString role() { return photo ? "photo" : "raw"; }
QString appName() { return photo ? "Photo Studio" : "Raw Studio"; }
QString scriptPath() {
    const auto bundled = QDir(QCoreApplication::applicationDirPath()).filePath("../Resources/media_studio.py");
    return QFileInfo::exists(bundled) ? QDir::cleanPath(bundled) : QString::fromUtf8(MEDIA_STUDIO_SOURCE_SCRIPT);
}

class Preview final : public QLabel {
public:
    explicit Preview(QWidget *parent) : QLabel(parent) {
        setAlignment(Qt::AlignCenter); setMinimumSize(320, 270); setWordWrap(true);
        setSizePolicy(QSizePolicy::Expanding, QSizePolicy::Expanding);
        setStyleSheet("background:#172435;color:#c9d4e1;border-radius:10px;padding:12px;");
        setText("Import a photograph to inspect its available data.");
    }
    void load(const QString &path) { image_ = QPixmap(path); if (image_.isNull()) reset("Preview could not be loaded."); else refresh(); }
    void reset(const QString &message) { image_ = QPixmap(); clear(); setText(message); }
protected:
    void resizeEvent(QResizeEvent *event) override { QLabel::resizeEvent(event); refresh(); }
private:
    void refresh() { if (!image_.isNull()) setPixmap(image_.scaled(size() - QSize(24, 24), Qt::KeepAspectRatio, Qt::SmoothTransformation)); }
    QPixmap image_;
};

class MediaWindow final : public QMainWindow {
public:
    MediaWindow() {
        setWindowTitle(appName()); setWindowIcon(IPDE::appIcon(role())); resize(1100, 860); setAcceptDrops(true);
        previews_ = std::make_unique<QTemporaryDir>();
        auto *central = new QWidget(this); auto *layout = new QVBoxLayout(central); layout->setContentsMargins(24, 20, 24, 20); layout->setSpacing(12);
        auto *heading = new QHBoxLayout;
        auto *icon = new QLabel(central); icon->setPixmap(IPDE::appIcon(role()).pixmap(52, 52)); heading->addWidget(icon);
        auto *title = new QLabel(appName(), central); QFont font = title->font(); font.setPointSize(font.pointSize() + 9); font.setBold(true); title->setFont(font); heading->addWidget(title); heading->addStretch(); layout->addLayout(heading);
        purpose_ = new QLabel(central); layout->addWidget(purpose_); reloadProjectSettings();
        auto *purposeRow = new QHBoxLayout; purposeRow->addWidget(new QLabel("Main purpose", central));
        auto *purposeChoice = new QComboBox(central); purposeChoice->addItem("Effects / displacement", "effect/map"); purposeChoice->addItem("Depth estimation", "depth-estimation"); purposeChoice->addItem("Portrait / photo effects", "photo-effects"); purposeChoice->addItem("Custom", "manual");
        QSettings initial(QDir(IPDE::projectRoot()).filePath("project.ini"), QSettings::IniFormat); purposeChoice->setCurrentIndex(qMax(0, purposeChoice->findData(initial.value("goal", "manual").toString()))); purposeChoice->setEnabled(!IPDE::projectRoot().isEmpty()); purposeChoice_ = purposeChoice;
        purposeRow->addWidget(purposeChoice, 1); layout->addLayout(purposeRow);
        connect(purposeChoice, &QComboBox::currentIndexChanged, this, [this] { const auto project = IPDE::projectRoot(); if (project.isEmpty()) return; QSettings settings(QDir(project).filePath("project.ini"), QSettings::IniFormat); settings.setValue("goal", purposeChoice_->currentData()); settings.sync(); reloadProjectSettings(); });
        auto *intro = new QLabel(photo
            ? "Inspect portrait depth and person mattes, choose the layers to keep, then export a composite or a precise array. Source photographs stay available for comparison."
            : "Capture stored RAW samples, camera metadata and embedded planes first. Create separate camera RGB, depth estimates or person masks when your workflow needs them.", central);
        intro->setWordWrap(true); layout->addWidget(intro);
        auto *sourceRow = new QHBoxLayout; source_ = new QLineEdit(central); source_->setReadOnly(true); source_->setPlaceholderText("Drop a photograph here or choose Import…");
        import_ = new QPushButton("Import…", central); import_->setIcon(IPDE::appIcon(role())); sourceRow->addWidget(source_, 1); sourceRow->addWidget(import_); layout->addLayout(sourceRow);
        auto *splitter = new QSplitter(central);
        assets_ = new QTreeWidget(splitter); assets_->setHeaderLabels({"Available data", "Precision / grid"}); assets_->setMinimumWidth(390); assets_->setRootIsDecorated(false); assets_->setColumnWidth(0, 190);
        preview_ = new Preview(splitter); splitter->setSizes({350, 650}); layout->addWidget(splitter, 1);
        previewNote_ = new QLabel("Previews are display copies. Array exports preserve stored values and precision.", central); previewNote_->setWordWrap(true); layout->addWidget(previewNote_);
        auto *controlsScroll = new QScrollArea(central); controlsScroll->setWidgetResizable(true); controlsScroll->setFrameShape(QFrame::NoFrame); controlsScroll->setMinimumHeight(240); controlsScroll->setMaximumHeight(380);
        auto *controls = new QWidget(controlsScroll); auto *controlsRoot = new QVBoxLayout(controls); controlsRoot->setContentsMargins(0, 0, 12, 0); controlsScroll->setWidget(controls); layout->addWidget(controlsScroll);
        auto *options = new QGroupBox(photo ? "Create a photo product" : "Capture and create products", central); auto *form = new QFormLayout(options); optionsForm_ = form;
        operation_ = new QComboBox(options);
        if (photo) {
            operation_->addItem("Exact selected image or auxiliary plane", "original");
            operation_->addItem("Embedded depth on the photo grid — derived approximation", "depth-upscale");
            operation_->addItem("Subject cutout from checked person matte layers", "cutout");
            operation_->addItem("Isolate color using the selected matte", "isolate");
            operation_->addItem("Detailed depth estimate from a trainer teacher", "learned-depth");
            operation_->addItem("Depth from a selected RAFT model — spatial pair required", "raft-depth");
        } else {
            operation_->addItem("Exact stored RAW samples and camera metadata", "sensor");
            operation_->addItem("Original camera RAW file — exact bytes", "original");
            operation_->addItem("Linear camera RGB — developed derivative", "rendered");
            operation_->addItem("Exact selected embedded auxiliary plane", "auxiliary");
            operation_->addItem("Depth estimate on the camera RGB grid", "learned-depth");
            operation_->addItem("Person mask — Apple Vision estimate", "person-mask");
        }
        form->addRow("Product", operation_);
        auto *formatRow = new QWidget(options); auto *formats = new QHBoxLayout(formatRow); formats->setContentsMargins(0, 0, 0, 0);
        format_ = new QComboBox(formatRow); format_->addItem("NumPy — exact array", "npy"); format_->addItem("TIFF — integer or float", "tiff"); format_->addItem("PNG — integer image", "png"); format_->addItem("OpenEXR — float data", "exr");
        formats->addWidget(format_); auto *formatHelp = new QLabel("Choose a format that can store this product. Incompatible formats report an error.", formatRow); formatHelp->setWordWrap(true); formats->addWidget(formatHelp, 1); form->addRow("Export format", formatRow);
        auto *models = new QWidget(options); auto *modelLayout = new QHBoxLayout(models); modelLayout->setContentsMargins(0, 0, 0, 0);
        model_ = new QComboBox(models); model_->addItem("DepthPro — estimated meters", "depthpro"); model_->addItem("Depth Anything V2 — relative", "depth-anything-v2"); model_->addItem("Depth Anything 3 — relative", "depth-anything-3"); modelLayout->addWidget(model_);
        device_ = new QComboBox(models); for (const auto &d : {"auto", "cpu", "mps", "cuda"}) device_->addItem(d); modelLayout->addWidget(device_); form->addRow("Teacher / device", models);
        modelPath_ = new QLineEdit(options); modelPath_->setPlaceholderText("Automatic local teacher lookup, or paste a checkpoint path"); form->addRow("Teacher checkpoint", modelPath_);
        modelRoot_ = new QLineEdit(options); modelRoot_->setPlaceholderText("Automatic local source lookup, or paste the teacher source directory"); form->addRow("Teacher source", modelRoot_);
        raftPath_ = new QLineEdit(options); raftPath_->setPlaceholderText("Project RAFT checkpoint, or paste an exported custom checkpoint path"); form->addRow("RAFT checkpoint", raftPath_);
        crop_ = new QLineEdit(options); crop_->setPlaceholderText("Optional X,Y,width,height in camera RGB pixels; depth is cropped on the same grid");
        if (!photo) form->addRow("Region", crop_); else crop_->hide();
        controlsRoot->addWidget(options);
        matteHelp_ = new QLabel(photo ? "Person mattes describe people. Check the portrait matte and useful semantic layers, preview the union, and uncheck a bad layer. Selecting a single hair, skin or other semantic matte isolates that region."
            : "Stored RAW samples may be a sensor mosaic or demosaiced LinearRaw. Calibration and any CFA pattern are exported with the samples. Camera RGB and Vision masks are derived products. Person masks are intended for people.", central);
        matteHelp_->setWordWrap(true); controlsRoot->addWidget(matteHelp_);
        auto *outputRow = new QHBoxLayout; outputRow->addWidget(new QLabel("Output", central)); output_ = new QLineEdit(defaultOutput(), central); outputRow->addWidget(output_, 1); auto *browse = new QPushButton("Choose…", central); outputRow->addWidget(browse); layout->addLayout(outputRow);
        auto *actions = new QHBoxLayout; composite_ = new QPushButton("Preview composite", central); composite_->setVisible(photo); actions->addWidget(composite_);
        export_ = new QPushButton("Export product", central); export_->setIcon(IPDE::appIcon("extractor")); actions->addWidget(export_);
        clipboard_ = new QPushButton("Export & copy", central); clipboard_->setToolTip("Copies exact PNG bytes for image pasting and exported file URLs for Finder. Other formats are copied as files."); actions->addWidget(clipboard_);
        auto *showOutput = new QPushButton("Show exports", central); actions->addWidget(showOutput); auto *metadata = new QPushButton("Inspect metadata…", central); actions->addWidget(metadata); cancel_ = new QPushButton("Cancel", central); cancel_->setEnabled(false); actions->addStretch(); actions->addWidget(cancel_); layout->addLayout(actions);
        connect(metadata, &QPushButton::clicked, this, [this] {
            QDialog dialog(this); dialog.setWindowTitle("Image data and calibration"); dialog.resize(780, 570); auto *root = new QVBoxLayout(&dialog);
            auto *help = new QLabel(photo ? "Each plane records its camera grid and stored precision. Depth representation and calibration determine how values can be interpreted; gain maps are HDR reconstruction data."
                : "Black level is the sensor's zero-light offset; white level describes saturation. CFA/color pattern and matrices describe demosaicing and color calibration. Sensor margins are retained in the full mosaic. RGB development uses a separate processing path.", &dialog); help->setWordWrap(true); root->addWidget(help); auto *text = new QPlainTextEdit(&dialog); text->setReadOnly(true); text->setPlainText(QJsonDocument(inspectReport_).toJson(QJsonDocument::Indented)); root->addWidget(text, 1); auto *close = new QPushButton("Close", &dialog); root->addWidget(close); connect(close, &QPushButton::clicked, &dialog, &QDialog::accept); dialog.exec();
        });
        if (photo) {
            auto *rewrite = new QGroupBox("Repair or share a HEIC", central); auto *rewriteLayout = new QVBoxLayout(rewrite);
            auto *text = new QLabel("Select an embedded plane and provide its replacement as a precise NPY array. HEIC saves verify retained pixels, auxiliary planes and spatial structure. If a compatible lossless rewrite cannot be verified, the save is refused.", rewrite); text->setWordWrap(true); rewriteLayout->addWidget(text);
            replacements_ = new QLabel("No replacement planes selected.", rewrite); replacements_->setWordWrap(true); rewriteLayout->addWidget(replacements_);
            auto *representationRow = new QHBoxLayout; representationRow->addWidget(new QLabel("Replacement values", rewrite)); replacementKind_ = new QComboBox(rewrite); replacementKind_->addItem("Same representation as selected embedded plane", "native"); replacementKind_->addItem("Depth in meters (DepthPro or calibrated RAFT)", "depth"); replacementKind_->addItem("Disparity in inverse meters", "disparity"); representationRow->addWidget(replacementKind_, 1); rewriteLayout->addLayout(representationRow);
            auto *row = new QHBoxLayout; replace_ = new QPushButton("Replace selected plane…", rewrite); row->addWidget(replace_); auto *clear = new QPushButton("Clear replacements", rewrite); row->addWidget(clear);
            privacy_ = new QCheckBox("Privacy export — remove identifying metadata, randomize names and IDs", rewrite); privacy_->setChecked(true); row->addWidget(privacy_, 1);
            saveHeic_ = new QPushButton("Save HEIC…", rewrite); row->addWidget(saveHeic_); rewriteLayout->addLayout(row); controlsRoot->addWidget(rewrite);
            connect(replace_, &QPushButton::clicked, this, [this] { chooseReplacement(); });
            connect(clear, &QPushButton::clicked, this, [this] { replacementsMap_.clear(); replacements_->setText("No replacement planes selected."); updateEnabled(); });
            connect(saveHeic_, &QPushButton::clicked, this, [this] { rewriteHeic(); });
            connect(privacy_, &QCheckBox::toggled, this, [this] { updateEnabled(); });
        }
        log_ = new QPlainTextEdit(central); log_->setReadOnly(true); log_->setMaximumHeight(90); log_->setMaximumBlockCount(150); layout->addWidget(log_); setCentralWidget(central);
        process_ = new QProcess(this);
        connect(process_, &QProcess::readyReadStandardOutput, this, [this] { stdout_ += process_->readAllStandardOutput(); });
        connect(process_, &QProcess::readyReadStandardError, this, [this] { stderr_ += process_->readAllStandardError(); });
        connect(process_, qOverload<int, QProcess::ExitStatus>(&QProcess::finished), this, [this](int code, QProcess::ExitStatus status) { finished(code, status); });
        connect(process_, &QProcess::errorOccurred, this, [this](QProcess::ProcessError error) {
            if (error == QProcess::FailedToStart) { busy_ = false; log_->appendPlainText("Python could not start: " + process_->errorString()); updateEnabled(); }
        });
        connect(import_, &QPushButton::clicked, this, [this] { const auto path = QFileDialog::getOpenFileName(this, "Import photograph", QStandardPaths::writableLocation(QStandardPaths::PicturesLocation), photo ? "HEIF portraits and spatial photos (*.heic *.HEIC *.heif *.HEIF);;All files (*)" : "Camera RAW (*.dng *.DNG *.cr2 *.CR2 *.cr3 *.CR3 *.nef *.NEF *.arw *.ARW *.raf *.RAF *.orf *.ORF *.rw2 *.RW2);;All files (*)"); if (!path.isEmpty()) inspect(path); });
        connect(browse, &QPushButton::clicked, this, [this] { const auto path = QFileDialog::getExistingDirectory(this, "Choose export folder", output_->text()); if (!path.isEmpty()) output_->setText(path); });
        connect(showOutput, &QPushButton::clicked, this, [this] { QDir().mkpath(output_->text()); QDesktopServices::openUrl(QUrl::fromLocalFile(output_->text())); });
        connect(export_, &QPushButton::clicked, this, [this] { exportProduct(false, false); });
        connect(clipboard_, &QPushButton::clicked, this, [this] { exportProduct(true, false); });
        connect(composite_, &QPushButton::clicked, this, [this] { exportProduct(false, true); });
        connect(cancel_, &QPushButton::clicked, this, [this] { if (busy_) { cancelled_ = true; process_->kill(); } });
        connect(assets_, &QTreeWidget::currentItemChanged, this, [this](QTreeWidgetItem *item) { if (item) { const auto path = item->data(0, Qt::UserRole + 1).toString(); if (!path.isEmpty()) preview_->load(path); } updateEnabled(); });
        connect(operation_, &QComboBox::currentIndexChanged, this, [this] { updateEnabled(); });
        reloadProjectSettings(); updateEnabled();
        const auto args = QCoreApplication::arguments(); const int input = args.indexOf("--input");
        if (input >= 0 && input + 1 < args.size()) QTimer::singleShot(0, this, [this, args, input] { inspect(args[input + 1]); });
    }
    ~MediaWindow() override { process_->disconnect(this); if (process_->state() != QProcess::NotRunning) { process_->kill(); process_->waitForFinished(3000); } }
    bool busy() const { return busy_; }
    void reloadProjectSettings() {
        const QString project = IPDE::projectRoot(); QSettings settings(QDir(project).filePath("project.ini"), QSettings::IniFormat);
        const QString goal = settings.value("goal", "manual").toString();
        const QString goalText = goal == "effect/map" ? "Effect / displacement maps" : goal == "depth-estimation" ? "Depth estimation" : goal == "photo-effects" ? "Portrait photo effects" : "Custom";
        purpose_->setText("Project: " + (project.isEmpty() ? QString("smoke preview") : settings.value("name", QFileInfo(project).fileName()).toString()) + "  ·  Main purpose: " + goalText + " (editable in Studio)");
        if (raftPath_) raftPath_->setText(settings.value("raft/model").toString());
        if (purposeChoice_) { const QSignalBlocker blocker(purposeChoice_); purposeChoice_->setCurrentIndex(qMax(0, purposeChoice_->findData(goal))); }
    }
protected:
    void closeEvent(QCloseEvent *event) override { if (busy_) { statusBar()->showMessage("Cancel the running operation before closing."); event->ignore(); } else event->accept(); }
    void dragEnterEvent(QDragEnterEvent *event) override { if (!busy_ && event->mimeData()->hasUrls()) event->acceptProposedAction(); }
    void dropEvent(QDropEvent *event) override { if (!busy_) for (const auto &url : event->mimeData()->urls()) if (url.isLocalFile()) { inspect(url.toLocalFile()); break; } }
private:
    QString defaultOutput() const { const auto project = IPDE::projectRoot(); return project.isEmpty() ? QDir(QStandardPaths::writableLocation(QStandardPaths::DocumentsLocation)).filePath("IPDE Exports/" + appName()) : QDir(project).filePath("workspace/exports/" + role()); }
    void updateEnabled() {
        const bool ready = !busy_ && !source_->text().isEmpty() && loaded_; import_->setEnabled(!busy_); operation_->setEnabled(!busy_); format_->setEnabled(!busy_);
        assets_->setEnabled(!busy_); export_->setEnabled(ready); clipboard_->setEnabled(ready); composite_->setEnabled(ready); cancel_->setEnabled(busy_);
        if (loaded_) {
            const auto assets = inspectReport_.value("assets").toArray(); bool depth = false, mattes = false, rendered = false, auxiliary = false;
            for (const auto &entry : assets) { const auto a = entry.toObject(); const auto kind = a.value("kind").toString(); depth |= kind == "native_depth" || kind == "depth"; mattes |= a.value("person_matte").toBool(); rendered |= kind == "rendered-linear-rgb"; auxiliary |= kind == "auxiliary"; }
            auto *choices = qobject_cast<QStandardItemModel *>(operation_->model());
            if (choices) for (int index = 0; index < operation_->count(); ++index) {
                const auto op = operation_->itemData(index).toString(); bool available = true;
                if (photo) { if (op == "depth-upscale") available = depth; if (op == "cutout" || op == "isolate") available = mattes; if (op == "raft-depth") available = inspectReport_.value("spatial").isObject() && !inspectReport_.value("spatial").toObject().isEmpty(); }
                else { if (op == "sensor") available = capabilities_.value("sensor").toBool(); if (op == "rendered" || op == "learned-depth") available = rendered; if (op == "person-mask") available = capabilities_.value("person_mask").toBool(); if (op == "auxiliary") available = auxiliary; }
                choices->item(index)->setEnabled(available);
            }
            if (choices && !choices->item(operation_->currentIndex())->isEnabled()) { const QSignalBlocker blocker(operation_); operation_->setCurrentIndex(qMax(0, operation_->findData("original"))); }
            composite_->setEnabled(ready && mattes);
        }
        const bool teacher = operation_->currentData() == "learned-depth", raft = operation_->currentData() == "raft-depth";
        model_->setEnabled(teacher && !busy_); model_->setVisible(teacher); modelPath_->setEnabled(teacher && !busy_); modelRoot_->setEnabled(teacher && !busy_); device_->setEnabled(!busy_ && (teacher || raft)); raftPath_->setEnabled(!busy_ && raft);
        optionsForm_->setRowVisible(model_->parentWidget(), teacher || raft); optionsForm_->setRowVisible(modelPath_, teacher); optionsForm_->setRowVisible(modelRoot_, teacher); optionsForm_->setRowVisible(raftPath_, raft);
        crop_->setPlaceholderText(operation_->currentData() == "sensor" ? "Optional X,Y,width,height in the stored RAW raster, including any sensor margins" : "Optional X,Y,width,height in camera RGB pixels; depth uses the same grid");
        if (saveHeic_) {
            privacy_->setEnabled(!busy_ && replacementsMap_.isEmpty());
            privacy_->setToolTip(replacementsMap_.isEmpty() ? "Remove recorded identifiers and save under a randomized name." : "Save the verified repair first, then import that HEIC and create its privacy copy.");
            saveHeic_->setEnabled(ready && (privacy_->isChecked() ? capabilities_.value("privacy").toBool() : capabilities_.value("rewrite").toBool()));
            replace_->setEnabled(ready && assets_->currentItem() && assets_->currentItem()->data(0, Qt::UserRole + 2).toBool() && capabilities_.value("rewrite").toBool());
        }
    }
    void run(QStringList args, const QString &job) {
        if (busy_) return; busy_ = true; cancelled_ = false; job_ = job; stdout_.clear(); stderr_.clear();
        process_->setProgram(QString::fromUtf8(IPDE_PYTHON_EXECUTABLE)); process_->setArguments(QStringList{scriptPath()} + args);
        auto env = QProcessEnvironment::systemEnvironment(); const QString helper = QDir(QCoreApplication::applicationDirPath()).filePath("../Resources/media-bridge"); if (QFileInfo::exists(helper)) env.insert("IPDE_MEDIA_BRIDGE", QDir::cleanPath(helper)); process_->setProcessEnvironment(env);
        updateEnabled(); statusBar()->showMessage(job + "…"); process_->start();
    }
    void inspect(const QString &path) {
        preview_->reset("Inspecting photograph…"); source_->setText(path); assets_->clear(); loaded_ = false; capabilities_ = {}; inspectReport_ = {}; replacementsMap_.clear(); if (replacements_) replacements_->setText("No replacement planes selected.");
        run({photo ? "photo-inspect" : "raw-inspect", path, "--preview-dir", previews_->path()}, "Inspect photograph");
    }
    QStringList selectedArgs(const QString &operation) const {
        QStringList args;
        if (auto *item = assets_->currentItem()) args << "--asset" << item->data(0, Qt::UserRole).toString();
        if (operation == "cutout") for (int i = 0; i < assets_->topLevelItemCount(); ++i) { auto *item = assets_->topLevelItem(i); if (item->checkState(0) == Qt::Checked) args << "--matte" << item->data(0, Qt::UserRole).toString(); }
        if (operation == "learned-depth") { args << "--model" << model_->currentData().toString() << "--device" << device_->currentText(); if (!modelPath_->text().trimmed().isEmpty()) args << "--model-path" << modelPath_->text().trimmed(); if (!modelRoot_->text().trimmed().isEmpty()) args << "--model-source-dir" << modelRoot_->text().trimmed(); }
        if (operation == "raft-depth") {
            args << "--device" << device_->currentText(); if (!raftPath_->text().trimmed().isEmpty()) args << "--raft-model" << raftPath_->text().trimmed();
            QSettings settings(QDir(IPDE::projectRoot()).filePath("project.ini"), QSettings::IniFormat);
            if (!settings.value("raft/root").toString().isEmpty()) args << "--raft-root" << settings.value("raft/root").toString();
            if (!settings.value("raft/member").toString().isEmpty()) args << "--raft-model-member" << settings.value("raft/member").toString();
        }
        if (!photo && !crop_->text().trimmed().isEmpty()) args << "--crop" << crop_->text().trimmed();
        return args;
    }
    void exportProduct(bool copy, bool preview) {
        if (busy_ || !loaded_) return;
        const QString op = preview ? "cutout" : operation_->currentData().toString();
        QStringList args{photo ? "photo-export" : "raw-export", source_->text(), "--output-dir", preview ? previews_->path() : output_->text(), "--operation", op, "--format", preview ? "png" : format_->currentData().toString()}; args += selectedArgs(op);
        copyAfter_ = copy; previewAfter_ = preview; run(args, preview ? "Preview composite" : "Export product");
    }
    void chooseReplacement() {
        auto *item = assets_->currentItem(); if (!item) return;
        const QString path = QFileDialog::getOpenFileName(this, "Choose replacement array for " + item->text(0), output_->text(), "NumPy arrays (*.npy)"); if (path.isEmpty()) return;
        replacementsMap_[item->data(0, Qt::UserRole).toString()] = path; privacy_->setChecked(false); QStringList rows; for (auto it = replacementsMap_.cbegin(); it != replacementsMap_.cend(); ++it) rows << "Plane " + it.key() + " ← " + it.value(); replacements_->setText(rows.join('\n')); updateEnabled();
    }
    void rewriteHeic() {
        QStringList args{"photo-rewrite", source_->text(), "--output-dir", output_->text()}; if (privacy_->isChecked()) args << "--privacy";
        args << "--replacement-kind" << replacementKind_->currentData().toString();
        for (auto it = replacementsMap_.cbegin(); it != replacementsMap_.cend(); ++it) args << "--replacement" << it.key() + "=" + it.value(); copyAfter_ = previewAfter_ = false; run(args, "Save HEIC");
    }
    void finished(int code, QProcess::ExitStatus status) {
        stdout_ += process_->readAllStandardOutput(); stderr_ += process_->readAllStandardError(); busy_ = false;
        const auto result = QJsonDocument::fromJson(stdout_.trimmed()).object();
        if (cancelled_) statusBar()->showMessage("Cancelled.");
        else if (code != 0 || status != QProcess::NormalExit || !result.value("ok").toBool()) {
            const auto error = QJsonDocument::fromJson(stderr_.trimmed()).object().value("error").toString();
            log_->appendPlainText(error.isEmpty() ? QString::fromUtf8(stderr_.isEmpty() ? stdout_ : stderr_).left(6000) : error); statusBar()->showMessage(job_ + " failed. See details below.");
        } else if (job_ == "Inspect photograph") {
            inspectReport_ = result; capabilities_ = result.value("capabilities").toObject(); loaded_ = true;
            for (const auto &entry : result.value("assets").toArray()) {
                const auto asset = entry.toObject(); auto *item = new QTreeWidgetItem(assets_);
                const QString name = asset.value("name").toString(asset.value("kind").toString()); QString label = name; label.replace('_', ' '); label.replace('-', ' '); if (!label.isEmpty()) label[0] = label[0].toUpper(); label.replace("Hdr", "HDR"); label.replace("Rgb", "RGB"); label.replace("Iso", "ISO"); item->setText(0, label); item->setData(0, Qt::UserRole, asset.value("index").toInt()); item->setData(0, Qt::UserRole + 1, asset.value("preview_path").toString());
                item->setData(0, Qt::UserRole + 2, asset.value("replaceable").toBool(asset.value("kind").toString() == "native_depth"));
                const auto grid = asset.value("shape").toArray(); QString dimensions;
                if (grid.size() >= 2) dimensions = QString::number(grid[1].toInt()) + " × " + QString::number(grid[0].toInt());
                if (grid.size() == 3) dimensions += " · " + QString::number(grid[2].toInt()) + "ch";
                item->setText(1, asset.value("dtype").toString() + "  " + dimensions);
                const bool matte = asset.value("person_matte").toBool(); if (matte) { item->setFlags(item->flags() | Qt::ItemIsUserCheckable); item->setCheckState(0, Qt::Checked); }
                item->setToolTip(0, QJsonDocument(asset).toJson(QJsonDocument::Indented));
            }
            const auto path = result.value("preview_path").toString();
            if (assets_->topLevelItemCount()) {
                auto *initialItem = assets_->topLevelItem(0);
                for (int i = 0; i < assets_->topLevelItemCount(); ++i) if (!path.isEmpty() && assets_->topLevelItem(i)->data(0, Qt::UserRole + 1).toString() == path) initialItem = assets_->topLevelItem(i);
                assets_->setCurrentItem(initialItem);
            }
            if (!path.isEmpty()) preview_->load(path);
            log_->clear(); showWarnings(result); statusBar()->showMessage("Photograph inspected. Choose a product and review its layers.");
        } else {
            QList<QUrl> urls; QString png; for (const auto &entry : result.value("outputs").toArray()) { const auto path = entry.toObject().value("path").toString(); if (!path.isEmpty()) { urls << QUrl::fromLocalFile(path); if (path.endsWith(".png", Qt::CaseInsensitive)) png = path; log_->appendPlainText(path); } }
            if (previewAfter_ && !png.isEmpty()) { preview_->load(png); previewNote_->setText("Composite preview from checked layers. Change the selection and preview again to remove an inaccurate matte."); }
            if (copyAfter_ && !urls.isEmpty()) { auto *mime = new QMimeData; mime->setUrls(urls); if (!png.isEmpty()) { QFile file(png); if (file.open(QIODevice::ReadOnly)) mime->setData("image/png", file.readAll()); } QApplication::clipboard()->setMimeData(mime); }
            showWarnings(result); statusBar()->showMessage(copyAfter_ ? "Exported and copied to the clipboard." : job_ + " complete.");
        }
        copyAfter_ = previewAfter_ = false; updateEnabled();
    }
    void showWarnings(const QJsonObject &result) { for (const auto &warning : result.value("warnings").toArray()) log_->appendPlainText(warning.toString()); }
    QLabel *purpose_ = nullptr, *previewNote_ = nullptr, *matteHelp_ = nullptr, *replacements_ = nullptr;
    QLineEdit *source_, *output_, *modelPath_, *modelRoot_, *raftPath_ = nullptr, *crop_;
    QComboBox *operation_, *format_, *model_, *device_, *replacementKind_ = nullptr, *purposeChoice_ = nullptr; QTreeWidget *assets_; Preview *preview_;
    QPushButton *import_, *export_, *clipboard_, *composite_, *cancel_, *replace_ = nullptr, *saveHeic_ = nullptr; QCheckBox *privacy_ = nullptr;
    QPlainTextEdit *log_; QProcess *process_; QFormLayout *optionsForm_; std::unique_ptr<QTemporaryDir> previews_;
    QMap<QString, QString> replacementsMap_; QJsonObject capabilities_, inspectReport_; QByteArray stdout_, stderr_; QString job_;
    bool busy_ = false, loaded_ = false, copyAfter_ = false, previewAfter_ = false, cancelled_ = false;
};
}

int main(int argc, char **argv) {
    QApplication app(argc, argv); app.setApplicationName(appName()); app.setOrganizationName("IPDE"); app.setWindowIcon(IPDE::appIcon(role()));
    IPDE::ProjectSession session(role(), &app); if (!session.start()) return 1;
    MediaWindow window;
    session.setChangedHandler(&window, [&window] { window.reloadProjectSettings(); });
    session.setDisconnectedHandler(&window, [&window] { window.statusBar()->showMessage("Studio disconnected. Reopen the project hub to launch other apps."); });
    QObject::connect(&app, &QCoreApplication::aboutToQuit, &session, [&session] { session.shutdown(); });
    window.show(); const auto args = app.arguments(); const int capture = args.indexOf("--screenshot");
    if (capture >= 0 && capture + 1 < args.size()) {
        auto *timer = new QTimer(&window); QObject::connect(timer, &QTimer::timeout, &window, [&app, &window, timer, args, capture] { if (window.busy()) return; timer->stop(); app.exit(window.grab().save(args[capture + 1]) ? 0 : 2); }); timer->start(350);
    } else if (args.contains("--smoke-test")) QTimer::singleShot(300, &app, &QCoreApplication::quit);
    return app.exec();
}
