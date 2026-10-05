#include "project_session.h"
#include "studio_icons.h"
#include "window_layout.h"
#include "help_support.h"
#include <QAction>
#include <QApplication>
#include <QCheckBox>
#include <QCloseEvent>
#include <QComboBox>
#include <QDesktopServices>
#include <QDialog>
#include <QDialogButtonBox>
#include <QDirIterator>
#include <QFile>
#include <QFileDialog>
#include <QFileSystemWatcher>
#include <QFont>
#include <QFormLayout>
#include <QGridLayout>
#include <QGroupBox>
#include <QJsonArray>
#include <QLabel>
#include <QLineEdit>
#include <QListWidget>
#include <QLocalServer>
#include <QMainWindow>
#include <QMenu>
#include <QMessageBox>
#include <QPlainTextEdit>
#include <QProcess>
#include <QProcessEnvironment>
#include <QPushButton>
#include <QSettings>
#include <QSet>
#include <QSignalBlocker>
#include <QSplitter>
#include <QSpinBox>
#include <QStandardPaths>
#include <QSystemTrayIcon>
#include <QTimer>
#include <QThread>
#include <QUrl>
#include <QUuid>
#include <QVBoxLayout>
#include <map>

namespace {
const QList<QPair<QString, QString>> appRoles{
    {"Extract maps", "extractor"}, {"Datasets", "datasets"}, {"Trainer", "trainer"},
    {"Photo Studio", "photo"}, {"Raw Studio", "raw"}
};

void populatePurposes(QComboBox *combo) {
    combo->addItem("Effects / displacement maps", "effect/map");
    combo->addItem("Depth estimation", "depth-estimation");
    combo->addItem("Portraits / photo effects", "photo-effects");
    combo->addItem("Custom / manual", "manual");
}
QString purposeDescription(const QString &goal) {
    if (goal == "effect/map") return "Detail first: RAFT stereo displacement, native-resolution teacher labels, and optional custom training. A generic RAFT model is a useful starting point.";
    if (goal == "depth-estimation") return "Distance first: calibrated geometry and depth in meters where supported. DepthPro supplies estimated meter labels; measured references and held-out captures help assess physical accuracy.";
    if (goal == "photo-effects") return "Portrait first: embedded Apple depth and person mattes for masking and photo effects. These images usually lack the calibrated stereo pair required for RAFT training.";
    return "Choose individual products, models, devices and training settings. Start by identifying the source image, coordinate grid, units and output precision your workflow needs.";
}
QString documentsFolder() {
    const QString path = QStandardPaths::writableLocation(QStandardPaths::DocumentsLocation);
    return path.isEmpty() ? QDir::homePath() : path;
}
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

struct Hub {
    QString project, token;
    std::unique_ptr<QLockFile> lock;
    QLocalServer *server = nullptr;
    QFileSystemWatcher *watcher = nullptr;
    QTimer *debounce = nullptr;
    QMap<QString, QLocalSocket *> clients;
    QMap<QString, QProcess *> processes;
    QMap<QString, QJsonObject> pendingActivations;
};
struct ProjectStats {
    int datasets = 0, linked = 0, readyDatasets = 0, samples = 0, models = 0;
    QSet<QString> sourcePhotos;
    QStringList warnings;
    bool selectedCustomModel = false;
};

class StudioWindow final : public QMainWindow {
public:
    StudioWindow() {
        setWindowTitle("IPDE Studio — Projects"); setWindowIcon(IPDE::appIcon("studio"));
        IPDE::installHelpMenu(this, "IPDE Studio", "studio");
        auto *central = new QWidget(this); auto *layout = new QVBoxLayout(central);
        layout->setContentsMargins(24, 24, 24, 24); layout->setSpacing(12);
        auto *title = new QLabel("IPDE Studio", central);
        QFont font = title->font(); font.setPointSize(font.pointSize() + 10); font.setBold(true); title->setFont(font);
        layout->addWidget(title);
        auto *intro = new QLabel("Create or open a project to share its purpose, datasets and model choices across the apps.", central);
        intro->setWordWrap(true); layout->addWidget(intro);
        auto *splitter = new QSplitter(Qt::Horizontal, central);
        auto *projectLibrary = new QWidget(splitter); auto *libraryLayout = new QVBoxLayout(projectLibrary);
        libraryLayout->setContentsMargins(0, 0, 12, 0);
        libraryLayout->addWidget(new QLabel("Projects", projectLibrary));
        projects_ = new QListWidget(projectLibrary); projects_->setMinimumWidth(220); libraryLayout->addWidget(projects_, 1);
        auto *create = new QPushButton("New project…", projectLibrary); auto *open = new QPushButton("Open project…", projectLibrary);
        create->setIcon(IPDE::appIcon("studio")); libraryLayout->addWidget(create); libraryLayout->addWidget(open);
        auto *projectArea = new QWidget(splitter); auto *areaLayout = new QVBoxLayout(projectArea); areaLayout->setContentsMargins(0, 0, 0, 0);
        empty_ = new QLabel("Start a project\n\nChoose a name, a purpose and a location in one place. Your Documents folder is the default home for new projects.", projectArea);
        empty_->setWordWrap(true); empty_->setAlignment(Qt::AlignCenter); areaLayout->addWidget(empty_, 1);
        projectPane_ = new QWidget(projectArea); auto *projectLayout = new QVBoxLayout(projectPane_); projectLayout->setContentsMargins(0, 0, 0, 0);
        projectTitle_ = new QLabel(projectPane_); QFont projectFont = projectTitle_->font(); projectFont.setPointSize(projectFont.pointSize() + 4); projectFont.setBold(true); projectTitle_->setFont(projectFont);
        projectLayout->addWidget(projectTitle_);
        auto *goalRow = new QHBoxLayout; goalRow->addWidget(new QLabel("Main purpose", projectPane_)); goal_ = new QComboBox(projectPane_); populatePurposes(goal_); goalRow->addWidget(goal_, 1);
        auto *folder = new QPushButton("Show folder", projectPane_); goalRow->addWidget(folder); projectLayout->addLayout(goalRow);
        description_ = new QLabel(projectPane_); description_->setWordWrap(true); projectLayout->addWidget(description_);
        summary_ = new QLabel(projectPane_); summary_->setWordWrap(true); projectLayout->addWidget(summary_);
        alerts_ = new QLabel(projectPane_); alerts_->setWordWrap(true); alerts_->setTextFormat(Qt::PlainText); alerts_->setStyleSheet("color: #a96912;"); projectLayout->addWidget(alerts_);
        auto *guide = new QGroupBox("Suggested next step", projectPane_); auto *guideLayout = new QVBoxLayout(guide);
        guidance_ = new QLabel(guide); guidance_->setWordWrap(true); guidance_->setTextFormat(Qt::PlainText); guideLayout->addWidget(guidance_);
        next_ = new QPushButton(guide); guideLayout->addWidget(next_); projectLayout->addWidget(guide);
        auto *toolsHeader = new QHBoxLayout; toolsHeader->addWidget(new QLabel("Project apps", projectPane_)); toolsHeader->addStretch(); allTools_ = new QCheckBox("Show all apps", projectPane_); toolsHeader->addWidget(allTools_); projectLayout->addLayout(toolsHeader);
        appLayout_ = new QGridLayout; auto *apps = appLayout_;
        int position = 0;
        for (const auto &entry : appRoles) {
            auto *button = new QPushButton(entry.first, projectPane_); button->setIcon(IPDE::appIcon(entry.second)); button->setIconSize(QSize(34, 34)); button->setMinimumHeight(48);
            appButtons_[entry.second] = button; apps->addWidget(button, position / 2, position % 2); ++position;
            connect(button, &QPushButton::clicked, this, [this, role=entry.second] { launch(role); });
        }
        appButtons_["extractor"]->setToolTip("Extract original auxiliary arrays or estimate stereo products. NPY and lossless TIFF retain numerical precision; display previews are separate.");
        appButtons_["datasets"]->setToolTip("Build and review labels, organize independent scenes, and reuse linked datasets without copying their files.");
        appButtons_["trainer"]->setToolTip("Train from prepared datasets, manage model runs, and compare with the generic baseline before choosing a model.");
        appButtons_["photo"]->setToolTip("Preview portrait depth and person mattes, select mask layers, and export image cutouts and depth products.");
        appButtons_["raw"]->setToolTip("Inspect decoded sensor data, precision and metadata, then preserve arrays for a depth workflow.");
        projectLayout->addLayout(apps);
        auto *manageDatasets = new QPushButton("Manage datasets and links in Dataset Studio", projectPane_); projectLayout->addWidget(manageDatasets);
        auto *performance = new QFormLayout;
        workers_ = new QSpinBox(projectPane_); workers_->setRange(0, 1024); workers_->setSpecialValueText(QString("Automatic (%1 cores)").arg(qMax(1, QThread::idealThreadCount())));
        workers_->setToolTip("Parallel file import, array preparation and run initialization. Zero uses available processor cores. Teacher inference and model training use the selected device, including MPS on Apple Silicon.");
        performance->addRow("File processing threads", workers_); projectLayout->addLayout(performance);
        projectLayout->addStretch(); areaLayout->addWidget(projectPane_, 1); projectPane_->hide();
        splitter->setStretchFactor(1, 1); splitter->setSizes({270, 730}); layout->addWidget(splitter, 1);
        status_ = new QLabel("Ready", central); status_->setWordWrap(true); layout->addWidget(status_);
        log_ = new QPlainTextEdit(central); log_->setReadOnly(true); log_->setMaximumHeight(80); log_->setMaximumBlockCount(300); layout->addWidget(log_); log_->hide();
        IPDE::setScrollableCentralWidget(this, central, QSize(1080, 780));
        connect(create, &QPushButton::clicked, this, [this] { newProject(); });
        connect(open, &QPushButton::clicked, this, [this] { openProject(); });
        connect(folder, &QPushButton::clicked, this, [this] { QDesktopServices::openUrl(QUrl::fromLocalFile(selectedProject())); });
        connect(projects_, &QListWidget::currentItemChanged, this, [this] { loadProject(); });
        connect(goal_, &QComboBox::currentIndexChanged, this, [this] {
            const QString project = selectedProject(); if (project.isEmpty() || loading_) return;
            QSettings settings(QDir(project).filePath("project.ini"), QSettings::IniFormat); settings.setValue("goal", goal_->currentData()); settings.sync();
            if (settings.status() != QSettings::NoError) status_->setText("Could not save the project purpose. Check folder permissions.");
            refreshProject();
        });
        connect(allTools_, &QCheckBox::toggled, this, [this] { refreshProject(); });
        connect(next_, &QPushButton::clicked, this, [this] { launch(nextRole_); });
        connect(manageDatasets, &QPushButton::clicked, this, [this] { launch("datasets"); });
        connect(workers_, &QSpinBox::valueChanged, this, [this](int value) {
            if (loading_ || selectedProject().isEmpty()) return;
            QSettings settings(QDir(selectedProject()).filePath("project.ini"), QSettings::IniFormat); settings.setValue("performance/workers", value); settings.sync();
            if (settings.status() != QSettings::NoError) status_->setText("Could not save the thread override. Check folder permissions.");
        });
        setupTray();
        QSettings recent("IPDE", "Studio");
        if (!QCoreApplication::arguments().contains("--smoke-test"))
            for (const auto &path : recent.value("projects").toStringList()) if (QFileInfo(path).isDir()) appendProject(path);
        const QString argumentProject = IPDE::projectRoot();
        if (!argumentProject.isEmpty()) addProject(argumentProject);
        else if (projects_->count()) {
            int row = 0; const QString last = recent.value("last_project").toString();
            for (int i=0; i<projects_->count(); ++i) if (projects_->item(i)->data(Qt::UserRole).toString() == last) row = i;
            projects_->setCurrentRow(row);
        }
        updateTray();
    }
    ~StudioWindow() override {
        // Hub data is destroyed before QMainWindow's child QObjects. Disconnect
        // their callbacks while hubs still exist, including socket destruction.
        for (QObject *object : findChildren<QObject *>()) object->disconnect(this);
        if (tray_) tray_->hide();
    }
protected:
    void closeEvent(QCloseEvent *event) override {
        if (tray_ && tray_->isVisible()) { event->ignore(); hide(); return; }
        if (hasActiveApps()) { event->ignore(); status_->setText("Close the project apps before closing Studio so their shared session remains available."); }
        else event->accept();
    }
#ifdef IPDE_STUDIO_REGRESSION
public:
#else
private:
#endif
    bool hasActiveApps() const {
        for (const auto &entry : hubs_) {
            if (!entry.second->clients.isEmpty()) return true;
            for (auto *process : entry.second->processes) if (process->state() != QProcess::NotRunning) return true;
        }
        return false;
    }
    QString selectedProject() const { return projects_->currentItem() ? projects_->currentItem()->data(Qt::UserRole).toString() : QString(); }
    void showStudio() { showNormal(); raise(); activateWindow(); }
    void appendProject(const QString &project) {
        QSettings settings(QDir(project).filePath("project.ini"), QSettings::IniFormat);
        auto *item = new QListWidgetItem(settings.value("name", QFileInfo(project).fileName()).toString(), projects_); item->setIcon(IPDE::appIcon("studio")); item->setToolTip(project); item->setData(Qt::UserRole, project);
    }
    void persistRecent() {
        if (QCoreApplication::arguments().contains("--smoke-test")) return;
        QStringList paths; for (int i=0; i<projects_->count(); ++i) paths << projects_->item(i)->data(Qt::UserRole).toString();
        QSettings recent("IPDE", "Studio"); recent.setValue("projects", paths); recent.setValue("last_project", selectedProject());
    }
    bool addProject(const QString &path, const QString &name = {}, const QString &goal = {}) {
        const QString project = QFileInfo(path).canonicalFilePath(); if (project.isEmpty()) return false;
        if (!QDir(project).mkpath("workspace/datasets") || !QDir(project).mkpath("workspace/runs")) { status_->setText("Could not create the project workspace. Check folder permissions."); return false; }
        QSettings settings(QDir(project).filePath("project.ini"), QSettings::IniFormat);
        if (!settings.contains("goal") || !goal.isEmpty()) settings.setValue("goal", goal.isEmpty() ? "effect/map" : goal);
        if (!settings.contains("name") || !name.isEmpty()) settings.setValue("name", name.isEmpty() ? QFileInfo(project).fileName() : name);
        settings.setValue("schema", "ipde-project-v1"); settings.sync();
        if (settings.status() != QSettings::NoError) { status_->setText("Could not save project.ini. Check folder permissions."); return false; }
        int row = -1; for (int i=0; i<projects_->count(); ++i) if (projects_->item(i)->data(Qt::UserRole).toString() == project) row = i;
        if (row < 0) { appendProject(project); row = projects_->count() - 1; }
        projects_->setCurrentRow(row); persistRecent(); updateTray(); return true;
    }
    void newProject() {
        QDialog dialog(this); dialog.setWindowTitle("New IPDE project"); dialog.resize(580, 330);
        auto *layout = new QVBoxLayout(&dialog); auto *intro = new QLabel("Give your project a home and a purpose. You can change its purpose later.", &dialog); intro->setWordWrap(true); layout->addWidget(intro);
        auto *form = new QFormLayout; auto *name = new QLineEdit(&dialog); name->setPlaceholderText("My displacement project"); name->setObjectName("newProjectName"); form->addRow("Project name", name);
        auto *purpose = new QComboBox(&dialog); populatePurposes(purpose); purpose->setObjectName("newProjectPurpose"); form->addRow("Main purpose", purpose);
        auto *location = new QLineEdit(documentsFolder(), &dialog); location->setObjectName("newProjectLocation"); auto *locationRow = new QHBoxLayout; locationRow->addWidget(location); auto *browse = new QPushButton("Choose…", &dialog); locationRow->addWidget(browse); form->addRow("Save in", locationRow); layout->addLayout(form);
        auto *hint = new QLabel(purposeDescription(purpose->currentData().toString()), &dialog); hint->setWordWrap(true); layout->addWidget(hint);
        auto *error = new QLabel(&dialog); error->setWordWrap(true); error->setStyleSheet("color: #b14537;"); layout->addWidget(error);
        auto *buttons = new QDialogButtonBox(QDialogButtonBox::Cancel | QDialogButtonBox::Ok, &dialog); buttons->button(QDialogButtonBox::Ok)->setText("Create project"); layout->addWidget(buttons);
        connect(browse, &QPushButton::clicked, &dialog, [&] { const QString path = QFileDialog::getExistingDirectory(&dialog, "Project parent folder", location->text()); if (!path.isEmpty()) location->setText(path); });
        connect(purpose, &QComboBox::currentIndexChanged, &dialog, [&] { hint->setText(purposeDescription(purpose->currentData().toString())); });
        connect(buttons, &QDialogButtonBox::rejected, &dialog, &QDialog::reject);
        connect(buttons, &QDialogButtonBox::accepted, &dialog, [&] {
            const QString projectName = name->text().trimmed(), parent = QDir::cleanPath(location->text().trimmed());
            if (projectName.isEmpty() || projectName.contains('/') || projectName.contains('\\') || projectName == "." || projectName == "..") { error->setText("Enter a project name without path separators."); name->setFocus(); return; }
            if (!QDir::isAbsolutePath(parent) || !QFileInfo(parent).isDir()) { error->setText("Choose an existing parent folder using its full path."); return; }
            const QString path = QDir(parent).filePath(projectName);
            if (QFileInfo::exists(path)) { error->setText("That folder already exists. Choose another name or open it as a project."); return; }
            if (!QDir().mkpath(path)) { error->setText("Could not create this folder. Check that the location is writable."); return; }
            if (!addProject(path, projectName, purpose->currentData().toString())) { error->setText(status_->text()); return; }
            dialog.accept();
        });
        name->setFocus(); dialog.exec(); showStudio();
    }
    void openProject() {
        const QString path = QFileDialog::getExistingDirectory(this, "Open an IPDE project folder", selectedProject().isEmpty() ? documentsFolder() : QFileInfo(selectedProject()).absolutePath());
        if (!path.isEmpty()) { addProject(path); showStudio(); }
    }
    void loadProject() {
        const QString project = selectedProject(); projectPane_->setVisible(!project.isEmpty()); empty_->setVisible(project.isEmpty()); if (project.isEmpty()) return;
        loading_ = true; QSettings settings(QDir(project).filePath("project.ini"), QSettings::IniFormat);
        const int index = goal_->findData(settings.value("goal", "effect/map").toString()); goal_->setCurrentIndex(qMax(0, index)); workers_->setValue(settings.value("performance/workers", 0).toInt()); loading_ = false;
        projectTitle_->setText(settings.value("name", QFileInfo(project).fileName()).toString()); setWindowTitle("IPDE Studio — " + projectTitle_->text());
        hubFor(project); persistRecent(); refreshProject(); updateStatus();
    }
    ProjectStats projectStats() const {
        ProjectStats stats; const QString project = selectedProject(); const QStringList own = localDatasets(project);
        for (const QString &path : projectDatasets(project)) {
            const QJsonObject manifest = readJson(QDir(path).filePath("dataset.json"));
            if (manifest.value("schema").toString() != "ipde-depth-dataset-v1") { stats.warnings << "Dataset unavailable or unreadable: " + path; continue; }
            ++stats.datasets; if (!own.contains(path)) ++stats.linked;
            const auto samples = manifest.value("samples").toArray(); stats.samples += samples.size();
            for (const auto &entry : samples) { const auto sample = entry.toObject(); const QString identity = sample.value("source_sha256").toString(sample.value("source_path").toString()); if (!identity.isEmpty()) stats.sourcePhotos.insert(identity); }
            if (manifest.value("generation_state").toString("complete") != "complete") stats.warnings << QFileInfo(path).fileName() + ": dataset generation is incomplete. Finish or review it before training.";
            else if (samples.isEmpty()) stats.warnings << QFileInfo(path).fileName() + ": dataset has no samples yet.";
            else ++stats.readyDatasets;
        }
        const QString model = QSettings(QDir(project).filePath("project.ini"), QSettings::IniFormat).value("raft/model").toString();
        QDirIterator reports(QDir(project).filePath("workspace/runs"), {"*.pth.json"}, QDir::Files, QDirIterator::Subdirectories);
        while (reports.hasNext()) {
            const QString report = reports.next(), checkpoint = report.left(report.size() - 5);
            if (!QFileInfo(checkpoint).isFile()) continue;
            const auto metadata = readJson(report); if (!QStringList{"ipde-raft-training-report-v1", "ipde-raft-training-report-v2", "ipde-display-training-report-v1"}.contains(metadata.value("schema").toString())) continue;
            ++stats.models; if (!model.isEmpty() && QFileInfo(checkpoint).canonicalFilePath() == QFileInfo(model).canonicalFilePath()) stats.selectedCustomModel = true;
        }
        if (!model.isEmpty() && !QFileInfo::exists(model)) stats.warnings << "The selected RAFT checkpoint is unavailable: " + model;
        if (QFileInfo(model).isFile() && QStringList{"ipde-raft-training-report-v1", "ipde-raft-training-report-v2", "ipde-display-training-report-v1"}.contains(readJson(model + ".json").value("schema").toString())) stats.selectedCustomModel = true;
        return stats;
    }
    bool relevantRole(const QString &role) const {
        const QString goal = goal_->currentData().toString();
        if (goal == "manual" || allTools_->isChecked()) return true;
        if (goal == "photo-effects") return role == "photo" || role == "extractor";
        if (goal == "depth-estimation") return role != "photo";
        return role == "datasets" || role == "trainer" || role == "extractor";
    }
    void refreshProject() {
        if (selectedProject().isEmpty()) return;
        const QString goal = goal_->currentData().toString(); const ProjectStats stats = projectStats();
        description_->setText(purposeDescription(goal));
        summary_->setText(QString("%1 datasets (%2 linked)  ·  %3 samples  ·  %4 source photos  ·  %5 trained models").arg(stats.datasets).arg(stats.linked).arg(stats.samples).arg(stats.sourcePhotos.size()).arg(stats.models));
        QStringList visibleWarnings = stats.warnings.mid(0, 3);
        if (stats.warnings.size() > 3) visibleWarnings << QString("%1 more alerts — hover to read all.").arg(stats.warnings.size() - 3);
        alerts_->setText(visibleWarnings.join('\n')); alerts_->setToolTip(stats.warnings.join('\n')); alerts_->setVisible(!stats.warnings.isEmpty());
        if (goal == "effect/map") {
            if (stats.selectedCustomModel) { nextRole_ = "extractor"; next_->setText("Extract displacement with your selected model"); guidance_->setText("Your project has a selected trained checkpoint. Extract depth/displacement maps now, and return to Datasets or Trainer to add captures, refine labels or compare a new model. Keep a generic baseline comparison for unfamiliar scenes."); }
            else if (stats.models) { nextRole_ = "trainer"; next_->setText("Compare and choose a trained model"); guidance_->setText("Training has produced a checkpoint. Compare it with the generic RAFT baseline on held-out captures, then explicitly choose it for this project. You can also extract using the generic model now."); }
            else if (stats.readyDatasets) { nextRole_ = "trainer"; next_->setText("Train from a prepared dataset"); guidance_->setText("Review labels and prepare validation groups in Dataset Studio, then train and compare a model in Trainer. Existing datasets with a suitable split can be trained directly. A generic RAFT model remains available for extraction."); }
            else if (stats.datasets) { nextRole_ = "datasets"; next_->setText("Finish and review your datasets"); guidance_->setText("Your datasets are empty or still being generated. Open Datasets to finish generation and review labels before assembly or training. Extract maps is available with a generic RAFT model in the meantime."); }
            else { nextRole_ = "datasets"; next_->setText("Build your first dataset"); guidance_->setText("Start with a varied collection of spatial photos, or link datasets from another project below. Generate and review teacher depth, then train and compare a custom RAFT model. Extract maps is available now with a generic RAFT model."); }
        } else if (goal == "depth-estimation") {
            nextRole_ = "extractor"; next_->setText("Inspect and extract calibrated depth"); guidance_->setText(stats.datasets ? "Extract calibrated depth and inspect unsupported values. Your datasets can support further model comparisons or training; review scene grouping and use independently measured references when physical accuracy matters." : "Start by inspecting a spatial capture and its calibration. Extract depth in meters where the camera geometry supports it. For sensor RAW or a single photo, use Raw Studio and a teacher estimate, then review it against measured references.");
        } else if (goal == "photo-effects") {
            nextRole_ = "photo"; next_->setText("Open a portrait in Photo Studio"); guidance_->setText("Preview embedded depth and person mattes, remove bad mask layers, and export the full subject or individual regions such as hair and skin. Embedded depth can be resized as a derived approximation; original arrays remain available in Extract maps. RAFT training requires calibrated stereo pairs.");
        } else {
            nextRole_ = "extractor"; next_->setText("Inspect source data and choose outputs"); guidance_->setText("Inspect a source image first and choose the exact arrays you need. Photo Studio handles portrait masks; Raw Studio preserves sensor samples; Datasets and Trainer support reviewed stereo learning. Keep outputs, coordinate registration and units explicit as you shape the workflow.");
        }
        next_->setIcon(IPDE::appIcon(nextRole_)); next_->setIconSize(QSize(24, 24));
        for (auto *button : appButtons_) appLayout_->removeWidget(button);
        const QStringList order = goal == "photo-effects" ? QStringList{"photo", "extractor", "raw", "datasets", "trainer"}
            : goal == "depth-estimation" ? QStringList{"extractor", "raw", "datasets", "trainer", "photo"}
            : goal == "effect/map" ? QStringList{"datasets", "trainer", "extractor", "photo", "raw"}
            : QStringList{"extractor", "photo", "raw", "datasets", "trainer"};
        int position = 0;
        for (const QString &role : order) {
            auto *button = appButtons_.value(role); const bool visible = relevantRole(role); button->setVisible(visible);
            if (visible) { appLayout_->addWidget(button, position / 2, position % 2); ++position; }
        }
        const auto owner = hubs_.find(selectedProject()); const bool available = owner != hubs_.end(); goal_->setEnabled(available); next_->setEnabled(available);
        for (auto *button : appButtons_) button->setEnabled(available);
        updateTray();
    }
    void setupTray() {
        if (QCoreApplication::arguments().contains("--smoke-test") || !QSystemTrayIcon::isSystemTrayAvailable()) return;
        tray_ = new QSystemTrayIcon(IPDE::menuBarIcon(), this); tray_->setToolTip("IPDE Studio"); trayMenu_ = new QMenu(this); tray_->setContextMenu(trayMenu_);
        connect(tray_, &QSystemTrayIcon::activated, this, [this](QSystemTrayIcon::ActivationReason reason) { if (reason == QSystemTrayIcon::Trigger || reason == QSystemTrayIcon::DoubleClick) showStudio(); });
        tray_->show(); QApplication::setQuitOnLastWindowClosed(false);
    }
    void updateTray() {
        if (!trayMenu_) return;
        for (QMenu *menu : trayMenu_->findChildren<QMenu *>(QString(), Qt::FindDirectChildrenOnly)) menu->deleteLater();
        trayMenu_->clear();
        connect(trayMenu_->addAction(IPDE::appIcon("studio"), "Show IPDE Studio"), &QAction::triggered, this, [this] { showStudio(); });
        auto *projects = trayMenu_->addMenu("Change project");
        for (int i=0; i<projects_->count(); ++i) {
            auto *item = projects_->item(i); auto *action = projects->addAction(item->text()); action->setCheckable(true); action->setChecked(item == projects_->currentItem());
            connect(action, &QAction::triggered, this, [this, i] { projects_->setCurrentRow(i); showStudio(); });
        }
        connect(projects->addAction("New project…"), &QAction::triggered, this, [this] { showStudio(); newProject(); });
        connect(projects->addAction("Open project…"), &QAction::triggered, this, [this] { showStudio(); openProject(); });
        trayMenu_->addSeparator();
        auto *launches = trayMenu_->addMenu(selectedProject().isEmpty() ? "Project apps" : "Launch for " + projectTitle_->text());
        for (const auto &entry : appRoles) {
            auto *action = launches->addAction(IPDE::appIcon(entry.second), entry.first); action->setEnabled(!selectedProject().isEmpty() && hubs_.find(selectedProject()) != hubs_.end());
            connect(action, &QAction::triggered, this, [this, role=entry.second] { launch(role); });
        }
        trayMenu_->addSeparator(); connect(trayMenu_->addAction("Quit IPDE Studio…"), &QAction::triggered, this, [this] {
            if (hasActiveApps() && QMessageBox::question(this, "Quit IPDE Studio", "Project apps need Studio to keep their session available. Quit Studio and close all project apps?", QMessageBox::Cancel | QMessageBox::Yes, QMessageBox::Cancel) != QMessageBox::Yes) return;
            for (const auto &entry : hubs_) for (auto *process : entry.second->processes) if (process->state() != QProcess::NotRunning) process->terminate();
            QCoreApplication::quit();
        });
        tray_->setToolTip(selectedProject().isEmpty() ? "IPDE Studio" : "IPDE Studio — " + projectTitle_->text());
    }
    Hub *hubFor(const QString &project) {
        const auto existing = hubs_.find(project); if (existing != hubs_.end()) return existing->second.get();
        auto hub = std::make_unique<Hub>(); hub->project = project; hub->token = QUuid::createUuid().toString(QUuid::WithoutBraces);
        QDir(project).mkpath(".studio"); hub->lock = std::make_unique<QLockFile>(QDir(project).filePath(".studio/hub.lock")); hub->lock->setStaleLockTime(0);
        if (!hub->lock->tryLock(0)) { status_->setText("This project is managed by another IPDE Studio window. Use its existing window to launch project apps."); return nullptr; }
        hub->server = new QLocalServer(this); const QString name = IPDE::serverName(project); QLocalServer::removeServer(name); hub->server->setSocketOptions(QLocalServer::UserAccessOption);
        if (!hub->server->listen(name)) { status_->setText("Could not create project IPC: " + hub->server->errorString()); hub->server->deleteLater(); return nullptr; }
        Hub *pointer = hub.get();
        connect(hub->server, &QLocalServer::newConnection, this, [this, pointer] {
            while (pointer->server->hasPendingConnections()) {
                auto *socket = pointer->server->nextPendingConnection(); socket->setParent(pointer->server); auto buffer = std::make_shared<QByteArray>();
                connect(socket, &QLocalSocket::readyRead, this, [this, pointer, socket, buffer] {
                    *buffer += socket->readAll(); if (buffer->size() > 65536) { socket->disconnectFromServer(); return; }
                    while (buffer->contains('\n')) {
                        const int end = buffer->indexOf('\n'); const auto message = QJsonDocument::fromJson(buffer->left(end)).object(); buffer->remove(0, end + 1); const QString role = message.value("role").toString();
                        const QString registeredRole = socket->property("role").toString();
                        if (!registeredRole.isEmpty() && pointer->clients.value(registeredRole) == socket) {
                            if (message.value("command") == "open_app") launchFor(pointer, role, message);
                            else if (message.value("command") == "busy") {
                                socket->setProperty("operation", message.value("operation").toString()); broadcastBusy(pointer);
                            }
                            continue;
                        }
                        const bool accepted = message.value("command") == "register" && message.value("token") == pointer->token && QStringList{"extractor", "datasets", "trainer", "photo", "raw"}.contains(role) && !pointer->clients.contains(role);
                        socket->write(QJsonDocument(QJsonObject{{"accepted", accepted}, {"error", accepted ? "" : "An app is already registered, or the project session is invalid."}}).toJson(QJsonDocument::Compact) + '\n'); socket->flush();
                        if (!accepted) { socket->disconnectFromServer(); return; }
                        pointer->clients[role] = socket; socket->setProperty("role", role); updateStatus();
                        if (pointer->pendingActivations.contains(role)) socket->write(QJsonDocument(pointer->pendingActivations.take(role)).toJson(QJsonDocument::Compact) + '\n');
                        broadcastBusy(pointer);
                    }
                });
                connect(socket, &QLocalSocket::disconnected, this, [this, pointer, socket] {
                    const QString role = socket->property("role").toString(); if (pointer->clients.value(role) == socket) pointer->clients.remove(role); socket->deleteLater(); broadcastBusy(pointer); updateStatus();
                });
            }
        });
        hub->watcher = new QFileSystemWatcher(this); hub->debounce = new QTimer(this); hub->debounce->setSingleShot(true); hub->debounce->setInterval(300);
        connect(hub->watcher, &QFileSystemWatcher::directoryChanged, hub->debounce, qOverload<>(&QTimer::start));
        connect(hub->watcher, &QFileSystemWatcher::fileChanged, hub->debounce, qOverload<>(&QTimer::start));
        connect(hub->debounce, &QTimer::timeout, this, [this, pointer] {
            watchProject(pointer); for (auto *client : pointer->clients) client->write("{\"event\":\"project_changed\"}\n");
            if (selectedProject() == pointer->project) loadProject();
        });
        watchProject(pointer); hubs_[project] = std::move(hub); return pointer;
    }
    void broadcastBusy(Hub *hub) {
        QJsonObject operations;
        for (auto iterator = hub->clients.cbegin(); iterator != hub->clients.cend(); ++iterator) {
            const QString operation = iterator.value()->property("operation").toString();
            if (!operation.isEmpty()) operations.insert(iterator.key(), operation);
        }
        const auto event = QJsonDocument(QJsonObject{{"event", "project_busy"}, {"operations", operations}}).toJson(QJsonDocument::Compact) + '\n';
        for (auto *client : hub->clients) { client->write(event); client->flush(); }
    }
    void watchProject(Hub *hub) {
        QStringList paths{hub->project, QDir(hub->project).filePath("project.ini"), QDir(hub->project).filePath("workspace"), QDir(hub->project).filePath("workspace/datasets"), QDir(hub->project).filePath("workspace/runs")};
        for (const QString &container : {"datasets", "runs"}) {
            QDir dir(QDir(hub->project).filePath("workspace/" + container)); for (const auto &child : dir.entryInfoList(QDir::Dirs | QDir::NoDotAndDotDot)) paths << child.absoluteFilePath();
        }
        for (const QString &dataset : projectDatasets(hub->project)) { paths << dataset; paths << QDir(dataset).filePath("dataset.json"); }
        const QStringList currently = hub->watcher->directories() + hub->watcher->files();
        QStringList removed; for (const auto &path : currently) if (!paths.contains(path)) removed << path; if (!removed.isEmpty()) hub->watcher->removePaths(removed);
        for (const auto &path : paths) if (QFileInfo::exists(path) && !currently.contains(path)) hub->watcher->addPath(path);
    }
    QString executable(const QString &role) const {
        const QString app = role == "extractor" ? "IPDE" : role == "datasets" ? "Dataset Studio" : role == "photo" ? "Photo Studio" : role == "raw" ? "Raw Studio" : "RAFT Studio";
        const QString bundle = app + ".app/Contents/MacOS/" + app; const QDir here(QCoreApplication::applicationDirPath());
        for (const QString &relative : {"../Applications/" + bundle, "../../../" + bundle}) { const QString path = QDir::cleanPath(here.filePath(relative)); if (QFileInfo::exists(path)) return path; }
        return QDir::cleanPath(here.filePath("../../../" + bundle));
    }
    void launch(const QString &role) {
        const QString project = selectedProject(); if (project.isEmpty()) { status_->setText("Create or open a project first."); showStudio(); return; }
        Hub *hub = hubFor(project); if (!hub) { showStudio(); return; }
        launchFor(hub, role);
    }
    void launchFor(Hub *hub, const QString &role, QJsonObject request = {}) {
        if (!QStringList{"extractor", "datasets", "trainer", "photo", "raw"}.contains(role)) return;
        request.remove("command"); request.remove("role"); request.insert("event", "activate");
        if (hub->clients.contains(role)) {
            hub->clients[role]->write(QJsonDocument(request).toJson(QJsonDocument::Compact) + '\n'); hub->clients[role]->flush(); return;
        }
        hub->pendingActivations[role] = request;
        if (hub->processes.contains(role) && hub->processes[role]->state() != QProcess::NotRunning) return;
        const QString project = hub->project;
        const QString path = executable(role); if (!QFileInfo::exists(path)) { status_->setText("The app is missing from Studio: " + path); showStudio(); return; }
        auto *process = new QProcess(this); process->setProgram(path); process->setArguments({"--project", project, "--mode", role});
        auto environment = QProcessEnvironment::systemEnvironment(); environment.insert("IPDE_STUDIO_TOKEN", hub->token); process->setProcessEnvironment(environment); hub->processes[role] = process;
        connect(process, &QProcess::readyReadStandardError, this, [this, process] { const QString output = QString::fromUtf8(process->readAllStandardError()).trimmed(); if (!output.isEmpty()) { log_->show(); log_->appendPlainText(output); } });
        connect(process, &QProcess::errorOccurred, this, [this, hub, role, process](QProcess::ProcessError error) {
            status_->setText("App launch failed: " + process->errorString()); showStudio();
            if (error == QProcess::FailedToStart) { hub->processes.remove(role); hub->pendingActivations.remove(role); process->deleteLater(); }
        });
        connect(process, qOverload<int, QProcess::ExitStatus>(&QProcess::finished), this, [this, hub, role, process](int code, QProcess::ExitStatus exitStatus) {
            if (hub->processes.value(role) == process) hub->processes.remove(role); hub->pendingActivations.remove(role); process->deleteLater();
            log_->appendPlainText(role + " closed (" + QString::number(code) + ")."); if (code || exitStatus == QProcess::CrashExit) { log_->show(); showStudio(); }
            updateStatus(); refreshProject();
            if (!isVisible() && !tray_ && !hasActiveApps()) QCoreApplication::quit();
        });
        process->start(); status_->setText("Opening " + role + " for " + QFileInfo(project).fileName() + "…");
    }
    void updateStatus() {
        QStringList active; for (const auto &entry : hubs_) for (const auto &role : entry.second->clients.keys()) active << QFileInfo(entry.first).fileName() + ": " + role;
        if (!selectedProject().isEmpty() && hubs_.find(selectedProject()) == hubs_.end()) return;
        status_->setText(active.isEmpty() ? (tray_ ? "Ready · Closing this window keeps Studio available in the menu bar." : "Ready") : "Open apps — " + active.join(" · "));
    }
    QListWidget *projects_ = nullptr; QComboBox *goal_ = nullptr; QWidget *projectPane_ = nullptr; QCheckBox *allTools_ = nullptr; QSpinBox *workers_ = nullptr;
    QLabel *empty_ = nullptr, *projectTitle_ = nullptr, *description_ = nullptr, *summary_ = nullptr, *alerts_ = nullptr, *guidance_ = nullptr, *status_ = nullptr;
    QPushButton *next_ = nullptr; QPlainTextEdit *log_ = nullptr; QGridLayout *appLayout_ = nullptr; QMap<QString, QPushButton *> appButtons_; QString nextRole_;
    QSystemTrayIcon *tray_ = nullptr; QMenu *trayMenu_ = nullptr; bool loading_ = false;
    std::map<QString, std::unique_ptr<Hub>> hubs_;
};
}

int main(int argc, char **argv) {
    QApplication application(argc, argv); application.setApplicationName("IPDE Studio"); application.setOrganizationName("IPDE"); application.setWindowIcon(IPDE::appIcon("studio"));
    StudioWindow window; window.show();
    const auto arguments = application.arguments(); const int screenshot = arguments.indexOf("--screenshot");
    if (screenshot >= 0 && screenshot + 1 < arguments.size()) QTimer::singleShot(150, &window, [&window, &application, path=arguments[screenshot + 1]] { application.exit(window.grab().save(path) ? 0 : 2); });
    if (arguments.contains("--smoke-test")) QTimer::singleShot(300, &application, &QCoreApplication::quit);
    return application.exec();
}
