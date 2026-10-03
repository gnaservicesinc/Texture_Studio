#include <QApplication>
#include <QComboBox>
#include <QCheckBox>
#include <QDateTime>
#include <QDesktopServices>
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
#include <QPlainTextEdit>
#include <QProcess>
#include <QProgressBar>
#include <QPushButton>
#include <QRegularExpression>
#include <QSettings>
#include <QSpinBox>
#include <QSplitter>
#include <QStandardPaths>
#include <QStatusBar>
#include <QStyledItemDelegate>
#include <QTabWidget>
#include <QTemporaryFile>
#include <QTimer>
#include <QTreeWidget>
#include <QUrl>
#include <QVBoxLayout>

#include <memory>

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

class TrainerWindow final : public QMainWindow {
public:
    TrainerWindow() : settings_("IPDE", "RAFTStudio") {
        setWindowTitle("RAFT Studio — Spatial-photo datasets and training");
        resize(1280, 900);
        auto *central = new QWidget(this);
        auto *root = new QVBoxLayout(central);
        root->setContentsMargins(18, 16, 18, 16);
        auto *title = new QLabel("RAFT Studio", central);
        QFont font = title->font(); font.setPointSize(font.pointSize() + 7); font.setBold(true); title->setFont(font);
        root->addWidget(title);
        auto *purpose = new QLabel("Create depth-teacher datasets from spatial HEICs, train RAFT-Stereo, and export a model for IPDE.", central);
        purpose->setWordWrap(true); root->addWidget(purpose);

        auto *workspaceRow = new QHBoxLayout;
        workspace_ = new QLineEdit(settings_.value("workspace", "/opt/ipde/raft-workspace").toString(), central);
        workspaceRow->addWidget(new QLabel("Workspace", central)); workspaceRow->addWidget(workspace_, 1);
        chooseWorkspace_ = new QPushButton("Choose…", central);
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

        auto *library = new QSplitter(Qt::Horizontal, central);
        auto *datasetBox = new QGroupBox("Datasets", library);
        auto *datasetLayout = new QVBoxLayout(datasetBox);
        datasets_ = new QTreeWidget(datasetBox);
        datasets_->setHeaderLabels({"Dataset", "Images", "Train / validation", "Teacher"});
        datasets_->header()->setSectionResizeMode(0, QHeaderView::Stretch);
        datasets_->setColumnWidth(1, 65); datasets_->setColumnWidth(2, 140); datasets_->setColumnWidth(3, 185);
        datasets_->setRootIsDecorated(false); datasets_->setAlternatingRowColors(true);
        datasetLayout->addWidget(datasets_);
        auto *datasetButtons = new QHBoxLayout;
        auto *inspect = new QPushButton("Inspect selected", datasetBox);
        auto *showDataset = new QPushButton("Show files", datasetBox);
        auto *archive = new QPushButton("Archive selected", datasetBox);
        datasetButtons->addWidget(inspect); datasetButtons->addWidget(showDataset); datasetButtons->addWidget(archive); datasetButtons->addStretch();
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
        runButtons->addWidget(export_); runButtons->addWidget(showRun); runButtons->addStretch();
        runLayout->addLayout(runButtons);
        library->addWidget(datasetBox); library->addWidget(runBox);
        root->addWidget(library, 2);
        connect(inspect, &QPushButton::clicked, this, [this] {
            const QString path = selectedPath(datasets_);
            if (!path.isEmpty()) startJob("Inspect dataset", {"inspect-dataset", path});
        });
        connect(showDataset, &QPushButton::clicked, this, [this] { showFolder(selectedPath(datasets_)); });
        connect(archive, &QPushButton::clicked, this, [this] { archiveDataset(); });
        connect(showRun, &QPushButton::clicked, this, [this] {
            const QString path = selectedPath(runs_); if (!path.isEmpty()) showFolder(QFileInfo(path).absolutePath());
        });
        connect(export_, &QPushButton::clicked, this, [this] { exportModel(); });

        tabs_ = new QTabWidget(central);
        buildDatasetTab(); buildTrainingTab();
        root->addWidget(tabs_, 3);
        auto *notice = new QLabel("Teacher depth is an estimate, not measured ground truth. Capture groups are held out by default; scene validation requires correctly grouped independent scenes.", central);
        notice->setWordWrap(true); root->addWidget(notice);
        log_ = new QPlainTextEdit(central); log_->setReadOnly(true); log_->setMaximumBlockCount(1500);
        log_->setMaximumHeight(135); log_->setPlaceholderText("Progress and results appear here."); root->addWidget(log_);
        auto *progressRow = new QHBoxLayout;
        progress_ = new QProgressBar(central); progress_->setRange(0, 1); progress_->setValue(0); progress_->setTextVisible(false);
        cancel_ = new QPushButton("Cancel current task", central); cancel_->setEnabled(false);
        progressRow->addWidget(progress_, 1); progressRow->addWidget(cancel_); root->addLayout(progressRow);
        setCentralWidget(central);

        process_ = new QProcess(this); process_->setProcessChannelMode(QProcess::SeparateChannels);
        connect(process_, &QProcess::readyReadStandardOutput, this, [this] { stdout_ += process_->readAllStandardOutput(); });
        connect(process_, &QProcess::readyReadStandardError, this, [this] { appendProgress(process_->readAllStandardError()); });
        connect(process_, qOverload<int, QProcess::ExitStatus>(&QProcess::finished), this,
            [this](int code, QProcess::ExitStatus status) { processFinished(code, status); });
        connect(process_, &QProcess::errorOccurred, this, [this](QProcess::ProcessError error) {
            if (error == QProcess::FailedToStart) {
                log_->appendPlainText("Could not launch Python: " + process_->errorString());
                setBusy(false); groupsFile_.reset();
            }
        });
        connect(cancel_, &QPushButton::clicked, this, [this] {
            cancelled_ = true; process_->kill(); log_->appendPlainText("Cancellation requested; partial work will not be presented as complete.");
        });
        statusBar()->showMessage("Ready");
        if (!QCoreApplication::arguments().contains("--smoke-test"))
            QTimer::singleShot(0, this, [this] { refreshLibrary(); });
    }

    ~TrainerWindow() override {
        settings_.setValue("workspace", workspace_->text());
        settings_.setValue("teacher_model", teacher_->currentData());
        settings_.setValue("raft_root", raftRoot_->text()); settings_.setValue("raft_model", raftModel_->text());
        if (process_->state() != QProcess::NotRunning) process_->kill();
    }

private:
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
            if (!value.isEmpty()) edit->setText(value);
        });
        return row;
    }

    void buildDatasetTab() {
        auto *tab = new QWidget; auto *root = new QVBoxLayout(tab);
        auto *row = new QHBoxLayout;
        datasetName_ = new QLineEdit("dataset-" + QDateTime::currentDateTime().toString("yyyyMMdd-HHmmss"), tab);
        row->addWidget(new QLabel("New dataset", tab)); row->addWidget(datasetName_, 1);
        auto *add = new QPushButton("Add spatial HEICs…", tab); auto *remove = new QPushButton("Remove selected", tab);
        row->addWidget(add); row->addWidget(remove); root->addLayout(row);
        sources_ = new QTreeWidget(tab); sources_->setHeaderLabels({"Spatial HEIC", "Scene group — edit related captures to match"});
        sources_->setSelectionMode(QAbstractItemView::ExtendedSelection); sources_->setRootIsDecorated(false);
        sources_->header()->setSectionResizeMode(0, QHeaderView::Stretch); sources_->header()->setSectionResizeMode(1, QHeaderView::Stretch);
        sources_->setItemDelegate(new GroupDelegate(sources_)); root->addWidget(sources_, 1);
        verifiedScenes_ = new QCheckBox("I verified these group IDs describe independent scenes (hold out scenes instead of capture groups)", tab);
        verifiedScenes_->setToolTip("Only enable this when each group contains all related photos of one scene and distinct groups really are independent scenes. Nearby captures otherwise receive conservative grouping.");
        root->addWidget(verifiedScenes_);
        connect(add, &QPushButton::clicked, this, [this] {
            const auto files = QFileDialog::getOpenFileNames(this, "Add spatial HEIC photos", {}, "HEIC / HEIF photos (*.heic *.HEIC *.heif *.HEIF)");
            for (const QString &file : files) {
                const QString path = QFileInfo(file).absoluteFilePath(); bool exists = false;
                for (int i = 0; i < sources_->topLevelItemCount(); ++i) exists |= sources_->topLevelItem(i)->data(0, Qt::UserRole).toString() == path;
                if (exists) continue;
                auto *item = new QTreeWidgetItem(sources_, {QFileInfo(path).fileName(), QFileInfo(path).completeBaseName()});
                item->setData(0, Qt::UserRole, path); item->setToolTip(0, path); item->setFlags(item->flags() | Qt::ItemIsEditable);
            }
        });
        connect(remove, &QPushButton::clicked, this, [this] { qDeleteAll(sources_->selectedItems()); });
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
        anchor_ = new QCheckBox("Anchor relative teacher scale to DepthPro (estimated meters)", tab);
        anchor_->setChecked(true);
        anchor_->setToolTip("Uses the separately installed Apple DepthPro model to estimate metric scale. Detail remains from the selected teacher. This adds inference and does not create measured ground truth.");
        left->addRow("Relative teacher scale", anchor_);
        controls->addLayout(left, 1); controls->addLayout(right, 1); root->addLayout(controls);
        connect(teacher_, &QComboBox::currentIndexChanged, this, [this] { teacherDefaults(); });
        teacher_->setCurrentIndex(qMax(0, teacher_->findData(settings_.value("teacher_model", "depthpro"))));
        teacherDefaults();
        generate_ = new QPushButton("Generate training dataset", tab); root->addWidget(generate_);
        connect(generate_, &QPushButton::clicked, this, [this] { generateDataset(); });
        tabs_->addTab(tab, "1. Generate dataset");
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
    }

    void buildTrainingTab() {
        auto *tab = new QWidget; auto *root = new QVBoxLayout(tab);
        auto *instruction = new QLabel("Select a dataset from the library above. Start with update-block fine-tuning; preserve the original RAFT checkpoint.", tab);
        instruction->setWordWrap(true); root->addWidget(instruction);
        auto *form = new QFormLayout;
        runName_ = new QLineEdit("run-" + QDateTime::currentDateTime().toString("yyyyMMdd-HHmmss"), tab); form->addRow("New run", runName_);
        form->addRow("RAFT source", pathRow(raftRoot_, settings_.value("raft_root", "/opt/ipde/RAFT-Stereo").toString(), true, "Choose RAFT-Stereo source", tab));
        form->addRow("Initial RAFT model", pathRow(raftModel_, settings_.value("raft_model", "/opt/ipde/models/raftstereo-middlebury.pth").toString(), false, "Choose original RAFT checkpoint", tab));
        root->addLayout(form);
        auto *options = new QHBoxLayout;
        epochs_ = spin(tab, 1, 10000, 10); steps_ = spin(tab, 1, 10000, 16); patch_ = spin(tab, 64, 2048, 256); patch_->setSingleStep(32);
        iterations_ = spin(tab, 1, 256, 4); scope_ = new QComboBox(tab); scope_->addItem("Update block", "update"); scope_->addItem("Full network", "full"); trainDevice_ = deviceBox(tab);
        for (auto pair : {qMakePair(QString("Epochs"), epochs_), qMakePair(QString("Steps / epoch"), steps_), qMakePair(QString("Patch pixels"), patch_), qMakePair(QString("RAFT iterations"), iterations_)}) {
            auto *group = new QFormLayout; group->addRow(pair.first, pair.second); options->addLayout(group);
        }
        auto *deviceForm = new QFormLayout; deviceForm->addRow("Train scope", scope_); deviceForm->addRow("Device", trainDevice_); options->addLayout(deviceForm);
        root->addLayout(options); root->addStretch();
        train_ = new QPushButton("Train selected dataset", tab); root->addWidget(train_);
        connect(train_, &QPushButton::clicked, this, [this] { trainDataset(); });
        tabs_->addTab(tab, "2. Train RAFT-Stereo");
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
            if (group.isEmpty()) { QMessageBox::information(this, "Scene groups", "Give every photo a scene group. Related captures should share the same group."); return; }
            const QString path = item->data(0, Qt::UserRole).toString(); args << path; groups.insert(path, group);
        }
        groupsFile_ = std::make_unique<QTemporaryFile>();
        if (!groupsFile_->open() || groupsFile_->write(QJsonDocument(groups).toJson()) < 0 || !groupsFile_->flush()) {
            log_->appendPlainText("Could not create the scene-group configuration."); groupsFile_.reset(); return;
        }
        args << "--output-dir" << QDir(workspace_->text()).filePath("datasets/" + name)
             << "--model" << teacher_->currentData().toString() << "--model-path" << teacherPath_->text()
             << "--source-dir" << teacherSource_->text() << "--device" << teacherDevice_->currentText()
             << "--input-size" << QString::number(inputSize_->value()) << "--groups" << groupsFile_->fileName();
        if (verifiedScenes_->isChecked()) args << "--grouping" << "scene";
        if (teacher_->currentData().toString() != "depthpro" && anchor_->isChecked()) args << "--metric-anchor" << "depthpro";
        refreshAfter_ = true; startJob("Generate dataset", args);
    }

    void trainDataset() {
        const QString dataset = selectedPath(datasets_); const QString name = runName_->text().trimmed();
        if (dataset.isEmpty()) { QMessageBox::information(this, "Choose dataset", "Select a dataset from the library above."); return; }
        if (!validName(name)) { QMessageBox::information(this, "Run name", "Use a run folder name without path separators."); return; }
        if (patch_->value() % 32) { QMessageBox::information(this, "Patch size", "RAFT patch size must be a multiple of 32 (for example 256 or 512)."); return; }
        const QString checkpoint = QDir(workspace_->text()).filePath("runs/" + name + "/checkpoint.pth");
        refreshAfter_ = true;
        startJob("Train RAFT-Stereo", {"train", dataset, "--checkpoint", checkpoint, "--raft-root", raftRoot_->text(), "--raft-model", raftModel_->text(),
            "--epochs", QString::number(epochs_->value()), "--steps", QString::number(steps_->value()), "--patch-size", QString::number(patch_->value()),
            "--iterations", QString::number(iterations_->value()), "--scope", scope_->currentData().toString(), "--device", trainDevice_->currentText()});
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
        startJob("Refresh library", {"workspace", workspace_->text()});
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
        refreshLibrary();
    }

    void startJob(const QString &label, const QStringList &arguments) {
        if (process_->state() != QProcess::NotRunning) return;
        if (!QFileInfo::exists(scriptPath())) { log_->appendPlainText("RAFT Studio script is missing: " + scriptPath()); groupsFile_.reset(); return; }
        job_ = label; stdout_.clear(); cancelled_ = false; setBusy(true);
        log_->appendPlainText(label + "…");
        process_->setProgram(pythonPath()); process_->setArguments(QStringList{scriptPath(), "--json"} + arguments);
        process_->start();
    }

    void appendProgress(const QByteArray &bytes) {
        QString text = QString::fromUtf8(bytes); text.remove(QRegularExpression("\\x1b\\[[0-9;]*m"));
        if (!text.trimmed().isEmpty()) log_->appendPlainText(text.trimmed());
    }

    void setBusy(bool busy) {
        tabs_->setEnabled(!busy); workspace_->setEnabled(!busy); chooseWorkspace_->setEnabled(!busy); refresh_->setEnabled(!busy); export_->setEnabled(!busy);
        cancel_->setEnabled(busy); progress_->setRange(0, busy ? 0 : 1); progress_->setValue(0);
        statusBar()->showMessage(busy ? job_ : "Ready");
    }

    void processFinished(int code, QProcess::ExitStatus status) {
        stdout_ += process_->readAllStandardOutput(); appendProgress(process_->readAllStandardError());
        setBusy(false); groupsFile_.reset();
        if (cancelled_) { refreshAfter_ = false; statusBar()->showMessage("Cancelled"); return; }
        QJsonParseError error; const QJsonDocument doc = QJsonDocument::fromJson(stdout_.trimmed(), &error);
        if (status != QProcess::NormalExit || code != 0 || !doc.isObject()) {
            log_->appendPlainText(job_ + " failed (exit " + QString::number(code) + ").");
            if (!stdout_.trimmed().isEmpty()) log_->appendPlainText(QString::fromUtf8(stdout_).left(12000));
            if (code == 0 && error.error != QJsonParseError::NoError) log_->appendPlainText("Could not parse the result: " + error.errorString());
            refreshAfter_ = false; return;
        }
        const QJsonObject result = doc.object();
        if (job_ == "Refresh library") populateLibrary(result);
        else log_->appendPlainText(QString::fromUtf8(QJsonDocument(result).toJson(QJsonDocument::Indented)).left(18000));
        statusBar()->showMessage(job_ + " complete");
        if (job_ == "Export RAFT model") {
            log_->appendPlainText("Export complete: " + exportDestination_ + "/raft-model.pth — choose this file in IPDE's RAFT model control.");
            showFolder(exportDestination_);
        }
        const bool refresh = refreshAfter_; refreshAfter_ = false;
        if (refresh) QTimer::singleShot(0, this, [this] { refreshLibrary(); });
    }

    void populateLibrary(const QJsonObject &result) {
        const QString oldDataset = selectedPath(datasets_), oldRun = selectedPath(runs_);
        datasets_->clear(); runs_->clear();
        for (const QJsonValue &value : result.value("datasets").toArray()) {
            const auto obj = value.toObject(); const QString path = obj.value("path").toString();
            QString teacher = obj.value("teacher").toString();
            if (teacher.isEmpty() && obj.value("teacher").isObject()) teacher = obj.value("teacher").toObject().value("model").toString();
            auto *item = new QTreeWidgetItem(datasets_, {obj.value("name").toString(QFileInfo(path).fileName()), QString::number(obj.value("sample_count").toInt()),
                QString("%1 / %2").arg(obj.value("train_count").toInt()).arg(obj.value("validation_count").toInt()), teacher});
            item->setData(0, Qt::UserRole, path); item->setToolTip(0, path);
            if (path == oldDataset) datasets_->setCurrentItem(item);
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
    }

    void showFolder(const QString &path) {
        if (!path.isEmpty()) QDesktopServices::openUrl(QUrl::fromLocalFile(path));
    }

    QSettings settings_;
    QLineEdit *workspace_ = nullptr, *datasetName_ = nullptr, *runName_ = nullptr;
    QLineEdit *teacherPath_ = nullptr, *teacherSource_ = nullptr, *raftRoot_ = nullptr, *raftModel_ = nullptr;
    QTreeWidget *datasets_ = nullptr, *runs_ = nullptr, *sources_ = nullptr;
    QComboBox *teacher_ = nullptr, *teacherDevice_ = nullptr, *trainDevice_ = nullptr, *scope_ = nullptr;
    QCheckBox *verifiedScenes_ = nullptr, *anchor_ = nullptr;
    QSpinBox *inputSize_ = nullptr, *epochs_ = nullptr, *steps_ = nullptr, *patch_ = nullptr, *iterations_ = nullptr;
    QTabWidget *tabs_ = nullptr; QPlainTextEdit *log_ = nullptr; QProgressBar *progress_ = nullptr;
    QPushButton *chooseWorkspace_ = nullptr, *refresh_ = nullptr, *generate_ = nullptr, *train_ = nullptr, *cancel_ = nullptr, *export_ = nullptr;
    QProcess *process_ = nullptr; QByteArray stdout_; QString job_, exportDestination_;
    bool cancelled_ = false, refreshAfter_ = false;
    std::unique_ptr<QTemporaryFile> groupsFile_;
};

} // namespace

int main(int argc, char **argv) {
    QApplication application(argc, argv);
    application.setApplicationName("RAFT Studio"); application.setOrganizationName("IPDE");
    TrainerWindow window; window.show();
    const QStringList args = application.arguments();
    if (args.contains("--smoke-test")) QTimer::singleShot(100, &application, &QCoreApplication::quit);
    const int screenshot = args.indexOf("--screenshot");
    if (screenshot >= 0 && screenshot + 1 < args.size()) {
        QTimer::singleShot(1000, &application, [&application, &window, args, screenshot] {
            const bool saved = window.grab().save(args.at(screenshot + 1));
            application.exit(saved ? 0 : 2);
        });
    }
    return application.exec();
}
