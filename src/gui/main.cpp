#include "project_session.h"
#include "studio_icons.h"
#include "window_layout.h"
#include "help_support.h"
#include "python_runtime.h"
#include <QApplication>
#include <QCheckBox>
#include <QComboBox>
#include <QCoreApplication>
#include <QDir>
#include <QDragEnterEvent>
#include <QDropEvent>
#include <QFileDialog>
#include <QFile>
#include <QFileInfo>
#include <QFont>
#include <QHBoxLayout>
#include <QHeaderView>
#include <QGroupBox>
#include <QSignalBlocker>
#include <QJsonArray>
#include <QJsonDocument>
#include <QJsonObject>
#include <QLabel>
#include <QLineEdit>
#include <QMainWindow>
#include <QMessageBox>
#include <QMimeData>
#include <QProcess>
#include <QProgressBar>
#include <QPushButton>
#include <QSet>
#include <QSettings>
#include <QScrollArea>
#include <QSpinBox>
#include <QMap>
#include <QStandardPaths>
#include <QStatusBar>
#include <QTextEdit>
#include <QTimer>
#include <QTreeWidget>
#include <QUrl>
#include <QVBoxLayout>
#include <QWidget>

#include <algorithm>

#ifndef IPDE_SOURCE_SCRIPT
#define IPDE_SOURCE_SCRIPT "ipde_extract.py"
#endif

#ifndef IPDE_PYTHON_EXECUTABLE
#define IPDE_PYTHON_EXECUTABLE "python3"
#endif

namespace {

QString dimensionText(const QJsonObject &asset) {
    if (asset.value("width").toInt() <= 0 || asset.value("height").toInt() <= 0)
        return QStringLiteral("Model prediction grid");
    QString value = QStringLiteral("%1 × %2")
                        .arg(asset.value(QStringLiteral("width")).toInt())
                        .arg(asset.value(QStringLiteral("height")).toInt());
    const int channels = asset.value(QStringLiteral("channels")).toInt(1);
    if (channels > 1) {
        value += QStringLiteral(" × %1 ch").arg(channels);
    }
    return value;
}

QString bundledScriptPath() {
#ifdef Q_OS_MACOS
    const QDir executableDir(QCoreApplication::applicationDirPath());
    const QString bundled = executableDir.absoluteFilePath(QStringLiteral("../Resources/ipde_extract.py"));
    if (QFileInfo::exists(bundled)) {
        return QDir::cleanPath(bundled);
    }
#endif
    return QString::fromUtf8(IPDE_SOURCE_SCRIPT);
}

QString configuredPython() {
    return IPDE::pythonExecutable();
}

bool individualDepthProduct(const QString &id) {
    return id == "learned-da3" || id == "learned-da2";
}

class MainWindow final : public QMainWindow {
public:
    MainWindow() {
        setWindowTitle(QStringLiteral("IPDE — Precision HEIF Auxiliary Extractor"));
        IPDE::installHelpMenu(this, "IPDE Extractor", "extractor");
        setAcceptDrops(true);

        auto *central = new QWidget(this);
        auto *root = new QVBoxLayout(central);
        root->setContentsMargins(20, 18, 20, 18);
        root->setSpacing(12);

        auto *title = new QLabel(QStringLiteral("Image Precision Data Extractor"), central);
        QFont titleFont = title->font();
        titleFont.setPointSize(titleFont.pointSize() + 6);
        titleFont.setBold(true);
        title->setFont(titleFont);
        root->addWidget(title);

        auto *subtitle = new QLabel(
            QStringLiteral("Extract decoded depth, gain maps, mattes, and alpha planes without normalization, tone mapping, or gamma conversion."),
            central);
        subtitle->setWordWrap(true);
        root->addWidget(subtitle);

        auto *goalRow = new QHBoxLayout;
        goalRow->addWidget(new QLabel("Main purpose", central));
        goal_ = new QComboBox(central);
        goal_->addItem("Effect / displacement map", "effect/map");
        goal_->addItem("Depth estimation", "depth-estimation");
        goal_->addItem("Photo effects / masking", "photo-effects");
        goal_->addItem("Manual", "manual");
        goalRow->addWidget(goal_, 1);
        advancedToggle_ = new QCheckBox("Advanced settings", central); goalRow->addWidget(advancedToggle_);
        root->addLayout(goalRow);
        auto *methodRow = new QHBoxLayout;
        methodRow->addWidget(new QLabel("Default depth source", central));
        depthMethod_ = new QComboBox(central); depthMethod_->setObjectName("depthMethod");
        depthMethod_->addItem("DepthPro", "depthpro");
        depthMethod_->addItem("DA3", "depth-anything-3");
        depthMethod_->addItem("DA2", "depth-anything-v2");
        depthMethod_->addItem("Selected RAFT model", "raft");
        methodRow->addWidget(depthMethod_, 1); root->addLayout(methodRow);
        goalHint_ = new QLabel(central); goalHint_->setWordWrap(true); root->addWidget(goalHint_);
        modelQuality_ = new QLabel(central); modelQuality_->setWordWrap(true);
        modelQuality_->setTextFormat(Qt::PlainText); modelQuality_->setObjectName("modelQuality");
        modelQuality_->hide(); root->addWidget(modelQuality_);

        files_ = new QTreeWidget(central);
        files_->setColumnCount(4);
        files_->setHeaderLabels({QStringLiteral("Source / output — check only what you want"), QStringLiteral("Status / dimensions"), QStringLiteral("Source precision"), QStringLiteral("Export storage")});
        files_->header()->setSectionResizeMode(0, QHeaderView::Stretch);
        files_->header()->setSectionResizeMode(1, QHeaderView::ResizeToContents);
        files_->header()->setSectionResizeMode(2, QHeaderView::ResizeToContents);
        files_->header()->setSectionResizeMode(3, QHeaderView::ResizeToContents);
        files_->setSelectionMode(QAbstractItemView::ExtendedSelection);
        files_->setAlternatingRowColors(true);
        root->addWidget(files_, 1);

        auto *fileButtons = new QHBoxLayout;
        auto *add = new QPushButton(QStringLiteral("Add HEIC files…"), central);
        remove_ = new QPushButton(QStringLiteral("Remove selected"), central);
        auto *clear = new QPushButton(QStringLiteral("Clear"), central);
        inspect_ = new QPushButton(QStringLiteral("Inspect"), central);
        fileButtons->addWidget(add);
        fileButtons->addWidget(remove_);
        fileButtons->addWidget(clear);
        fileButtons->addStretch();
        fileButtons->addWidget(inspect_);
        root->addLayout(fileButtons);

        auto *outputRow = new QHBoxLayout;
        outputRow->addWidget(new QLabel(QStringLiteral("Output folder:"), central));
        output_ = new QLineEdit(central);
        output_->setPlaceholderText(QStringLiteral("Leave blank to save beside each source image"));
        auto *browse = new QPushButton(QStringLiteral("Choose…"), central);
        outputRow->addWidget(output_, 1);
        outputRow->addWidget(browse);
        root->addLayout(outputRow);

        auto *options = new QHBoxLayout;
        exactNpy_ = new QCheckBox(QStringLiteral("Write exact .npy companions"), central);
        exactNpy_->setChecked(false);
        manifest_ = new QCheckBox(QStringLiteral("Write JSON manifest"), central);
        manifest_->setChecked(false);
        manifest_->setToolTip(QStringLiteral("Optional provenance report with calibration, hashes, and extraction details."));
        colorMatching_ = new QCheckBox(QStringLiteral("Color Matching"), central);
        colorMatching_->setChecked(false);
        colorMatching_->setToolTip(QStringLiteral(
            "Before inference only, histogram-match each RGB channel of the non-Hero stereo view to "
            "the Hero view. Raw extracted views remain untouched. This matches marginal code-value "
            "curves; it is not an ICC conversion or a guarantee of pixelwise color equality."));
        colorHero_ = new QComboBox(central);
        colorHero_->addItem(QStringLiteral("Hero: Left view"), QStringLiteral("left"));
        colorHero_->addItem(QStringLiteral("Hero: Right view"), QStringLiteral("right"));
        colorHero_->setEnabled(false);
        colorHero_->setToolTip(QStringLiteral(
            "The Hero view is preserved unchanged; the other view receives the recorded histogram LUTs."));
        raftDevice_ = new QComboBox(central);
        raftDevice_->addItem(QStringLiteral("RAFT device: Automatic"), QStringLiteral("auto"));
        raftDevice_->addItem(QStringLiteral("RAFT device: Apple Metal"), QStringLiteral("mps"));
        raftDevice_->addItem(QStringLiteral("RAFT device: CPU"), QStringLiteral("cpu"));
        raftDevice_->addItem(QStringLiteral("RAFT device: CUDA"), QStringLiteral("cuda"));
        raftDevice_->setToolTip(QStringLiteral(
            "Automatic prefers Apple Metal on this Mac and falls back to CPU only when Metal is unavailable."));
        overwrite_ = new QCheckBox(QStringLiteral("Replace existing outputs"), central);
        options->addWidget(exactNpy_);
        options->addWidget(manifest_);

        options->addWidget(overwrite_);
        options->addStretch();
        root->addLayout(options);

        advanced_ = new QWidget(central);
        auto *advancedLayout = new QVBoxLayout(advanced_); advancedLayout->setContentsMargins(0, 0, 0, 0);
        advancedScroll_ = new QScrollArea(central); advancedScroll_->setWidgetResizable(true);
        advancedScroll_->setFrameShape(QFrame::NoFrame); advancedScroll_->setMaximumHeight(240);
        advancedScroll_->setWidget(advanced_); root->addWidget(advancedScroll_);
        auto *learnedOptions = new QHBoxLayout;
        learnedDevice_ = new QComboBox(central);
        for (const auto &device : {QString("auto"), QString("mps"), QString("cpu"), QString("cuda")})
            learnedDevice_->addItem(device == "auto" ? "AI device: Automatic" : "AI device: " + device, device);
        learnedInputSize_ = new QSpinBox(central); learnedInputSize_->setRange(0, 4096); learnedInputSize_->setValue(0);
        learnedInputSize_->setKeyboardTracking(false);
        learnedInputSize_->setSpecialValueText("Full display image");
        learnedInputSize_->setToolTip("Process the full display image by default. An explicit size reduces model input resolution; DepthPro uses its fixed model grid.");
        learnedOptions->addWidget(learnedDevice_); learnedOptions->addWidget(new QLabel("AI processing size", central));
        learnedOptions->addWidget(learnedInputSize_); learnedOptions->addStretch(); advancedLayout->addLayout(learnedOptions);
        auto addLearnedPath = [this, central, advancedLayout](const QString &label, bool source) {
            auto *row = new QHBoxLayout; row->addWidget(new QLabel(label, central));
            auto *edit = new QLineEdit(central); edit->setPlaceholderText("Automatic local model lookup");
            auto *choose = new QPushButton("Choose…", central); row->addWidget(edit, 1); row->addWidget(choose);
            choose->setProperty("depthFamily", "ai");
            advancedLayout->addLayout(row);
            connect(edit, &QLineEdit::textChanged, this, [this, source](const QString &value) {
                setSharedValue("learned/" + depthMethod_->currentData().toString() + (source ? "/source" : "/model"), value);
            });
            connect(edit, &QLineEdit::editingFinished, this, [this] { modelSelectionChanged(); });
            connect(choose, &QPushButton::clicked, this, [this, edit, source] {
                if (running_) return;
                const bool directory = source || depthMethod_->currentData().toString() == "depth-anything-3";
                const QString path = directory ? QFileDialog::getExistingDirectory(this, source ? "Choose model source folder" : "Choose DA3 model directory", edit->text())
                    : QFileDialog::getOpenFileName(this, "Choose depth checkpoint", edit->text(), "Checkpoints (*.pt *.pth);;All files (*)");
                if (!path.isEmpty()) { edit->setText(path); modelSelectionChanged(); }
            });
            return edit;
        };
        learnedModel_ = addLearnedPath("AI checkpoint:", false); learnedSource_ = addLearnedPath("AI source folder:", true);
        auto *spatialOptions = new QHBoxLayout;
        spatialOptions->addWidget(new QLabel(QStringLiteral("Spatial Photo:"), central));
        spatialOptions->addWidget(colorMatching_);
        spatialOptions->addWidget(colorHero_);
        spatialOptions->addWidget(raftDevice_);
        spatialOptions->addStretch();
        advancedLayout->addLayout(spatialOptions);

        auto addPathRow = [this, advancedLayout, central](const QString &label, const QString &key,
                                                bool directory) {
            auto *row = new QHBoxLayout;
            row->addWidget(new QLabel(label, central));
            auto *edit = new QLineEdit(sharedValue(key).toString(), central);
            edit->setPlaceholderText(QStringLiteral("Automatic lookup (or choose a path)"));
            auto *choose = new QPushButton(QStringLiteral("Choose…"), central);
            choose->setProperty("depthFamily", "raft");
            row->addWidget(edit, 1);
            row->addWidget(choose);
            advancedLayout->addLayout(row);
            connect(edit, &QLineEdit::textChanged, this, [this, key](const QString &text) {
                setSharedValue(key, text);
            });
            connect(choose, &QPushButton::clicked, this, [this, edit, directory] {
                if (running_) return;
                const QString chosen = directory
                    ? QFileDialog::getExistingDirectory(this, QStringLiteral("Choose RAFT source folder"), edit->text())
                    : QFileDialog::getOpenFileName(this, QStringLiteral("Choose RAFT model"), edit->text(),
                          QStringLiteral("Model checkpoints (*.pth *.pt *.zip);;All files (*)"));
                if (!chosen.isEmpty()) {
                    edit->setText(chosen);
                    modelSelectionChanged();
                }
            });
            return edit;
        };
        raftModel_ = addPathRow(QStringLiteral("RAFT model:"), QStringLiteral("raft/model"), false);
        connect(raftModel_, &QLineEdit::editingFinished, this, [this] { modelSelectionChanged(); });
        raftRoot_ = addPathRow(QStringLiteral("RAFT source folder:"), QStringLiteral("raft/root"), true);
        connect(raftRoot_, &QLineEdit::editingFinished, this, [this] { modelSelectionChanged(); });
        auto *memberRow = new QHBoxLayout;
        memberRow->addWidget(new QLabel(QStringLiteral("Model inside ZIP:"), central));
        raftMember_ = new QLineEdit(sharedValue(QStringLiteral("raft/member")).toString(), central);
        raftMember_->setPlaceholderText(QStringLiteral("raftstereo-middlebury.pth (only for ZIP models)"));
        memberRow->addWidget(raftMember_);
        advancedLayout->addLayout(memberRow);
        connect(raftMember_, &QLineEdit::textChanged, this, [this](const QString &text) {
            setSharedValue(QStringLiteral("raft/member"), text);
        });
        connect(raftMember_, &QLineEdit::editingFinished, this, [this] { modelSelectionChanged(); });
        auto *help = new QLabel(QStringLiteral(
            "Check the sources to save, then Export checked. For DA3 or DA2, select one row and use Export this map."), central);
        help->setWordWrap(true);
        root->addWidget(help);


        auto *runRow = new QHBoxLayout;
        progress_ = new QProgressBar(central);
        progress_->setRange(0, 1);
        progress_->setValue(0);
        extract_ = new QPushButton(QStringLiteral("Export checked"), central);
        exportOne_ = new QPushButton(QStringLiteral("Export this map"), central);
        runRow->addWidget(exportOne_);
        cancel_ = new QPushButton(QStringLiteral("Cancel"), central);
        cancel_->setEnabled(false);
        runRow->addWidget(progress_, 1);
        runRow->addWidget(extract_);
        runRow->addWidget(cancel_);
        root->addLayout(runRow);

        log_ = new QTextEdit(central);
        log_->setReadOnly(true);
        log_->setMaximumHeight(150);
        log_->setPlaceholderText(QStringLiteral("Structured extraction results appear here."));
        root->addWidget(log_);

        IPDE::setScrollableCentralWidget(this, central, QSize(1300, 780));
        statusBar()->showMessage(QStringLiteral("Drop Apple HEIC portrait photos here, or choose Add HEIC files."));

        process_ = new QProcess(this);
        process_->setProcessChannelMode(QProcess::SeparateChannels);

        connect(add, &QPushButton::clicked, this, [this] {
            const QStringList paths = QFileDialog::getOpenFileNames(
                this,
                QStringLiteral("Select HEIF images"),
                QString(),
                QStringLiteral("HEIF images (*.heic *.HEIC *.heif *.HEIF *.hif *.HIF);;All files (*)"));
            addFiles(paths);
        });
        connect(remove_, &QPushButton::clicked, this, [this] {
            const auto selected = files_->selectedItems();
            QSet<QTreeWidgetItem *> roots;
            for (auto *item : selected) {
                while (item->parent()) item = item->parent();
                roots.insert(item);
            }
            for (auto *item : roots) {
                sources_.removeAll(item->data(0, Qt::UserRole).toString());
                delete item;
            }
            updateButtons();
        });
        connect(clear, &QPushButton::clicked, this, [this] {
            if (!running_) {
                sources_.clear();
                files_->clear();
                updateButtons();
            }
        });
        connect(browse, &QPushButton::clicked, this, [this] {
            if (running_) return;
            const QString chosen = QFileDialog::getExistingDirectory(this, QStringLiteral("Choose output folder"), output_->text());
            if (!chosen.isEmpty()) {
                output_->setText(chosen);
            }
        });
        connect(inspect_, &QPushButton::clicked, this, [this] { beginQueue(true); });
        connect(extract_, &QPushButton::clicked, this, [this] { beginQueue(false); });
        connect(exportOne_, &QPushButton::clicked, this, [this] {
            auto *item = files_->currentItem();
            if (!item || item->data(0, Qt::UserRole + 1).toString().isEmpty()) return;
            singleSource_ = item->parent()->data(0, Qt::UserRole).toString();
            singleProduct_ = item->data(0, Qt::UserRole + 1).toString();
            beginQueue(false);
            singleSource_.clear();
            singleProduct_.clear();
        });
        connect(files_, &QTreeWidget::itemSelectionChanged, this, [this] { updateButtons(); });
        connect(files_, &QTreeWidget::itemChanged, this, [this] { updateButtons(); });
        connect(colorMatching_, &QCheckBox::toggled, this, [this](bool checked) {
            colorHero_->setEnabled(checked && !running_);
        });
        connect(cancel_, &QPushButton::clicked, this, [this] {
            cancelled_ = true;
            queue_.clear();
            if (process_->state() != QProcess::NotRunning) {
                process_->kill();
            }
            log_->append(QStringLiteral("Cancelled by user."));
        });
        connect(process_, qOverload<int, QProcess::ExitStatus>(&QProcess::finished), this,
                [this](int exitCode, QProcess::ExitStatus exitStatus) { processFinished(exitCode, exitStatus); });
        connect(process_, &QProcess::errorOccurred, this, [this](QProcess::ProcessError error) {
            if (error == QProcess::FailedToStart) {
                const QString message = QStringLiteral("Could not start Python: %1").arg(process_->errorString());
                log_->append(message);
                if (auto *item = rootForPath(current_)) {
                    item->setText(1, QStringLiteral("Error"));
                    item->setToolTip(1, message);
                }
                ++completed_;
                progress_->setValue(completed_);
                startNext();
            }
        });

        connect(advancedToggle_, &QCheckBox::toggled, advancedScroll_, &QWidget::setVisible);
        connect(depthMethod_, &QComboBox::currentIndexChanged, this, [this] {
            setSharedValue("learned/method", depthMethod_->currentData()); loadLearnedSettings();
            applyGoal(false); modelSelectionChanged(); updateButtons();
        });
        connect(learnedDevice_, &QComboBox::currentIndexChanged, this, [this] {
            setSharedValue("learned/device", learnedDevice_->currentData()); modelSelectionChanged();
        });
        connect(learnedInputSize_, &QSpinBox::valueChanged, this, [this](int value) {
            setSharedValue("learned/input_size", value);
        });
        connect(learnedInputSize_, &QSpinBox::editingFinished, this, [this] { modelSelectionChanged(); });
        connect(goal_, &QComboBox::currentIndexChanged, this, [this] {
            setSharedValue("goal", goal_->currentData()); applyGoal();
        });
        reloadProjectSettings();
        inspect_->setIcon(IPDE::appIcon("extractor"));
        extract_->setIcon(IPDE::appIcon("extractor"));
        exportOne_->setIcon(IPDE::appIcon("extractor"));
        updateButtons();
    }

    ~MainWindow() override {
        // Child QProcess teardown can emit finished while our members and
        // widgets are already being destroyed. Stop its UI callbacks first.
        process_->disconnect(this);
        if (process_->state() != QProcess::NotRunning) {
            process_->kill();
            process_->waitForFinished(1500);
        }
    }

    void reloadProjectSettings() {
        if (running_) { reloadPending_ = true; return; }
        reloadPending_ = false;
        for (auto pair : {qMakePair(raftRoot_, QString("raft/root")), qMakePair(raftModel_, QString("raft/model")), qMakePair(raftMember_, QString("raft/member"))}) {
            const QSignalBlocker block(pair.first); pair.first->setText(sharedValue(pair.second).toString());
        }
        { const QSignalBlocker block(depthMethod_);
          depthMethod_->setCurrentIndex(qMax(0, depthMethod_->findData(sharedValue("learned/method", "depthpro")))); }
        loadLearnedSettings();
        const QSignalBlocker block(goal_);
        const int index = goal_->findData(sharedValue("goal", "depth-estimation").toString()); const bool changed = goal_->currentIndex() != qMax(0, index);
        goal_->setCurrentIndex(qMax(0, index)); if (changed || !settingsLoaded_) applyGoal(); settingsLoaded_ = true;
        if (!IPDE::projectRoot().isEmpty()) {
            if (output_->text().isEmpty()) output_->setText(QDir(IPDE::projectRoot()).filePath("exports"));
            setWindowTitle("IPDE Extractor — " + sharedValue("name", QFileInfo(IPDE::projectRoot()).fileName()).toString());
        }
        modelSelectionChanged();
        updateButtons();
    }

protected:
    void dragEnterEvent(QDragEnterEvent *event) override {
        if (event->mimeData()->hasUrls()) {
            event->acceptProposedAction();
        }
    }

    void dropEvent(QDropEvent *event) override {
        QStringList paths;
        for (const QUrl &url : event->mimeData()->urls()) {
            const QString path = url.toLocalFile();
            if (QFileInfo(path).isFile()) {
                paths << path;
            }
        }
        addFiles(paths);
        event->acceptProposedAction();
    }

#ifdef IPDE_EXTRACTOR_REGRESSION
public:
#else
private:
#endif
    QVariant sharedValue(const QString &key, const QVariant &fallback = {}) const {
        if (IPDE::projectRoot().isEmpty()) return QSettings().value(key, fallback);
        return QSettings(QDir(IPDE::projectRoot()).filePath("project.ini"), QSettings::IniFormat).value(key, fallback);
    }
    void setSharedValue(const QString &key, const QVariant &value) {
        if (IPDE::projectRoot().isEmpty()) { QSettings().setValue(key, value); return; }
        QSettings settings(QDir(IPDE::projectRoot()).filePath("project.ini"), QSettings::IniFormat); settings.setValue(key, value); settings.sync();
    }
    bool directDepthSelected() const { return depthMethod_->currentData().toString() != "raft"; }
    void loadLearnedSettings() {
        const QString key = "learned/" + depthMethod_->currentData().toString();
        for (auto pair : {qMakePair(learnedModel_, key + "/model"), qMakePair(learnedSource_, key + "/source")}) {
            const QSignalBlocker block(pair.first); pair.first->setText(sharedValue(pair.second).toString());
        }
        const QSignalBlocker deviceBlock(learnedDevice_), sizeBlock(learnedInputSize_);
        learnedDevice_->setCurrentIndex(qMax(0, learnedDevice_->findData(sharedValue("learned/device", "auto"))));
        if (!sharedValue("learned/full_display_sources", false).toBool()) {
            if (sharedValue("learned/input_size", 0).toInt() == 1036) setSharedValue("learned/input_size", 0);
            setSharedValue("learned/full_display_sources", true);
        }
        learnedInputSize_->setValue(sharedValue("learned/input_size", 0).toInt());
    }
    void applyGoal(bool updateSelections = true) {
        const QString goal = goal_->currentData().toString();
        if (updateSelections) {
            advancedToggle_->setChecked(goal == "manual"); advancedScroll_->setVisible(advancedToggle_->isChecked());
        }
        if (directDepthSelected() && (goal == "depth-estimation" || goal == "effect/map")) {
            goalHint_->setText("DepthPro saves full-resolution float32 depth. DA3 and DA2 are individual exports with a delay warning.");
        }
        else if (goal == "effect/map") goalHint_->setText(displayStudentSelected() ? "Suggested output: selected RAFT model height map on the display grid. The stereo pair produces one map; compare it with held-out reference depth." : "Suggested output: selected RAFT model displacement on the left stereo grid. All RAFT export choices remain available when you change models.");
        else if (goal == "depth-estimation") goalHint_->setText(displayStudentSelected() ? "Suggested output: selected RAFT model depth. Units follow its training labels; relative outputs are not meter distances. All RAFT export choices remain available." : "Suggested output: calibrated RAFT meter depth plus its support mask. Unknown or unsupported values need review.");
        else if (goal == "photo-effects") goalHint_->setText("Suggested outputs: embedded Apple depth and mattes from portrait photos. Original depth values are preserved.");
        else goalHint_->setText("Choose individual products and override any inference settings.");
        if (updateSelections && goal != "manual" && !running_) {
            const QSignalBlocker blocker(files_);
            for (int i = 0; i < files_->topLevelItemCount(); ++i) for (int j = 0; j < files_->topLevelItem(i)->childCount(); ++j) {
                auto *child = files_->topLevelItem(i)->child(j);
                child->setCheckState(0, suggestedProduct(child->data(0, Qt::UserRole + 1).toString(), child->text(0).toLower()) ? Qt::Checked : Qt::Unchecked);
            }
        }
    }
    bool suggestedProduct(const QString &id, const QString &kind) const {
        const QString goal = goal_->currentData().toString();
        if (individualDepthProduct(id)) return false;
        if (goal == "effect/map") return id == (directDepthSelected() ? "learned-depthpro" : "raft-displacement");
        if (goal == "depth-estimation") return directDepthSelected() ? id == "learned-depthpro"
            : id == "raft-depth" || (!displayStudentSelected() && id == "raft-support");
        if (goal == "photo-effects") return id.startsWith("raw:") && (kind.contains("depth") || kind.contains("matte"));
        return false;
    }
    bool displayStudentSelected() const {
        return !directDepthSelected() && selectedModelKind_ == "display_student" && resolvedModelSelection_ == modelSelectionKey();
    }
    QString selectedLearnedProduct() const {
        const auto model = depthMethod_->currentData().toString();
        return model == "depth-anything-3" ? "learned-da3" : model == "depth-anything-v2" ? "learned-da2" : "learned-depthpro";
    }
    QStringList modelSelectionKey() const {
        if (directDepthSelected()) return {depthMethod_->currentData().toString(), learnedModel_->text().trimmed(),
            learnedSource_->text().trimmed(), learnedDevice_->currentData().toString(), QString::number(learnedInputSize_->value())};
        return {QString("raft"), raftModel_->text().trimmed(), raftRoot_->text().trimmed(), raftMember_->text().trimmed()};
    }
    void modelSelectionChanged() {
        if (running_ || !raftModel_ || !raftRoot_ || !raftMember_) return;
        const auto selection = modelSelectionKey();
        if (selection == lastModelSelection_) return;
        lastModelSelection_ = selection;
        selectedModelKind_.clear();
        resolvedModelSelection_.clear();
        modelQuality_->clear(); modelQuality_->hide();
        applyGoal(false);
        if (!sources_.isEmpty()) beginQueue(true);
    }
    QString productForSelectedModel(const QString &id) const {
        if (directDepthSelected()) {
            if (id.startsWith("raft-") || id.startsWith("student-") || id == "learned-depth" || id.startsWith("learned-display-")
                || id == "learned-native" || id == "learned-preview" || id == "learned-displacement") return selectedLearnedProduct();
        } else {
            if (id.startsWith("learned-")) return "raft-depth";
        }
        if (id == "student-display-depth") return "raft-display-depth";
        if (id == "student-display-displacement") return "raft-displacement";
        if (id == "student-display-preview") return "raft-display-preview";
        return id;
    }
    void addFiles(const QStringList &paths) {
        if (running_) return;
        bool added = false;
        for (const QString &raw : paths) {
            const QString path = QFileInfo(raw).absoluteFilePath();
            if (sources_.contains(path)) {
                continue;
            }
            sources_ << path;
            auto *item = new QTreeWidgetItem(files_);
            item->setText(0, QFileInfo(path).fileName());
            item->setToolTip(0, path);
            item->setText(1, QStringLiteral("Pending inspection"));
            item->setData(0, Qt::UserRole, path);
            added = true;
        }
        updateButtons();
        if (added && !running_) {
            beginQueue(true);
        }
    }

    QTreeWidgetItem *rootForPath(const QString &path) const {
        for (int i = 0; i < files_->topLevelItemCount(); ++i) {
            auto *item = files_->topLevelItem(i);
            if (item->data(0, Qt::UserRole).toString() == path) {
                return item;
            }
        }
        return nullptr;
    }

    bool approveSelectedExports() {
        bool hasIndividualDepth = false;
        for (const auto &products : selectedProducts_) {
            for (const auto &id : products) hasIndividualDepth |= individualDepthProduct(id);
        }
        if (!hasIndividualDepth) return true;
        if (!individualDepthProduct(singleProduct_) || selectedProducts_.size() != 1
            || selectedProducts_.value(singleSource_) != QStringList{singleProduct_}) {
            QMessageBox message(QMessageBox::Information, "Individual depth export",
                "DA3 and DA2 are available as individual exports. Select one model row under a photo and use Export this map.",
                QMessageBox::Ok, this);
            message.setObjectName("individualDepthExportRequired");
            message.exec();
            return false;
        }
        const QString text = singleProduct_ == "learned-da3"
            ? "DA3 full-resolution processing took about 20 minutes for one 5712 × 4284 photo in testing. Export this one photo now?"
            : "DA2 full-resolution processing can take several minutes for one high-resolution photo. Export this one photo now?";
        QMessageBox warning(QMessageBox::Warning, "Slow full-resolution export", text,
            QMessageBox::Ok | QMessageBox::Cancel, this);
        warning.setObjectName("depthAnythingExportWarning");
        warning.button(QMessageBox::Ok)->setText("Continue export");
        warning.setDefaultButton(QMessageBox::Cancel);
        return warning.exec() == QMessageBox::Ok;
    }

    void beginQueue(bool inspectOnly) {
        if (running_ || sources_.isEmpty()) {
            return;
        }
        const QString python = configuredPython();
        const QString script = bundledScriptPath();
        if (!QFileInfo::exists(python)) {
            log_->append(QStringLiteral("Configured Python does not exist: %1").arg(python));
            return;
        }
        if (!QFileInfo::exists(script)) {
            log_->append(QStringLiteral("Bundled extractor does not exist: %1").arg(script));
            return;
        }
        selectedProducts_.clear();
        if (!inspectOnly) {
            if (!singleProduct_.isEmpty()) {
                selectedProducts_[singleSource_] = {singleProduct_};
            } else {
                for (int i = 0; i < files_->topLevelItemCount(); ++i) {
                    auto *source = files_->topLevelItem(i);
                    QStringList products;
                    for (int j = 0; j < source->childCount(); ++j) {
                        auto *child = source->child(j);
                        if (child->checkState(0) == Qt::Checked)
                            products << child->data(0, Qt::UserRole + 1).toString();
                    }
                    if (!products.isEmpty()) selectedProducts_[source->data(0, Qt::UserRole).toString()] = products;
                }
            }
            if (selectedProducts_.isEmpty()) {
                log_->append(QStringLiteral("Check an output or select a row and use Export this map."));
                return;
            }
            if (!approveSelectedExports()) return;
        }
        inspectOnly_ = inspectOnly;
        cancelled_ = false;
        running_ = true;
        queue_ = inspectOnly ? sources_ : selectedProducts_.keys();
        total_ = queue_.size();
        completed_ = 0;
        progress_->setRange(0, total_);
        progress_->setValue(0);
        log_->append(inspectOnly ? QStringLiteral("Inspecting %1 source(s)…").arg(total_)
                                 : QStringLiteral("Extracting %1 source(s)…").arg(total_));
        updateButtons();
        startNext();
    }

    void startNext() {
        if (queue_.isEmpty()) {
            running_ = false;
            if (reloadPending_) {
                reloadProjectSettings();
                if (running_) return;
            }
            updateButtons();
            statusBar()->showMessage(cancelled_ ? QStringLiteral("Cancelled") : QStringLiteral("Finished"), 5000);
            return;
        }
        current_ = queue_.takeFirst();
        const auto arguments = processArguments();
        if (auto *item = rootForPath(current_)) {
            item->setText(1, inspectOnly_ ? QStringLiteral("Inspecting…") : QStringLiteral("Extracting…"));
        }
        statusBar()->showMessage(QStringLiteral("%1 %2").arg(inspectOnly_ ? QStringLiteral("Inspecting") : QStringLiteral("Extracting"), QFileInfo(current_).fileName()));
        process_->start(configuredPython(), arguments);
    }

    QStringList processArguments() const {
        QStringList arguments{bundledScriptPath(), QStringLiteral("--json")};
        if (inspectOnly_) {
            arguments << QStringLiteral("--inspect");
        } else {
            if (!output_->text().trimmed().isEmpty()) {
                arguments << QStringLiteral("--output-dir") << output_->text().trimmed();
            }
            if (overwrite_->isChecked()) {
                arguments << QStringLiteral("--overwrite");
            }
            if (!exactNpy_->isChecked()) {
                arguments << QStringLiteral("--no-npy");
            }
            if (manifest_->isChecked()) arguments << QStringLiteral("--manifest");
            for (const auto &id : selectedProducts_.value(current_))
                arguments << QStringLiteral("--select") << id;
            if (!directDepthSelected()) arguments << QStringLiteral("--raft-device") << raftDevice_->currentData().toString();
            if (!directDepthSelected() && colorMatching_->isChecked()) {
                arguments << QStringLiteral("--color-matching") << QStringLiteral("--color-hero")
                          << colorHero_->currentData().toString();
            }
        }
        if (directDepthSelected()) {
            arguments << "--learned-depth" << "--learned-model" << depthMethod_->currentData().toString()
                << "--learned-device" << learnedDevice_->currentData().toString()
                << "--learned-input-size" << QString::number(learnedInputSize_->value());
            if (!learnedModel_->text().trimmed().isEmpty()) arguments << "--learned-model-path" << learnedModel_->text().trimmed();
            if (!learnedSource_->text().trimmed().isEmpty()) arguments << "--learned-source-dir" << learnedSource_->text().trimmed();
            QJsonObject modelSettings;
            for (const auto &model : {QString("depthpro"), QString("depth-anything-3"), QString("depth-anything-v2")}) {
                QJsonObject paths;
                const QString key = "learned/" + model;
                const QString checkpoint = model == depthMethod_->currentData().toString() ? learnedModel_->text().trimmed() : sharedValue(key + "/model").toString().trimmed();
                const QString source = model == depthMethod_->currentData().toString() ? learnedSource_->text().trimmed() : sharedValue(key + "/source").toString().trimmed();
                if (!checkpoint.isEmpty()) paths.insert("model_path", checkpoint);
                if (!source.isEmpty()) paths.insert("source_dir", source);
                if (!paths.isEmpty()) modelSettings.insert(model, paths);
            }
            if (!modelSettings.isEmpty()) arguments << "--learned-model-settings" << QString::fromUtf8(QJsonDocument(modelSettings).toJson(QJsonDocument::Compact));
        } else {
            if (!raftModel_->text().trimmed().isEmpty()) arguments << "--raft-model" << raftModel_->text().trimmed();
            if (!raftRoot_->text().trimmed().isEmpty()) arguments << "--raft-root" << raftRoot_->text().trimmed();
            if (!raftMember_->text().trimmed().isEmpty()) arguments << "--raft-model-member" << raftMember_->text().trimmed();
        }
        arguments << current_;
        return arguments;
    }

    void processFinished(int exitCode, QProcess::ExitStatus exitStatus) {
        const QByteArray stdoutBytes = process_->readAllStandardOutput().trimmed();
        const QString stderrText = QString::fromUtf8(process_->readAllStandardError()).trimmed();
        QJsonParseError parseError;
        const QJsonDocument document = QJsonDocument::fromJson(stdoutBytes, &parseError);
        auto *root = rootForPath(current_);
        bool ok = exitStatus == QProcess::NormalExit && exitCode == 0 && document.isObject();
        if (document.isObject()) {
            const QJsonObject object = document.object();
            if (object.contains(QStringLiteral("error"))) {
                ok = false;
                const QString message = object.value(QStringLiteral("error")).toString();
                log_->append(QStringLiteral("%1: %2").arg(QFileInfo(current_).fileName(), message));
                if (root) {
                    root->setText(1, QStringLiteral("Error"));
                    root->setToolTip(1, message);
                }
            } else if (root) {
                const auto selection = modelSelectionKey();
                const auto selectedModel = object.value(QStringLiteral("selected_model")).toObject();
                const auto reportedKind = selectedModel.value(QStringLiteral("kind")).toString();
                // Raw-only exports do not inspect their unused model. Keep the
                // previous inspection valid when its selection still matches.
                if (inspectOnly_ || !reportedKind.isEmpty() || resolvedModelSelection_ != selection) {
                    selectedModelKind_ = reportedKind;
                    resolvedModelSelection_ = selection;
                    QStringList warnings;
                    for (const auto &warning : selectedModel.value("quality_assessment").toObject().value("warnings").toArray()) {
                        if (warning.isString()) warnings << warning.toString();
                    }
                    modelQuality_->setText(warnings.join('\n'));
                    modelQuality_->setVisible(!warnings.isEmpty());
                }
                lastModelSelection_ = selection;
                applyGoal(false);
                const bool firstInspection = root->childCount() == 0;
                QSet<QString> checked;
                const QString currentProduct = files_->currentItem()
                    ? productForSelectedModel(files_->currentItem()->data(0, Qt::UserRole + 1).toString()) : QString();
                for (int j = 0; j < root->childCount(); ++j) {
                    if (root->child(j)->checkState(0) == Qt::Checked)
                        checked.insert(productForSelectedModel(root->child(j)->data(0, Qt::UserRole + 1).toString()));
                }
                while (root->childCount() > 0) {
                    delete root->takeChild(0);
                }
                const QJsonArray assets = object.value(QStringLiteral("assets")).toArray();
                const QJsonObject source = object.value(QStringLiteral("source")).toObject();
                const QJsonObject spatial = source.value(QStringLiteral("spatial_photo")).toObject();
                int outputCount = 0;
                root->setText(
                    1,
                    spatial.isEmpty() ? QStringLiteral("%1 plane(s)").arg(assets.size())
                                      : QStringLiteral("Spatial Photo · %1 plane(s)").arg(assets.size()));
                root->setText(2, inspectOnly_ ? QStringLiteral("Decoded inventory") : QStringLiteral("Verified outputs"));
                for (const QJsonValue &value : assets) {
                    const auto outputs = value.toObject().value(QStringLiteral("outputs")).toArray();
                    outputCount += outputs.size();
                    for (const auto &entry : outputs)
                        log_->append(entry.toObject().value(QStringLiteral("path")).toString().toHtmlEscaped());
                }
                const auto products = object.value(QStringLiteral("available_products")).toArray();
                for (const auto &entry : products) {
                    const auto product = entry.toObject();
                    const auto id = product.value(QStringLiteral("id")).toString();
                    if (id.startsWith("stereo-") || id.startsWith("student-")) continue;
                    if (id.startsWith("learned-") && id != "learned-depthpro" && id != "learned-da3" && id != "learned-da2") continue;
                    if ((directDepthSelected() && id.startsWith("raft-")) || (!directDepthSelected() && id.startsWith("learned-"))) continue;
                    auto *child = new QTreeWidgetItem(root);
                    child->setText(0, product.value("name").toString());
                    child->setText(1, dimensionText(product));
                    child->setText(2, product.value(QStringLiteral("source_precision")).toString());
                    child->setText(3, product.value(QStringLiteral("precision")).toString());
                    auto description = product.value(QStringLiteral("description")).toString();
                    if (individualDepthProduct(id)) description += " Individual export only: select this row and use Export this map. Full-resolution processing takes minutes.";
                    for (int column = 0; column < 4; ++column) child->setToolTip(column, description);
                    child->setData(0, Qt::UserRole + 1, id);
                    if (individualDepthProduct(id)) {
                        child->setFlags(child->flags() & ~Qt::ItemIsUserCheckable);
                    } else {
                        child->setFlags(child->flags() | Qt::ItemIsUserCheckable);
                        child->setCheckState(0, (checked.contains(id) || (firstInspection && suggestedProduct(id, product.value("name").toString().toLower()))) ? Qt::Checked : Qt::Unchecked);
                    }
                    if (id == currentProduct) files_->setCurrentItem(child);
                }
                for (const auto &warning : object.value(QStringLiteral("warnings")).toArray())
                    log_->append(warning.toString().toHtmlEscaped());
                root->setExpanded(true);
                if (!spatial.isEmpty()) {
                    log_->append(
                        QStringLiteral("%1: Spatial Photo — left %2, right %3, baseline %4 mm, disparity adjustment %5%")
                            .arg(QFileInfo(current_).fileName())
                            .arg(spatial.value(QStringLiteral("left_image_index")).toInt())
                            .arg(spatial.value(QStringLiteral("right_image_index")).toInt())
                            .arg(spatial.value(QStringLiteral("baseline_millimeters")).toDouble(), 0, 'f', 6)
                            .arg(spatial.value(QStringLiteral("disparity_adjustment_fraction_of_width")).toDouble() * 100.0, 0, 'f', 4));
                }
                if (!inspectOnly_) {
                    log_->append(QStringLiteral("%1: wrote %2 verified file(s)")
                                     .arg(QFileInfo(current_).fileName())
                                     .arg(outputCount));
                    const auto manifestPath = object.value(QStringLiteral("manifest_path")).toString();
                    if (!manifestPath.isEmpty()) log_->append(QStringLiteral("Manifest: %1").arg(manifestPath.toHtmlEscaped()));
                }
            }
        }
        if (!ok && !document.isObject()) {
            const QString problem = !stderrText.isEmpty()
                                        ? stderrText
                                        : QStringLiteral("Invalid extractor response: %1").arg(parseError.errorString());
            log_->append(QStringLiteral("%1: %2").arg(QFileInfo(current_).fileName(), problem));
            if (root) {
                root->setText(1, QStringLiteral("Error"));
            }
        }
        ++completed_;
        progress_->setValue(completed_);
        if (!cancelled_) {
            startNext();
        } else {
            queue_.clear();
            startNext();
        }
    }

    void updateButtons() {
        const bool hasFiles = !sources_.isEmpty();
        inspect_->setEnabled(hasFiles && !running_);
        extract_->setEnabled(hasFiles && !running_);
        remove_->setEnabled(hasFiles && !running_);
        cancel_->setEnabled(running_);
        output_->setEnabled(!running_);
        exactNpy_->setEnabled(!running_);
        manifest_->setEnabled(!running_);
        goal_->setEnabled(!running_); depthMethod_->setEnabled(!running_);
        learnedDevice_->setEnabled(!running_ && directDepthSelected());
        learnedInputSize_->setEnabled(!running_ && directDepthSelected() && depthMethod_->currentData().toString() != "depthpro");
        learnedModel_->setEnabled(!running_ && directDepthSelected()); learnedSource_->setEnabled(!running_ && directDepthSelected());
        colorMatching_->setEnabled(!running_ && !directDepthSelected());
        colorHero_->setEnabled(!running_ && !directDepthSelected() && colorMatching_->isChecked());
        raftDevice_->setEnabled(!running_ && !directDepthSelected());
        raftModel_->setEnabled(!running_ && !directDepthSelected());
        raftRoot_->setEnabled(!running_ && !directDepthSelected());
        raftMember_->setEnabled(!running_ && !directDepthSelected());
        for (auto *button : findChildren<QPushButton *>()) {
            const QString family = button->property("depthFamily").toString();
            if (!family.isEmpty()) button->setEnabled(!running_ && (family == "ai" ? directDepthSelected() : !directDepthSelected()));
        }
        files_->setEnabled(!running_);
        const auto *item = files_->currentItem();
        exportOne_->setEnabled(!running_ && item && !item->data(0, Qt::UserRole + 1).toString().isEmpty());
        overwrite_->setEnabled(!running_);
    }

    QWidget *advanced_ = nullptr;
    QScrollArea *advancedScroll_ = nullptr;
    QComboBox *depthMethod_ = nullptr, *learnedDevice_ = nullptr;
    QSpinBox *learnedInputSize_ = nullptr;
    QLineEdit *learnedModel_ = nullptr, *learnedSource_ = nullptr;
    QCheckBox *advancedToggle_ = nullptr;
    QComboBox *goal_ = nullptr;
    QLabel *goalHint_ = nullptr;
    QLabel *modelQuality_ = nullptr;
    bool reloadPending_ = false, settingsLoaded_ = false;
    QTreeWidget *files_ = nullptr;
    QLineEdit *output_ = nullptr;
    QLineEdit *raftModel_ = nullptr;
    QLineEdit *raftRoot_ = nullptr;
    QLineEdit *raftMember_ = nullptr;
    QString selectedModelKind_;
    QStringList lastModelSelection_;
    QStringList resolvedModelSelection_;
    QPushButton *exportOne_ = nullptr;
    QMap<QString, QStringList> selectedProducts_;
    QString singleSource_;
    QString singleProduct_;
    QCheckBox *exactNpy_ = nullptr;
    QCheckBox *manifest_ = nullptr;
    QCheckBox *colorMatching_ = nullptr;
    QComboBox *colorHero_ = nullptr;
    QComboBox *raftDevice_ = nullptr;
    QCheckBox *overwrite_ = nullptr;
    QPushButton *inspect_ = nullptr;
    QPushButton *extract_ = nullptr;
    QPushButton *remove_ = nullptr;
    QPushButton *cancel_ = nullptr;
    QProgressBar *progress_ = nullptr;
    QTextEdit *log_ = nullptr;
    QProcess *process_ = nullptr;
    QStringList sources_;
    QStringList queue_;
    QString current_;
    int total_ = 0;
    int completed_ = 0;
    bool inspectOnly_ = true;
    bool running_ = false;
    bool cancelled_ = false;
};

}  // namespace

int main(int argc, char *argv[]) {
    QApplication application(argc, argv);
    application.setApplicationName(QStringLiteral("IPDE"));
    application.setOrganizationName(QStringLiteral("OpenAI"));
    IPDE::ProjectSession session("extractor", &application);
    if (!session.start()) return 2;
    MainWindow window;
    application.setWindowIcon(IPDE::appIcon("extractor"));
    window.setWindowIcon(IPDE::appIcon("extractor"));
    session.setChangedHandler(&window, [&window] { window.reloadProjectSettings(); });
    QObject::connect(&application, &QCoreApplication::aboutToQuit, &session, [&session] { session.shutdown(); });
    window.show();
    if (application.arguments().contains(QStringLiteral("--smoke-test"))) {
        QTimer::singleShot(300, &application, &QCoreApplication::quit);
    }
    return application.exec();
}
