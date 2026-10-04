// Exercise the real Qt workflow without starting teacher inference or training.
#define IPDE_STUDIO_REGRESSION
#define main ipde_studio_application_main
#include "../src/gui/trainer.cpp"
#undef main

#include <QElapsedTimer>
#include <QThread>
#include <iostream>
#include <stdexcept>

namespace {
void require(bool condition, const char *message) {
    if (!condition) throw std::runtime_error(message);
}

bool await(const std::function<bool()> &condition, int timeout = 5000) {
    QElapsedTimer timer; timer.start();
    while (!condition() && timer.elapsed() < timeout) {
        QCoreApplication::processEvents(QEventLoop::AllEvents, 10);
        QThread::msleep(1);
    }
    return condition();
}

QJsonObject dataset(const QString &path) {
    return {{"path", path}, {"name", QFileInfo(path).fileName()}, {"sample_count", 4}, {"train_count", 3}, {"validation_count", 1}};
}

void trainingSetReadiness(TrainerWindow &window, const QString &workspace) {
    window.workspace_->setText(workspace);
    require(!window.compose_->isEnabled(), "empty library enabled composition");
    const QString first = QDir(workspace).filePath("datasets/imported");
    const auto single = QJsonObject{{"datasets", QJsonArray{dataset(first)}}};
    window.populateLibrary(single);
    auto *source = window.collectionSources_->topLevelItem(0);
    require(source->checkState(0) == Qt::Checked && window.compose_->isEnabled(), "one imported dataset was not immediately usable");
    window.reviewedDataset_ = first;
    auto *reviewPhoto = new QTreeWidgetItem(window.reviewSamples_, {"review fixture"});
    auto *reviewEntry = new QTreeWidgetItem(reviewPhoto, {"Teacher"}); reviewEntry->setData(0, Qt::UserRole, QJsonObject{{"id", "excluded"}, {"split", "train"}}); reviewEntry->setCheckState(0, Qt::Unchecked);
    window.updateReviewCount();
    require(!window.compose_->isEnabled() && window.collectionStatus_->text().contains("Save reviewed copy"), "composition ignored unsaved review exclusions");
    window.compose_->clicked();
    require(!window.busy_ && window.process_->state() == QProcess::NotRunning, "direct composition signal bypassed unsaved exclusion guard");
    reviewEntry->setCheckState(0, Qt::Checked);
    require(window.compose_->isEnabled(), "restoring excluded entries did not restore composition readiness");
    { QSignalBlocker blocker(window.reviewSamples_); window.reviewSamples_->clear(); } window.reviewedDataset_.clear(); window.updateReviewCount();
    source->setCheckState(0, Qt::Unchecked);
    require(!window.compose_->isEnabled() && window.collectionStatus_->text().contains("Tick a dataset"), "missing selection was not explained");
    window.populateLibrary(single);
    source = window.collectionSources_->topLevelItem(0);
    require(source->checkState(0) == Qt::Unchecked, "refresh overwrote an intentional unchecked selection");
    source->setCheckState(0, Qt::Checked);
    window.collectionName_->setText("bad/name");
    require(!window.compose_->isEnabled(), "invalid output name enabled composition");
    window.collectionName_->setText("test-training-set");
    window.splitMode_->setCurrentIndex(window.splitMode_->findData("explicit"));
    require(!window.compose_->isEnabled() && window.collectionStatus_->text().contains("separate dataset"), "explicit validation was not explained");
    const QString second = QDir(workspace).filePath("datasets/validation");
    window.populateLibrary({{"datasets", QJsonArray{dataset(first), dataset(second)}}});
    auto *validation = window.collectionSources_->topLevelItem(1);
    validation->setCheckState(1, Qt::Checked);
    require(window.compose_->isEnabled(), "training and explicit validation selection stayed disabled");
    validation->setCheckState(0, Qt::Checked);
    require(validation->checkState(1) == Qt::Unchecked && !window.compose_->isEnabled(), "a source was allowed in both roles");
    validation->setCheckState(1, Qt::Checked);
    window.job_ = "Create training set"; window.setBusy(true);
    require(!window.compose_->isEnabled() && !window.collectionSources_->isEnabled() && window.cancel_->isEnabled(), "running composition allowed a second job or changed inputs");
    require(window.compose_->text() == "Creating training set…" && window.collectionStatus_->text().contains("several minutes"), "running composition did not explain the disabled button");
    window.appendProgress("IPDE_EVENT {\"phase\":\"sample_composed\",\"processed\":2,");
    window.appendProgress("\"total\":4,\"unique_arrays\":8}\n");
    require(window.progress_->maximum() == 4 && window.progress_->value() == 2 && window.collectionStatus_->text().contains("2 of 4"), "composition phase progress was discarded");
    window.appendProgress("IPDE_EVENT {\"phase\":\"verifying_output\"}\n");
    require(window.progress_->maximum() == 0 && window.collectionStatus_->text().contains("before publishing"), "output verification was presented as completed");
    window.stdout_ = "{\"error\":\"fixture failure\"}"; window.processFinished(2, QProcess::NormalExit);
    require(window.compose_->isEnabled() && !window.cancel_->isEnabled() && window.statusBar()->currentMessage().contains("failed"), "failed composition did not restore an explained retry state");
    require(window.collectionStatus_->text().contains("fixture failure"), "composition failure reason was hidden from the training-set step");
    window.job_ = "Create training set"; window.setBusy(true); window.cancelled_ = true;
    window.processFinished(9, QProcess::CrashExit); window.cancelled_ = false;
    require(window.compose_->isEnabled() && window.collectionSources_->topLevelItem(0)->checkState(0) == Qt::Checked, "cancelled composition lost selection or stayed disabled");
    window.splitMode_->setCurrentIndex(window.splitMode_->findData("global-random"));
    const QString output = QDir(workspace).filePath("datasets/test-training-set");
    window.job_ = "Create training set"; window.stdout_ = QJsonDocument(QJsonObject{{"dataset_path", output}}).toJson(); window.setBusy(true);
    window.processFinished(0, QProcess::NormalExit);
    require(window.pendingTrainingPath_ == output, "completed composition did not retain its output for continuation");
    window.job_ = "Refresh library"; window.stdout_ = QJsonDocument(QJsonObject{{"datasets", QJsonArray{dataset(first), dataset(output)}}}).toJson(); window.setBusy(true);
    window.processFinished(0, QProcess::NormalExit);
    require(window.pendingTrainingPath_.isEmpty() && window.tabs_->currentIndex() == 4 && TrainerWindow::selectedPath(window.datasets_) == output, "completed composition did not select its set and continue to training");
    window.startJob("Inspect dataset", {"inspect-dataset", QDir(workspace).filePath("missing")});
    require(window.busy_, "real backend failure fixture did not start");
    require(await([&] { return !window.busy_; }), "backend failure left the interface busy");
    require(window.compose_->isEnabled() && window.statusBar()->currentMessage().contains("failed"), "backend failure did not re-enable composition");
    window.job_ = "Failed launch"; window.setBusy(true); window.process_->setProgram("/missing/ipde-python-fixture"); window.process_->start();
    require(await([&] { return !window.busy_; }), "failed process launch left the interface busy");
    require(window.compose_->isEnabled() && window.statusBar()->currentMessage().contains("failed to start"), "failed launch did not re-enable composition with an explanation");
}

void streamedFolderImport(TrainerWindow &window, const QString &workspace) {
    const QString photo = QDir(workspace).filePath("sample.HEIC");
    QFile file(photo); require(file.open(QIODevice::WriteOnly), "cannot create folder-import fixture"); file.write("fixture"); file.close();
    require(window.addSourcePhoto(photo), "manual source insertion failed");
    require(!window.addSourcePhoto(QDir(workspace).filePath("./sample.HEIC")), "canonical path alias duplicated a source");
    const QString found = QDir(workspace).filePath("found.HEIC");
    window.job_ = "Scan spatial photos"; window.setBusy(true);
    const QJsonObject accepted{{"source_path", found}, {"photo_metadata", QJsonObject{{"camera_model", "iPhone fixture"}, {"captured_at", "fixture date"}}}};
    QJsonObject event = accepted; event.insert("event", "spatial_photo_found");
    window.appendProgress("IPDE_EVENT " + QJsonDocument(event).toJson(QJsonDocument::Compact) + "\n");
    require(window.sources_->topLevelItemCount() == 2 && window.sources_->topLevelItem(1)->text(2) == "iPhone fixture", "folder scan did not stream found photos with metadata");
    delete window.sources_->takeTopLevelItem(1);
    const QString fallbackPath = QDir(workspace).filePath("not-streamed.HEIC");
    const QJsonObject fallback{{"source_path", fallbackPath}};
    window.stdout_ = QJsonDocument(QJsonObject{{"accepted", QJsonArray{accepted, fallback}}, {"skipped", QJsonArray{}}}).toJson(); window.processFinished(0, QProcess::NormalExit);
    require(window.sources_->topLevelItemCount() == 2 && window.sources_->topLevelItem(1)->data(0, Qt::UserRole).toString() == fallbackPath, "final scan restored a deliberately removed photo or lost an unstreamed fallback");
    window.job_ = "Scan spatial photos"; window.setBusy(true); window.cancelled_ = true; window.processFinished(9, QProcess::CrashExit); window.cancelled_ = false;
    require(window.sources_->topLevelItemCount() == 2 && window.statusBar()->currentMessage().contains("remain in the list"), "cancelled scan discarded usable found photos or failed to explain retention");
    window.startJob("Scan spatial photos", {"scan-spatial", QDir(workspace).filePath("missing-folder")});
    require(window.scanDeliveredPaths_.isEmpty(), "new scan inherited delivered paths from an earlier scan");
    require(await([&] { return !window.busy_; }), "scan error fixture did not complete");
}

void failedGenerationStreaming(TrainerWindow &window, const QString &workspace) {
    const QString temporaryDataset = QDir(workspace).filePath("datasets/.generating-fixture");
    auto prepare = [&] {
        window.job_ = "Generate dataset"; window.streamingDataset_ = temporaryDataset; window.reviewedDataset_ = temporaryDataset; window.reviewGenerating_ = true;
        window.requestedReviewPath_ = temporaryDataset; window.previewKind_ = "review"; window.previewProcess_->setArguments({"review-dataset", temporaryDataset});
        window.pendingPreviewArgs_ = {"preview-sample", temporaryDataset};
        auto *photo = new QTreeWidgetItem(window.reviewSamples_, {"partial photo"});
        auto *entry = new QTreeWidgetItem(photo, {"partial teacher"}); entry->setData(0, Qt::UserRole, QJsonObject{{"id", "partial"}, {"split", "train"}}); entry->setCheckState(0, Qt::Checked);
        window.streamingTimer_->start(); window.setBusy(true);
    };
    auto verify = [&] {
        require(!window.streamingTimer_->isActive() && window.streamingDataset_.isEmpty(), "failed generation scheduled a late review of temporary files");
        require(!window.reviewGenerating_ && window.reviewedDataset_.isEmpty() && window.reviewEntries().isEmpty() && !window.saveReviewed_->isEnabled(), "failed generation retained a saveable stale review");
        require(window.pendingPreviewArgs_.isEmpty() && window.previewKind_ == "discarded", "failed generation retained or accepted stale preview results");
    };
    prepare(); window.stdout_ = "{\"error\":\"generation fixture failed\"}"; window.processFinished(1, QProcess::NormalExit); verify();
    prepare(); window.cancelled_ = true; window.processFinished(9, QProcess::CrashExit); window.cancelled_ = false; verify();
    prepare(); window.process_->setProgram("/missing/ipde-generation-python"); window.process_->start();
    require(await([&] { return !window.busy_; }), "failed generation launch left the interface busy"); verify();
    const QString unrelated = QDir(workspace).filePath("datasets/existing-complete");
    window.streamingDataset_ = temporaryDataset; window.reviewedDataset_ = unrelated; window.reviewGenerating_ = false; window.requestedReviewPath_ = unrelated;
    window.pendingPreviewArgs_ = {"review-dataset", unrelated};
    window.stopGenerationStreaming("fixture cleanup");
    require(window.reviewedDataset_ == unrelated && window.pendingPreviewArgs_.contains(unrelated) && window.requestedReviewPath_ == unrelated, "generation cleanup discarded unrelated review work");
    window.pendingPreviewArgs_.clear();
}
} // namespace

int main(int argc, char **argv) {
    Q_UNUSED(argc);
    QTemporaryDir temporary;
    require(temporary.isValid(), "cannot create workflow fixture");
    QByteArray project = temporary.path().toUtf8();
    char projectOption[] = "--project", smoke[] = "--smoke-test";
    char *arguments[] = {argv[0], projectOption, project.data(), smoke, nullptr}; int count = 4;
    QApplication application(count, arguments); application.setQuitOnLastWindowClosed(false);
    QSettings::setDefaultFormat(QSettings::IniFormat);
    QSettings::setPath(QSettings::IniFormat, QSettings::UserScope, temporary.path());
    try {
        TrainerWindow window;
        const QString workspace = QDir(temporary.path()).filePath("workspace"); QDir().mkpath(workspace);
        trainingSetReadiness(window, workspace);
        streamedFolderImport(window, workspace);
        failedGenerationStreaming(window, workspace);
        std::cout << "Trainer: readiness, review exclusions, progress, retry/cancel, continuation, scan removals, and failed generation streaming passed\n";
        return 0;
    } catch (const std::exception &error) { std::cerr << error.what() << '\n'; return 1; }
}
