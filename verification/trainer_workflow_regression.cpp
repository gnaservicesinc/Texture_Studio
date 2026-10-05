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
    // Test preparation on its own page so unrelated photo-preview failures from
    // deliberately nonexistent fixture datasets cannot overwrite task results.
    window.tabs_->setCurrentIndex(2);
    require(window.datasetMode_ && window.tabs_->isTabVisible(2), "Dataset Studio did not expose training-set preparation");
    require(!window.compose_->isEnabled(), "empty library enabled composition");
    const QString first = QDir(workspace).filePath("datasets/imported");
    QJsonObject imported = dataset(first); imported.insert("training_eligibility", QJsonObject{{"eligible_count", 3}, {"excluded_count", 1}});
    const auto single = QJsonObject{{"datasets", QJsonArray{imported}}};
    window.populateLibrary(single);
    auto *source = window.collectionSources_->topLevelItem(0);
    require(TrainerWindow::collectionRole(source) == "train" && window.compose_->isEnabled(), "one imported dataset was not immediately usable");
    require(!window.train_->isEnabled(), "Dataset Studio enabled a local model training action");
    require(window.trainingDataset_->text().contains("3 usable targets") && window.trainingDataset_->text().contains("1 unusable target"), "direct training did not explain which targets will be skipped");
    window.reviewedDataset_ = first;
    auto *reviewPhoto = new QTreeWidgetItem(window.reviewSamples_, {"review fixture"});
    auto *reviewEntry = new QTreeWidgetItem(reviewPhoto, {"Teacher"}); reviewEntry->setData(0, Qt::UserRole, QJsonObject{{"id", "excluded"}, {"split", "train"}}); TrainerWindow::setReviewItemIncluded(reviewEntry, false);
    window.updateReviewCount();
    require(!window.compose_->isEnabled() && window.collectionStatus_->text().contains("saving", Qt::CaseInsensitive), "composition ignored pending review exclusions");
    window.compose_->clicked();
    require(!window.busy_ && window.process_->state() == QProcess::NotRunning, "direct composition signal bypassed unsaved exclusion guard");
    TrainerWindow::setReviewItemIncluded(reviewEntry, true); window.updateReviewCount();
    require(window.compose_->isEnabled(), "restoring excluded entries did not restore composition readiness");
    { QSignalBlocker blocker(window.reviewSamples_); window.reviewSamples_->clear(); } window.reviewedDataset_.clear(); window.updateReviewCount();
    window.setCollectionRole(source, "unused");
    require(!window.compose_->isEnabled() && window.collectionStatus_->text().contains("Use for training"), "missing selection was not explained");
    window.populateLibrary(single);
    source = window.collectionSources_->topLevelItem(0);
    require(TrainerWindow::collectionRole(source) == "unused", "refresh overwrote an intentionally unused dataset");
    window.setCollectionRole(source, "train");
    window.collectionName_->setText("bad/name");
    require(!window.compose_->isEnabled(), "invalid output name enabled composition");
    window.collectionName_->setText("test-training-set");
    window.splitMode_->setCurrentIndex(window.splitMode_->findData("explicit"));
    require(!window.compose_->isEnabled() && window.collectionStatus_->text().contains("separate dataset"), "explicit validation was not explained");
    const QString second = QDir(workspace).filePath("datasets/validation");
    window.populateLibrary({{"datasets", QJsonArray{dataset(first), dataset(second)}}});
    auto *validation = window.collectionSources_->topLevelItem(1);
    window.setCollectionRole(validation, "validation");
    require(window.compose_->isEnabled(), "training and explicit validation selection stayed disabled");
    window.setCollectionRole(validation, "train");
    require(TrainerWindow::collectionRole(validation) == "train" && !window.compose_->isEnabled(), "a source was allowed in both roles");
    window.setCollectionRole(validation, "validation");
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
    require(window.compose_->isEnabled() && TrainerWindow::collectionRole(window.collectionSources_->topLevelItem(0)) == "train", "cancelled composition lost selection or stayed disabled");
    window.splitMode_->setCurrentIndex(window.splitMode_->findData("global-random"));
    require(!window.compose_->isEnabled() && TrainerWindow::collectionRole(window.collectionSources_->topLevelItem(1)) == "validation",
            "changing to automatic validation silently included a held-out dataset");
    window.setCollectionRole(window.collectionSources_->topLevelItem(1), "unused");
    require(window.compose_->isEnabled(), "removing the explicit validation role did not restore automatic composition");
    const QString output = QDir(workspace).filePath("datasets/test-training-set");
    window.job_ = "Create training set"; window.continueToTrainer_ = true; window.stdout_ = QJsonDocument(QJsonObject{{"dataset_path", output}}).toJson(); window.setBusy(true);
    window.processFinished(0, QProcess::NormalExit);
    require(window.pendingTrainingPath_ == output, "completed composition did not retain its output for continuation");
    require(window.trainingStatus_->text().contains("Start model training") && !window.trainingStatus_->text().contains("Training complete", Qt::CaseInsensitive), "preparing a training set was presented as model training completion");
    window.job_ = "Refresh library"; window.stdout_ = QJsonDocument(QJsonObject{{"datasets", QJsonArray{dataset(first), dataset(output)}}}).toJson(); window.setBusy(true);
    window.processFinished(0, QProcess::NormalExit);
    require(window.pendingTrainingPath_.isEmpty() && TrainerWindow::selectedPath(window.datasets_) == output, "completed composition did not select its prepared dataset");
    require(window.lastAppRequest_.value("role") == "trainer" && window.lastAppRequest_.value("dataset") == output, "completed preparation did not route the prepared dataset to Trainer");
    window.startJob("Inspect dataset", {"inspect-dataset", QDir(workspace).filePath("missing")});
    require(window.busy_, "real backend failure fixture did not start");
    require(await([&] { return !window.busy_; }), "backend failure left the interface busy");
    require(window.compose_->isEnabled(), "backend failure did not re-enable composition");
    require(window.statusBar()->currentMessage().contains("failed"), "backend failure did not explain its result");
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
        auto *entry = new QTreeWidgetItem(photo, {"partial teacher"}); entry->setData(0, Qt::UserRole, QJsonObject{{"id", "partial"}, {"split", "train"}}); TrainerWindow::setReviewItemIncluded(entry, true);
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

void modelTrainingProgress(TrainerWindow &window, const QString &workspace) {
    window.populateLibrary({{"datasets", QJsonArray{dataset(QDir(workspace).filePath("datasets/imported"))}}});
    const QString selectedDataset = TrainerWindow::selectedPath(window.datasets_);
    require(!selectedDataset.isEmpty(), "training progress fixture has no selected dataset");
    window.job_ = "Train RAFT-Stereo"; window.stdout_.clear(); window.progressBuffer_.clear();
    window.refreshAfter_ = false; window.setBusy(true);
    require(!window.train_->isEnabled() && window.cancel_->isEnabled(), "training did not disable duplicate starts or enable cancellation");
    require(window.trainingStatus_ && !window.trainingStatus_->text().trimmed().isEmpty(), "training has no visible progress explanation");
    auto event = [&](const QString &stage, int epoch, int step, int completed, const QJsonObject &extra = {}) {
        QJsonObject payload{{"phase", "training_progress"}, {"stage", stage}, {"epoch", epoch}, {"epochs", 4},
            {"step", step}, {"steps_per_epoch", 3}, {"completed_steps", completed}, {"total_steps", 12}};
        for (auto it = extra.begin(); it != extra.end(); ++it) payload.insert(it.key(), it.value());
        window.appendProgress("IPDE_EVENT " + QJsonDocument(payload).toJson(QJsonDocument::Compact) + "\n");
    };
    event("checking_dataset", 0, 0, 0);
    require(window.progress_->maximum() == 12 && window.progress_->value() == 0 && window.progress_->isTextVisible(), "checking the dataset was presented as training updates or hid the update count");
    require(window.trainingStatus_->text().contains("check", Qt::CaseInsensitive), "dataset verification stage was hidden");
    event("filtering_targets", 0, 0, 0, {{"eligible_count", 4}, {"excluded_count", 2}});
    require(window.progress_->value() == 0 && window.trainingStatus_->text().contains("2 invalid targets skipped"), "automatically skipped training targets were not explained");
    event("preparing_targets", 0, 0, 0, {{"processed", 2}, {"total", 4}, {"role", "train"}});
    require(window.progress_->value() == 0 && window.trainingStatus_->text().contains("2"), "target preparation did not explain its progress before model updates");
    event("model_setup", 0, 0, 0);
    require(window.trainingStatus_->text().contains("model", Qt::CaseInsensitive), "model loading stage was hidden");
    event("baseline_validation", 0, 0, 0, {{"processed", 1}, {"total", 2}, {"sample_patch", 2}, {"sample_patches", 8}, {"status", "running"}});
    require(window.progress_->value() == 0 && window.trainingStatus_->text().contains("baseline", Qt::CaseInsensitive), "baseline validation was presented as trained model progress");
    require(window.trainingStatus_->text().contains("crop 2 / 8"), "baseline crop progress hid long validation work within a photo");
    event("epoch_step", 1, 1, 0, {{"status", "started"}});
    require(window.progress_->value() == 0 && window.trainingStatus_->text().contains("Epoch 1 / 4") && window.trainingStatus_->text().contains("step 1 / 3"), "a running optimizer step was counted as finished or lacked cycle and step counts");
    window.appendProgress("IPDE_EVENT {\"phase\":\"training_progress\",\"stage\":\"epoch_step\",\"epoch\":1,\"epochs\":4,\"step\":1,");
    require(window.progress_->value() == 0, "a fragmented progress event was consumed before its newline");
    window.appendProgress("\"steps_per_epoch\":3,\"completed_steps\":1,\"total_steps\":12,\"status\":\"finished\",\"loss\":0.125}\n");
    require(window.progress_->maximum() == 12 && window.progress_->value() == 1 && window.trainingStatus_->text().contains("1 / 12"), "completed optimizer update did not advance visible training progress");
    event("epoch_validation", 1, 3, 3, {{"processed", 1}, {"total", 2}, {"status", "running"}});
    require(window.progress_->value() == 3 && window.trainingStatus_->text().contains("validation", Qt::CaseInsensitive), "held-out validation reset the update count or was not explained");
    event("writing_checkpoint", 1, 3, 3, {{"best_epoch", 1}, {"status", "started"}});
    require(window.progress_->value() == 3 && window.trainingStatus_->text().contains("checkpoint", Qt::CaseInsensitive), "checkpoint saving was reported as another optimizer update");
    event("training_stopped", 1, 3, 3, {{"stop_reason", "loss exceeded guard"}});
    require(window.trainingStatus_->text().contains("loss exceeded guard"), "loss safeguard stop reason was hidden");
    event("final_validation", 4, 3, 12, {{"processed", 1}, {"total", 2}, {"status", "running"}});
    require(window.progress_->value() == 12 && window.trainingStatus_->text().contains("validation", Qt::CaseInsensitive) && window.busy_, "final validation hid its stage or reported process completion early");
    const QString checkpoint = QDir(workspace).filePath("runs/progress-fixture/checkpoint.pth");
    const QString bestCheckpoint = QDir(workspace).filePath("runs/progress-fixture/checkpoint-best-step-00000003.pth");
    QDir().mkpath(QFileInfo(bestCheckpoint).absolutePath()); QFile bestFixture(bestCheckpoint);
    require(bestFixture.open(QIODevice::WriteOnly), "could not write best checkpoint fixture"); bestFixture.close();
    event("completed", 4, 3, 12, {{"checkpoint_path", checkpoint}, {"best_epoch", 1}, {"status", "finished"}});
    require(window.busy_ && !window.train_->isEnabled() && window.cancel_->isEnabled(), "checkpoint publication allowed another run before process exit");
    require(!window.trainingStatus_->text().contains("Training complete", Qt::CaseInsensitive), "structured completion claimed process success before exit");
    window.stdout_ = QJsonDocument(QJsonObject{{"checkpoint_path", checkpoint}, {"best_checkpoint_path", bestCheckpoint}, {"best_epoch", 1}, {"epochs_completed", 4}, {"total_steps", 12},
        {"history", QJsonArray{QJsonObject{{"epoch", 1}}, QJsonObject{{"epoch", 2}}, QJsonObject{{"epoch", 3}}, QJsonObject{{"epoch", 4}}}},
        {"quality_assessment", QJsonObject{{"status", "not_improved_full_validation"}, {"warnings", QJsonArray{"Final model did not improve full held-out validation. Review the best checkpoint."}}}},
        {"best_validation", QJsonObject{{"mean_absolute_flow_error_pixels", 1.0}, {"full_validation", true}}},
        {"validation", QJsonObject{{"mean_absolute_flow_error_pixels", 1.25}}}}).toJson();
    window.processFinished(0, QProcess::NormalExit);
    require(window.train_->isEnabled() && !window.cancel_->isEnabled() && window.trainingStatus_->text().contains("finished", Qt::CaseInsensitive), "successful training did not report a completed model and restore controls");
    require(window.trainingStatus_->text().contains("did not improve full held-out validation") && window.trainingStatus_->text().contains(bestCheckpoint), "training summary hid checkpoint quality warnings or immutable best checkpoint path");
    require(!window.bestCheckpoint_->isHidden(), "training result did not offer the best checkpoint");
    const QString initialModel = window.raftModel_->text(); window.bestCheckpoint_->clicked();
    require(TrainerWindow::selectedPath(window.runs_) == bestCheckpoint && window.raftModel_->text() == initialModel, "explicit best checkpoint action failed selection or changed the initial model default");
    const QString completedStatus = window.trainingStatus_->text();
    window.job_ = "Refresh library"; window.stdout_ = QJsonDocument(QJsonObject{{"datasets", QJsonArray{dataset(selectedDataset)}}, {"runs", QJsonArray{}}}).toJson(); window.setBusy(true);
    window.processFinished(0, QProcess::NormalExit);
    require(window.trainingStatus_->text() == completedStatus, "automatic library refresh erased the training result");
    window.job_ = "Train RAFT-Stereo"; window.setBusy(true); event("epoch_step", 2, 1, 3, {{"status", "started"}});
    require(window.bestCheckpoint_->isHidden(), "new training run retained a stale best checkpoint action");
    window.stdout_ = "{\"error\":\"optimizer fixture failure\"}"; window.processFinished(2, QProcess::NormalExit);
    require(window.train_->isEnabled() && !window.cancel_->isEnabled() && window.trainingStatus_->text().contains("failed", Qt::CaseInsensitive), "failed training retained a running or completed status");
    require(window.trainingStatus_->text().contains("optimizer fixture failure"), "training failure reason was hidden from the training step");
    window.job_ = "Train RAFT-Stereo"; window.setBusy(true); event("epoch_step", 2, 2, 4, {{"status", "started"}});
    window.cancelled_ = true; window.processFinished(9, QProcess::CrashExit); window.cancelled_ = false;
    require(window.train_->isEnabled() && !window.cancel_->isEnabled() && window.trainingStatus_->text().contains("cancel", Qt::CaseInsensitive), "cancelled training was not clearly distinguished from completion");
    window.job_ = "Train RAFT-Stereo"; window.setBusy(true); window.process_->setProgram("/missing/ipde-training-python"); window.process_->start();
    require(await([&] { return !window.busy_; }), "failed training launch left the interface busy");
    require(window.trainingStatus_->text().contains("failed", Qt::CaseInsensitive), "failed training launch hid its result from the training step");
}

void trainingControls(TrainerWindow &window, const QString &workspace) {
    require(window.inputSize_->value() == 1036, "fresh configurable teachers did not default to bounded 1036-pixel processing");
    require(window.student_->currentData() == "display" && window.patchLabel_->text().contains("tile"), "display student was not the default visible architecture");
    require(window.earlyStopError_->suffix().contains("fraction") && window.learningRate_->value() == .0001, "display errors or new-head learning rate retained native RAFT units/defaults");
    window.populateLibrary({{"datasets", QJsonArray{dataset(QDir(workspace).filePath("datasets/control-fixture"))}}});
    window.advanced_->setChecked(true); window.epochs_->setValue(4); window.stepsPerUpdate_->setValue(1);
    window.iterations_->setValue(4); window.patch_->setValue(768);
    require(window.trainingDataset_->text().contains("4 RAFT iterations") && !window.trainingDataset_->text().contains("24 RAFT iterations"), "manual refinement changes left the plan summary stale");
    require(window.trainingDataset_->text().contains("768 × 768 pixels") && window.patch_->toolTip().contains("8 columns × 6 rows = 48 tiles"), "tile side length or one-map coverage was unclear");
    require(window.trainingLabelHelp_->text().contains("reference depth pixel") && window.trainingLabelHelp_->text().contains("calculate the loss"), "training explanation omitted the reference depth labels");
    window.validationSamples_->setValue(16);
    require(window.validationSamples_->value() == 1 && window.trainingDataset_->text().contains("all 1 held-out image"), "validation plan requested more held-out images than exist");
    require(window.plannedTrainingSteps() == 12 && window.steps_->value() == 12, "epochs did not visit every training image");
    window.stepsPerUpdate_->setValue(2);
    require(window.plannedTrainingSteps() == 8 && window.steps_->value() == 8, "gradient accumulation did not update Total Steps immediately");
    window.limitMode_->setCurrentIndex(window.limitMode_->findData("steps")); window.steps_->setValue(7);
    require(window.plannedTrainingSteps() == 7 && !window.epochs_->isEnabled() && window.steps_->isEnabled(), "exact Total Steps mode retained epoch controls");
    window.limitMode_->setCurrentIndex(0); window.advanced_->setChecked(false);
    window.quality_->setValue(0); const int lowPatch = window.patch_->value(), lowIterations = window.iterations_->value();
    window.quality_->setValue(2);
    require(window.patch_->value() > lowPatch && window.iterations_->value() > lowIterations, "quality did not increase decoder tile size and refinement");
    QJsonObject limitedValidation = dataset(QDir(workspace).filePath("datasets/three-validation"));
    limitedValidation.insert("sample_count", 160); limitedValidation.insert("train_count", 157); limitedValidation.insert("validation_count", 3);
    window.populateLibrary({{"datasets", QJsonArray{limitedValidation}}});
    require(window.validationSamples_->maximum() == 3 && window.validationSamples_->value() == 3
        && window.trainingDataset_->text().contains("all 3 held-out images") && !window.trainingDataset_->text().contains("16 random"), "large-training preset ignored the small validation set");
    window.advanced_->setChecked(true); window.validationSamples_->setValue(1);
    require(window.trainingDataset_->text().contains("1 random image of 3 held-out images"), "manual validation count did not immediately update the plan");
    window.validationSchedule_->setCurrentIndex(window.validationSchedule_->findData("epoch"));
    require(window.trainingDataset_->text().contains("Validation: Every epoch"), "manual validation schedule left the plan stale");
    window.advanced_->setChecked(false);
    window.length_->setValue(0); const int fastEpochs = window.epochs_->value(); window.length_->setValue(2);
    require(window.epochs_->value() > fastEpochs && window.scope_->currentData() == "update", "duration or small-dataset scope preset ignored dataset size");
    window.teacher_->setCurrentIndex(window.teacher_->findData("depthpro"));
    for (auto *check : window.teacherChecks_) check->setChecked(check->property("model") == "depthpro");
    require(!window.inputSize_->isEnabled(), "fixed-size DepthPro exposed an ineffective custom size");
    for (auto *check : window.teacherChecks_) if (check->property("model") == "depth-anything-3") check->setChecked(true);
    require(window.inputSize_->isEnabled(), "selected DA3 teacher could not use custom input size with DepthPro primary");
    window.inputSize_->setValue(0); require(window.inputSize_->value() == 0, "native teacher size was silently clamped");
    window.inputSize_->setValue(1036);
    window.goal_->setCurrentIndex(window.goal_->findData("effect/map")); window.applyGoal(false);
    require(window.inputSize_->value() == 1036, "effect preset discarded an explicit teacher size override");
    require(window.settings_.value("teacher_input_size").toInt() == 1036, "teacher size override was not persisted");
    window.advanced_->setChecked(true);
    QJsonObject models = dataset(QDir(workspace).filePath("datasets/model-eligibility"));
    models.insert("training_eligibility", QJsonObject{{"trainable", true}, {"train_count", 5}, {"eligible_count", 6}, {"excluded_count", 0}, {"units", "relative_depth"}});
    models.insert("raft_training_eligibility", QJsonObject{{"trainable", false}, {"train_count", 2}, {"reason", "Native left-grid targets unavailable"}});
    window.populateLibrary({{"datasets", QJsonArray{models}}});
    window.epochs_->setValue(3); window.stepsPerUpdate_->setValue(1);
    require(window.train_->isEnabled() && window.trainingImageCount() == 5 && window.plannedTrainingSteps() == 15, "display selection ignored display target eligibility/counts");
    require(window.trainingDataset_->text().contains("relative_depth") && window.trainingDataset_->text().contains("No automatic meter conversion"), "display training concealed the relative checkpoint unit convention");
    models.insert("training_eligibility_by_mode", QJsonObject{{"supervised", QJsonObject{{"trainable", false}, {"train_count", 1}, {"reason", "Only one display-grid reference"}}}, {"distillation", models.value("training_eligibility")}});
    window.populateLibrary({{"datasets", QJsonArray{models}}});
    window.trainingMode_->setCurrentIndex(window.trainingMode_->findData("supervised"));
    require(!window.train_->isEnabled() && window.trainingImageCount() == 1 && window.plannedTrainingSteps() == 3, "label mode reused automatic eligibility and step counts");
    window.trainingMode_->setCurrentIndex(window.trainingMode_->findData("distillation"));
    require(window.train_->isEnabled() && window.trainingImageCount() == 5 && window.plannedTrainingSteps() == 15, "distillation did not restore matching target counts");
    window.student_->setCurrentIndex(window.student_->findData("raft"));
    require(!window.train_->isEnabled() && window.trainingImageCount() == 2 && window.plannedTrainingSteps() == 6, "stock selection reused display eligibility/counts");
    require(window.patchLabel_->text().contains("Native stereo") && window.earlyStopError_->suffix().contains("px") && window.learningRate_->value() == .00001, "stock selection retained display settings semantics");
    window.student_->setCurrentIndex(window.student_->findData("display"));
    require(window.train_->isEnabled() && window.trainingDataset_->text().contains("Display tiles") && !window.trainingDataset_->text().contains("px MAE"), "switching back failed to restore display readiness/units");
    require(TrainerWindow::validationDescription({{"mean_relative_depth_error", .125}, {"mean_absolute_depth_error", 2.0}, {"units", "meters"}}).contains("12.50%"), "display report was presented as pixel error");
    window.exportDestination_ = QDir(workspace).filePath("exports/fixture");
    require(window.exportedCheckpointPath({{"schema", "ipde-display-model-export-v1"}, {"checkpoint", "display-model.pth"}}).endsWith("/display-model.pth"), "display export filename was forced to stock RAFT");
    require(window.exportedCheckpointPath({{"checkpoint", "raft-model.pth"}}).endsWith("/raft-model.pth"), "stock export filename was lost");
    window.updateTrainingProgress({{"stage", "display_tile"}, {"epoch", 1}, {"epochs", 3}, {"step", 0}, {"steps_per_epoch", 5}, {"tile", 2}, {"tiles", 10}, {"completed_steps", 0}, {"total_steps", 15}});
    require(window.trainingStatus_->text().contains("display tile 2 / 10"), "display training hid full-grid tile progress");
    models.insert("training_eligibility", QJsonObject{{"trainable", true}, {"train_count", 1000000}}); models.remove("training_eligibility_by_mode");
    window.populateLibrary({{"datasets", QJsonArray{models}}}); window.epochs_->setValue(10000);
    require(window.plannedTrainingSteps() == qint64(10000000000LL) && !window.train_->isEnabled() && window.trainingDataset_->text().contains("10000000000"), "large epoch plans overflowed or silently launched with a clamped step display");
    window.epochs_->setValue(3); window.populateLibrary({{"datasets", QJsonArray{dataset(QDir(workspace).filePath("datasets/control-fixture"))}}});
}

void da3TeacherSizeMigration(const QString &project) {
    QSettings settings(QSettings::defaultFormat(), QSettings::UserScope, "IPDE", "RAFTStudio");
    QSettings projectSettings(QDir(project).filePath("project.ini"), QSettings::IniFormat);
    projectSettings.setValue("goal", "manual"); projectSettings.sync();
    const auto legacy = [&](const QString &primary) {
        settings.clear(); settings.setValue("teacher_model", primary); settings.setValue("teacher_input_size", 0); settings.sync();
    };
    legacy("depth-anything-3");
    {
        TrainerWindow window(true); window.autosavePaused_ = true;
        require(window.inputSize_->value() == 1036 && settings.value("teacher_da3_size_default_migrated").toBool(), "saved native DA3 primary was not migrated once");
        window.inputSize_->setValue(728);
        window.teacher_->setCurrentIndex(window.teacher_->findData("depthpro"));
        window.teacher_->setCurrentIndex(window.teacher_->findData("depth-anything-3"));
        require(window.inputSize_->value() == 728, "DA3 selection discarded an explicit custom processing size");
        window.inputSize_->setValue(0); window.settings_.sync();
    }
    {
        TrainerWindow window(true); window.autosavePaused_ = true;
        require(window.inputSize_->value() == 0, "a later explicit native-size choice was migrated again on restart");
    }
    legacy("depthpro");
    {
        TrainerWindow window(true); window.autosavePaused_ = true;
        require(window.inputSize_->value() == 0, "DA3 migration altered an existing native setting without DA3 selection");
        for (auto *check : window.teacherChecks_) if (check->property("model") == "depth-anything-3") check->setChecked(true);
        require(window.inputSize_->value() == 1036, "saved native additional DA3 teacher was not migrated");
    }
    legacy("depthpro");
    {
        TrainerWindow window(true); window.autosavePaused_ = true;
        window.reviewedDataset_ = QDir(project).filePath("dataset"); window.requestedReviewPath_ = window.reviewedDataset_;
        auto *photo = new QTreeWidgetItem(window.reviewSamples_, {"photo"});
        auto *entry = new QTreeWidgetItem(photo, {"teacher"});
        entry->setData(0, Qt::UserRole, QJsonObject{{"id", "photo"}, {"source_id", "source"}});
        window.busy_ = true; window.activeOperation_ = "generate-teacher";
        window.queuePhotoTeacher("generate-teacher", "depth-anything-3", {entry});
        require(window.pendingTeacherJobs_.size() == 1, "per-photo DA3 migration fixture did not queue generation");
        const auto args = window.pendingTeacherJobs_.first();
        require(args.value(args.indexOf("--input-size") + 1) == "1036", "per-photo DA3 generation retained the old unsafe native-size setting");
        window.pendingTeacherJobs_.clear(); window.busy_ = false; window.activeOperation_.clear();
    }
    settings.clear(); settings.sync(); projectSettings.remove("goal"); projectSettings.sync();
}

void checkpointExportLifetime(TrainerWindow &window) {
    window.configureExportProcess();
    window.exportProcess_->setProgram("/bin/sleep"); window.exportProcess_->setArguments({"0.1"}); window.exportProcess_->start();
    require(window.exportProcess_->waitForStarted(), "checkpoint export lifetime fixture failed to start");
    require(window.taskRunning(), "independent checkpoint export was omitted from task lifetime");
    QCloseEvent close; QApplication::sendEvent(&window, &close);
    require(!close.isAccepted() && window.closeAfterTraining_, "closing terminated an unfinished checkpoint export");
    require(await([&] { return !window.exportRunning() && !window.closeAfterTraining_; }), "completed export did not release pending close");
    require(window.exportProcess_->exitStatus() == QProcess::NormalExit && window.exportProcess_->exitCode() == 0, "pending close killed the checkpoint export");
}

void applicationResponsibilities(TrainerWindow &trainer, TrainerWindow &datasets, const QString &workspace) {
    require(trainer.tabs_->currentIndex() == 4 && !trainer.tabs_->isTabVisible(0) && !trainer.tabs_->isTabVisible(2) && !trainer.tabs_->isTabVisible(3), "Trainer exposed dataset import or preparation tabs");
    require(trainer.tabs_->isTabVisible(1) && trainer.tabs_->tabText(1) == "Compare models" && trainer.reviewSamples_->isHidden() && trainer.saveReviewed_->isHidden(), "Trainer model comparison exposed dataset editing controls");
    require(trainer.findChild<QPushButton *>("compactDataset")->isHidden() && trainer.findChild<QPushButton *>("archiveDataset")->isHidden(), "Trainer exposed compact or archive dataset actions");
    const QString path = QDir(workspace).filePath("datasets/imported");
    trainer.reviewDataset(path); require(trainer.lastAppRequest_.value("role") == "datasets" && trainer.lastAppRequest_.value("dataset") == path, "Trainer did not route review with the selected dataset");
    for (const QString &operation : {"dataset", "compose-datasets", "compact-dataset", "curate-dataset", "import-hf", "cleanup-dataset", "archive-dataset"}) {
        trainer.startJob("Wrong app fixture", {operation, path});
        require(!trainer.busy_ && trainer.process_->state() == QProcess::NotRunning && trainer.lastAppRequest_.value("role") == "datasets", "Trainer started a dataset mutation through a hidden action");
    }
    datasets.trainDataset(); require(datasets.lastAppRequest_.value("role") == "trainer", "Dataset Studio started model training locally");
    QDir().mkpath(path); datasets.populateLibrary({{"datasets", QJsonArray{dataset(path)}}});
    datasets.projectOperations_ = {{"trainer", "train"}}; datasets.updateCleanupActions(); require(!datasets.cleanupDataset_->isEnabled(), "Dataset cleanup remained available during model training");
    datasets.projectOperations_ = {}; datasets.updateCleanupActions(); require(datasets.cleanupDataset_->isEnabled(), "Owned dataset cleanup stayed disabled after model training");
    datasets.projectSettings_->setValue("performance/workers", 3); require(datasets.workerCount() == 3, "Dataset Studio ignored the project's file-worker override");
    datasets.reviewedDataset_ = path; datasets.requestedReviewPath_ = path;
    datasets.clearUnavailableReview(path, "Removed fixture");
    require(datasets.reviewedDataset_.isEmpty() && datasets.requestedReviewPath_.isEmpty() && datasets.reviewEntries().isEmpty() && !datasets.saveReviewed_->isEnabled(), "removing a reviewed dataset left saveable stale review state");
}

void visibleDisplayRegistration(TrainerWindow &window, const QString &workspace) {
    const QString image = QDir(workspace).filePath("alignment-preview.png"); QPixmap fixture(96, 64); fixture.fill(Qt::gray);
    require(fixture.save(image), "could not save alignment preview fixture");
    QJsonObject registration{{"accepted", false}, {"reference_role", "right"}, {"reason", "held-out p90 13.20 px exceeds 3 px"},
        {"heldout_median_error_pixels", 2.15}, {"heldout_p90_error_pixels", 13.20}};
    QJsonObject preview{{"sample_id", "aligned-native"}, {"label", "training"}, {"label_title", "Selected training target: Teacher"},
        {"rgb_preview_path", image}, {"depth_preview_path", image}, {"width", 96}, {"height", 64}, {"valid_fraction", 1.0},
        {"min", 1.0}, {"max", 2.0}, {"units", "meters"}, {"rgb_reference", "spatial_left"}, {"display_registration", registration}};
    window.populatePreview(preview);
    QString text = window.previewStats_->text();
    require(text.contains("Source: Native left stereo grid") && window.sourceTitle_->text() == "Native left stereo grid", "native stereo-left preview did not identify its actual source grid");
    require(text.contains("Display alignment rejected") && text.contains("fit reference: right") && text.contains("median 2.15 px / p90 13.20 px"), "display rejection and held-out geometry evidence remained hidden in tooltips");
    require(text.contains("Native-left targets remain usable") && !text.contains("Display alignment accepted"), "rejected display registration implied the direct native-left label was unusable or the display was accepted");
    preview.insert("rgb_reference", "display"); window.populatePreview(preview);
    require(window.sourceTitle_->text().contains("separate grid") && window.previewStats_->text().contains("cannot supply an aligned stereo target") && !window.previewStats_->text().contains("Source: Native left"), "display preview was mislabeled as a stereo-left grid");
    registration.insert("accepted", true); registration.insert("reference_role", "left"); preview.insert("display_registration", registration); preview.insert("rgb_reference", "spatial_left");
    window.populatePreview(preview);
    require(window.previewStats_->text().contains("Display alignment accepted") && window.previewStats_->text().contains("fit reference: left") && !window.previewStats_->text().contains("Display alignment rejected"), "accepted registration retained a stale rejected status");
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
        da3TeacherSizeMigration(temporary.path());
        TrainerWindow window(true), trainer(false); window.autosavePaused_ = true; trainer.autosavePaused_ = true;
        const QString workspace = QDir(temporary.path()).filePath("workspace"); QDir().mkpath(workspace);
        trainingSetReadiness(window, workspace);
        streamedFolderImport(window, workspace);
        failedGenerationStreaming(window, workspace);
        trainer.workspace_->setText(workspace); modelTrainingProgress(trainer, workspace);
        trainingControls(trainer, workspace);
        applicationResponsibilities(trainer, window, workspace);
        visibleDisplayRegistration(window, workspace);
        checkpointExportLifetime(trainer);
        std::cout << "Dataset Studio / Trainer: exclusive dataset management, cross-app continuation, cleanup guards, worker override, readiness, review, retry, scan, and training progress passed\n";
        return 0;
    } catch (const std::exception &error) { std::cerr << error.what() << '\n'; return 1; }
}
