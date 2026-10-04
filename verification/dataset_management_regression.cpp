// Exercise dataset selection and photo edits against the real Qt interface.
#define IPDE_STUDIO_REGRESSION
#define main ipde_studio_application_main
#include "../src/gui/trainer.cpp"
#undef main

#include <QCheckBox>
#include <QElapsedTimer>
#include <QMouseEvent>
#include <QThread>
#include <QTimer>
#include <QTreeWidgetItemIterator>
#include <iostream>
#include <stdexcept>

namespace {
void require(bool condition, const char *message) {
    if (!condition) throw std::runtime_error(message);
}

bool await(const std::function<bool()> &condition, int timeout = 15000) {
    QElapsedTimer timer; timer.start();
    while (!condition() && timer.elapsed() < timeout) {
        QApplication::processEvents(QEventLoop::AllEvents, 10); QThread::msleep(1);
    }
    return condition();
}

QJsonObject libraryDataset(const QString &path) {
    return {{"path", path}, {"name", QFileInfo(path).fileName()}, {"sample_count", 4},
            {"train_count", 3}, {"validation_count", 1}};
}

QJsonObject sample(const QString &id, const QString &photo, const QString &group,
                   const QString &teacher, const QString &split = "train") {
    return {{"id", id}, {"source_id", photo}, {"source_path", "/fixture/" + photo + ".HEIC"},
            {"group_id", group}, {"requested_group", group}, {"teacher_id", teacher},
            {"split", split}, {"included", true}, {"excluded", false}, {"labels", QJsonArray{}}, {"photo_metadata", QJsonObject{}}};
}

QJsonObject review(const QString &path) {
    // Two teachers for photo one, a related second photo, and one independent photo.
    return {{"dataset_path", path}, {"samples", QJsonArray{
        sample("shared-1", "one", "capture-a", "Teacher A"),
        sample("shared-2", "one", "capture-a", "Teacher B"),
        sample("related", "two", "capture-a", "Teacher A"),
        sample("held-out", "three", "capture-b", "Teacher A", "validation")}}};
}

void populateFixtureReview(TrainerWindow &window, const QJsonObject &result) {
    // A direct fixture result represents completion of its own review request.
    window.requestedReviewPath_ = result.value("dataset_path").toString();
    window.populateReview(result);
}

void selectOnly(QTreeWidget *tree, QTreeWidgetItem *item) {
    tree->clearSelection(); tree->setCurrentItem(item); item->setSelected(true);
}

void mouseClick(QWidget *widget, const QPoint &position) {
    const QPointF local(position), global(widget->mapToGlobal(position));
    QMouseEvent press(QEvent::MouseButtonPress, local, local, global,
                      Qt::LeftButton, Qt::LeftButton, Qt::NoModifier);
    QApplication::sendEvent(widget, &press);
    QMouseEvent release(QEvent::MouseButtonRelease, local, local, global,
                        Qt::LeftButton, Qt::NoButton, Qt::NoModifier);
    QApplication::sendEvent(widget, &release);
    QApplication::processEvents();
}

void requireNoItemCheckboxes(QTreeWidget *tree) {
    QTreeWidgetItemIterator entries(tree);
    while (*entries) {
        auto *item = *entries;
        require(!(item->flags() & Qt::ItemIsUserCheckable), "tree row still exposes a native checkbox");
        for (int column = 0; column < tree->columnCount(); ++column)
            require(!item->data(column, Qt::CheckStateRole).isValid(), "tree row still stores a native checkbox state");
        ++entries;
    }
}

QByteArray readBytes(const QString &path) {
    QFile file(path); require(file.open(QIODevice::ReadOnly), "cannot read fixture manifest");
    return file.readAll();
}

void stopEditProcess(TrainerWindow &window) {
    if (window.editProcess_->state() != QProcess::NotRunning) {
        window.editProcess_->kill(); window.editProcess_->waitForFinished(1500);
    }
}

void clearPendingFixtureEdits(TrainerWindow &window) {
    window.autosaveTimer_->stop(); window.reviewDrafts_.clear(); window.reviewAdditions_.clear();
    require(window.persistReviewDrafts(), "cannot clear the isolated fixture edit journal");
}

void writeFixtureManifest(const QString &path, QJsonObject manifest) {
    QDir().mkpath(path); QFile file(QDir(path).filePath("dataset.json"));
    require(file.open(QIODevice::WriteOnly), "cannot write fixture manifest");
    require(file.write(QJsonDocument(manifest).toJson()) >= 0, "cannot serialize fixture manifest");
}

void completeMetadataSave(TrainerWindow &window, const QString &path, QJsonObject manifest) {
    writeFixtureManifest(path, manifest);
    const QString hash = window.manifestHash(path);
    const auto samples = manifest.value("samples").toArray(); int kept = 0, train = 0, validation = 0;
    for (const auto &value : samples) {
        const auto sample = value.toObject(); if (sample.value("excluded").toBool()) continue;
        ++kept; train += sample.value("split").toString() == "train";
        validation += sample.value("split").toString() == "validation";
    }
    window.editStdout_ = QJsonDocument(QJsonObject{{"dataset_path", path}, {"manifest_sha256", hash},
        {"summary", QJsonObject{{"samples", kept}, {"train_samples", train}, {"validation_samples", validation}}}}).toJson();
    window.finishDatasetSave(0, QProcess::NormalExit);
}

void librarySelection(TrainerWindow &window, const QString &workspace) {
    // The fixture changes the workspace programmatically; its later focus loss
    // must not replace the supplied library while mouse interactions are tested.
    QSignalBlocker fixtureWorkspaceSignals(window.workspace_);
    const QString first = QDir(workspace).filePath("datasets/first");
    const QString second = QDir(workspace).filePath("datasets/second");
    const QJsonObject library{{"datasets", QJsonArray{libraryDataset(first), libraryDataset(second)}}};
    window.populateLibrary(library);
    window.tabs_->setCurrentIndex(2); window.show(); window.resize(1280, 900);
    QApplication::processEvents();
    require(window.findChildren<QCheckBox *>().isEmpty(), "Dataset Studio still contains native checkbox widgets");
    requireNoItemCheckboxes(window.collectionSources_);
    require(TrainerWindow::collectionRole(window.collectionSources_->topLevelItem(0)) == "unused" &&
            TrainerWindow::collectionRole(window.collectionSources_->topLevelItem(1)) == "unused",
            "multi-dataset fixture started with an unexpected training role");
    auto *lowerSecond = window.collectionSources_->topLevelItem(1);
    window.collectionSources_->scrollToItem(lowerSecond);
    const QRect secondRect = window.collectionSources_->visualItemRect(lowerSecond);
    require(secondRect.isValid() && window.collectionSources_->viewport()->rect().contains(secondRect.center()),
            "second dataset is not available for an actual viewport click");
    mouseClick(window.collectionSources_->viewport(), secondRect.center());
    require(TrainerWindow::selectedPath(window.datasets_) == second,
            "clicking the second preparation dataset left library actions on the first dataset");
    auto *useTraining = window.findChild<QPushButton *>("useDatasetForTraining");
    require(useTraining && useTraining->isVisible() && useTraining->isEnabled(),
            "selected dataset has no usable training assignment button");
    mouseClick(useTraining, useTraining->rect().center());
    require(TrainerWindow::collectionRole(lowerSecond) == "train" &&
            TrainerWindow::collectionRole(window.collectionSources_->topLevelItem(0)) == "unused" &&
            window.compose_->isEnabled(),
            "clicking Use for training failed to include only the second dataset or enable composition");
    require(lowerSecond->text(1) == "Training", "second dataset did not display its training role");
    {
        QSignalBlocker blocker(window.process_);
        window.compose_->click();
        const QStringList arguments = window.process_->arguments();
        require(arguments.contains("compose-datasets") && arguments.contains(second) && !arguments.contains(first),
                "creating the training set ignored the second row's button assignment");
        if (window.process_->state() != QProcess::NotRunning) {
            window.process_->kill(); window.process_->waitForFinished(1500);
        }
        window.stdout_ = "{\"error\":\"isolated composition fixture stopped\"}";
        window.processFinished(1, QProcess::NormalExit);
    }
    window.populateLibrary(library);
    require(TrainerWindow::selectedPath(window.datasets_) == second &&
            TrainerWindow::selectedPath(window.collectionSources_) == second,
            "refresh changed the active dataset between library and preparation");
    require(TrainerWindow::collectionRole(window.collectionSources_->topLevelItem(1)) == "train" &&
            TrainerWindow::collectionRole(window.collectionSources_->topLevelItem(0)) == "unused",
            "refresh lost the training role while preserving active selection");
    auto *skip = window.findChild<QPushButton *>("skipDataset");
    require(skip && skip->isVisible() && skip->isEnabled(), "selected dataset has no usable skip button");
    mouseClick(skip, skip->rect().center());
    require(TrainerWindow::collectionRole(window.collectionSources_->topLevelItem(1)) == "unused" &&
            !window.compose_->isEnabled(), "Do not use failed to remove the second dataset from composition");
    window.collectionSources_->topLevelItem(0)->setSelected(true);
    require(window.collectionSources_->selectedItems().size() == 2, "bulk role fixture did not select both datasets");
    mouseClick(useTraining, useTraining->rect().center());
    require(window.collectionSources_->selectedItems().size() == 2 &&
            TrainerWindow::collectionRole(window.collectionSources_->topLevelItem(0)) == "train" &&
            TrainerWindow::collectionRole(window.collectionSources_->topLevelItem(1)) == "train",
            "bulk Use for training lost a selected dataset or failed to assign both rows");
    mouseClick(skip, skip->rect().center());
    require(window.collectionSources_->selectedItems().size() == 2 &&
            TrainerWindow::collectionRole(window.collectionSources_->topLevelItem(0)) == "unused" &&
            TrainerWindow::collectionRole(window.collectionSources_->topLevelItem(1)) == "unused",
            "bulk Do not use changed only the final row from the preceding assignment");
    requireNoItemCheckboxes(window.collectionSources_);
    window.datasets_->setCurrentItem(window.datasets_->topLevelItem(0));
    require(TrainerWindow::selectedPath(window.collectionSources_) == first,
            "selecting the library dataset did not update preparation's active dataset");
}

void ordinaryGenerationButtonsAndLinks(TrainerWindow &window) {
    for (auto *button : {window.advanced_, window.useGroups_, window.verifiedScenes_,
                         window.anchor_, window.compareTeachers_, window.includeDisplayTeacher_})
        require(button && button->isCheckable(), "a generation or review option is no longer an ordinary toggle button");
    for (auto *button : window.teacherChecks_)
        require(button && button->isCheckable(), "teacher selection no longer uses ordinary toggle buttons");
    const bool groups = window.useGroups_->isChecked();
    window.useGroups_->click();
    require(window.useGroups_->isChecked() != groups &&
            window.useGroups_->text().contains(window.useGroups_->isChecked() ? "On" : "Off"),
            "group toggle button failed to display its enabled or disabled state");
    window.useGroups_->click();
    require(window.useGroups_->isChecked() == groups, "group toggle button could not return to its original state");

    require(window.projectSettings_ != nullptr, "dataset links fixture has no project settings");
    const QVariant previous = window.projectSettings_->value("dataset_links");
    window.projectSettings_->setValue("dataset_links", QStringList{"/fixture/first-link", "/fixture/second-link"});
    window.projectSettings_->sync();
    bool sawDialog = false, ordinaryRows = false, noCheckboxWidgets = false;
    QTimer::singleShot(0, &window, [&] {
        auto *dialog = window.findChild<QDialog *>("datasetLinksDialog");
        if (!dialog) dialog = qobject_cast<QDialog *>(QApplication::activeModalWidget());
        sawDialog = dialog != nullptr;
        if (!dialog) return;
        auto *list = dialog->findChild<QListWidget *>("datasetLinks");
        if (!list) list = dialog->findChild<QListWidget *>();
        ordinaryRows = list && list->count() == 2;
        if (list) for (int i = 0; i < list->count(); ++i) {
            const auto *item = list->item(i);
            ordinaryRows = ordinaryRows && !(item->flags() & Qt::ItemIsUserCheckable) &&
                           !item->data(Qt::CheckStateRole).isValid();
        }
        noCheckboxWidgets = dialog->findChildren<QCheckBox *>().isEmpty();
        dialog->reject();
    });
    window.editDatasetLinks();
    if (previous.isValid()) window.projectSettings_->setValue("dataset_links", previous);
    else window.projectSettings_->remove("dataset_links");
    window.projectSettings_->sync();
    require(sawDialog && ordinaryRows && noCheckboxWidgets,
            "link management still requires native checkboxes to choose datasets");
}

void photoEditsAndDrafts(TrainerWindow &window, const QString &workspace) {
    require(window.reviewSamples_->selectionMode() == QAbstractItemView::ExtendedSelection,
            "photo management does not allow selecting multiple photos");
    const QString first = QDir(workspace).filePath("datasets/first");
    const QString second = QDir(workspace).filePath("datasets/second");
    QDir().mkpath(first); QDir().mkpath(second);
    const QString manifest = QDir(first).filePath("dataset.json");
    QFile file(manifest); require(file.open(QIODevice::WriteOnly), "cannot create fixture manifest");
    file.write(QJsonDocument(review(first)).toJson()); file.close();
    const QByteArray original = readBytes(manifest);
    populateFixtureReview(window, review(first));
    auto *photo = window.reviewSamples_->topLevelItem(0);
    selectOnly(window.reviewSamples_, photo);
    window.setReviewIncluded(false);
    require(!TrainerWindow::reviewIncluded(photo->child(0)) &&
            !TrainerWindow::reviewIncluded(photo->child(1)) && photo->text(5) == "Removed",
            "removing a selected photo retained one of its teacher entries");
    window.setReviewIncluded(true);
    selectOnly(window.reviewSamples_, photo->child(0));
    window.setReviewIncluded(false);
    require(!TrainerWindow::reviewIncluded(photo->child(0)) &&
            TrainerWindow::reviewIncluded(photo->child(1)) && photo->text(5) == "Some removed",
            "removing one selected teacher changed the other teacher for that photo");
    window.setReviewSplit("validation");
    require(photo->child(0)->text(1) == "validation" && photo->child(1)->text(1) == "validation" &&
            window.reviewSamples_->topLevelItem(1)->child(0)->text(1) == "validation",
            "moving a photo to validation split its related capture group or teachers");
    QJsonObject edits = window.selectedReviewEdits();
    require(!edits.value("keep").toArray().contains("shared-1") &&
            edits.value("keep").toArray().contains("shared-2") &&
            edits.value("splits").toObject().value("related").toString() == "validation",
            "save payload lost the selected teacher exclusion or group split edits");
    require(readBytes(manifest) == original, "a paused metadata fixture changed its source before autosave ran");
    require(window.hasPendingDatasetEdits() && QFileInfo::exists(QDir(workspace).filePath(".dataset-studio-edits.json")),
            "photo edits were not recorded durably before their autosave began");

    populateFixtureReview(window, review(second));
    require(TrainerWindow::reviewIncluded(window.reviewSamples_->topLevelItem(0)->child(0)) &&
            window.reviewSamples_->topLevelItem(0)->child(0)->text(1) == "train",
            "opening another dataset with shared sample IDs inherited the first dataset's draft");
    require(window.reviewDrafts_.contains(first), "switching datasets discarded the first dataset's draft");
    populateFixtureReview(window, review(first));
    require(!TrainerWindow::reviewIncluded(window.reviewSamples_->topLevelItem(0)->child(0)) &&
            window.reviewSamples_->topLevelItem(0)->child(0)->text(1) == "validation" &&
            window.reviewSamples_->topLevelItem(1)->child(0)->text(1) == "validation",
            "returning to a dataset did not restore its photo and split edits");
    require(readBytes(manifest) == original, "restoring a paused draft changed its array-free fixture manifest");

    // A bulk selection applies to complete photos and all their teacher targets.
    window.reviewSamples_->clearSelection();
    window.reviewSamples_->topLevelItem(0)->setSelected(true);
    window.reviewSamples_->topLevelItem(1)->setSelected(true);
    window.setReviewIncluded(false);
    require(window.selectedReviewEdits().value("keep").toArray() == QJsonArray{"held-out"},
            "bulk photo removal did not retain only the independent photo");
    window.setReviewIncluded(true);
    require(window.selectedReviewEdits().value("keep").toArray().size() == 4,
            "bulk photo restoration failed to restore every selected teacher");
    require(window.reviewSamples_->topLevelItem(0)->text(5) == "Included" && !window.reviewSamples_->isColumnHidden(5),
            "photo management hides the inclusion status needed in place of checkboxes");
    requireNoItemCheckboxes(window.reviewSamples_);
}

void filteredNavigation(TrainerWindow &window, const QString &workspace) {
    populateFixtureReview(window, review(QDir(workspace).filePath("datasets/navigation")));
    window.reviewFilter_->setText("one");
    auto *first = window.reviewSamples_->topLevelItem(0)->child(1);
    auto *hidden = window.reviewSamples_->topLevelItem(1);
    require(hidden->isHidden(), "filter did not hide an unrelated photo");
    selectOnly(window.reviewSamples_, first);
    window.advanceReview(1);
    require(window.selectedReviewEntry() == first,
            "Next navigated into a photo hidden by the active filter");
    window.advanceReview(-1);
    require(window.selectedReviewEntry() == window.reviewSamples_->topLevelItem(0)->child(0),
            "Previous skipped a visible teacher entry");
    window.reviewFilter_->clear();
}

void streamingAndUngroupedPhotos(TrainerWindow &window, const QString &workspace) {
    const QString path = QDir(workspace).filePath("datasets/generating");
    QJsonObject generating{{"dataset_path", path}, {"generation_state", "generating"},
                           {"samples", QJsonArray{sample("initial", "initial-photo", "first-group", "Teacher")}}};
    populateFixtureReview(window, generating);
    selectOnly(window.reviewSamples_, window.reviewSamples_->topLevelItem(0)->child(0));
    window.setReviewIncluded(false);
    require(TrainerWindow::reviewIncluded(window.reviewSamples_->topLevelItem(0)->child(0)),
            "an incomplete generating dataset allowed photo edits before generation finished");
    auto *removeNext = window.findChild<QPushButton *>("removePhotoAndNext");
    require(removeNext && !removeNext->isEnabled(), "generation left Remove and next enabled");
    removeNext->clicked();
    require(TrainerWindow::reviewIncluded(window.reviewSamples_->topLevelItem(0)->child(0)),
            "Remove and next's signal bypassed the incomplete-generation guard");
    // Seed an existing draft exclusion directly to verify that streaming refresh
    // preserves it; editing a still-generating dataset is intentionally disabled.
    {
        QSignalBlocker blocker(window.reviewSamples_);
        TrainerWindow::setReviewItemIncluded(window.reviewSamples_->topLevelItem(0)->child(0), false);
    }
    window.updateReviewCount();
    window.rememberReviewDraft();
    auto recoveredDraft = window.reviewDrafts_.value(path); recoveredDraft.insert("dirty", true);
    recoveredDraft.insert("serial", ++window.editSerial_); window.reviewDrafts_.insert(path, recoveredDraft);
    require(window.persistReviewDrafts(), "cannot persist the seeded recoverable streaming draft");
    generating.insert("samples", QJsonArray{sample("initial", "initial-photo", "first-group", "Teacher"),
                                            sample("new", "new-photo", "new-group", "Teacher")});
    populateFixtureReview(window, generating);
    require(!TrainerWindow::reviewIncluded(window.reviewSamples_->topLevelItem(0)->child(0)) &&
            TrainerWindow::reviewIncluded(window.reviewSamples_->topLevelItem(1)->child(0)),
            "streaming refresh either lost an existing exclusion or excluded a newly generated photo");
    const QString ungrouped = QDir(workspace).filePath("datasets/ungrouped");
    populateFixtureReview(window, {{"dataset_path", ungrouped}, {"samples", QJsonArray{
        sample("ungrouped-a", "ungrouped-a", "", "Teacher"),
        sample("ungrouped-b", "ungrouped-b", "", "Teacher")}}});
    selectOnly(window.reviewSamples_, window.reviewSamples_->topLevelItem(0));
    window.setReviewSplit("validation");
    require(window.reviewSamples_->topLevelItem(0)->child(0)->text(1) == "validation" &&
            window.reviewSamples_->topLevelItem(1)->child(0)->text(1) == "train",
            "moving an ungrouped photo changed another unrelated photo with an empty group ID");
}

void asynchronousDatasetSwitch(TrainerWindow &window, const QString &workspace) {
    const QString first = QDir(workspace).filePath("datasets/async-first");
    const QString second = QDir(workspace).filePath("datasets/async-second");
    populateFixtureReview(window, review(first));
    selectOnly(window.reviewSamples_, window.reviewSamples_->topLevelItem(0));
    window.setReviewIncluded(false);
    window.previewRecords_ = QJsonArray{QJsonObject{{"teacher_id", "old teacher"}}};
    window.differencePath_ = "/fixture/old-disagreement.png";
    window.depthPreviews_.at(1)->setVisible(true);
    {
        // Exercise the loading state without allowing a fixture process result to replace it.
        QSignalBlocker blocker(window.previewProcess_);
        window.reviewDataset(second);
        require(window.reviewEntries().isEmpty(), "switching datasets left editable photos from the old dataset");
        require(window.hasUnsavedReviewExclusions(first),
                "the loading state allowed training an old dataset with unsaved photo changes");
        require(window.previewRecords_.isEmpty() && window.differencePath_.isEmpty() &&
                window.depthPreviews_.at(1)->isHidden(),
                "switching datasets retained the old dataset's preview records or teacher image");
        window.pendingPreviewArgs_.clear(); window.previewKind_ = "discarded";
        if (window.previewProcess_->state() != QProcess::NotRunning) {
            window.previewProcess_->kill(); window.previewProcess_->waitForFinished(1500);
        }
    }
    populateFixtureReview(window, review(second));
    require(window.hasUnsavedReviewExclusions(first),
            "loading the second dataset discarded the unsaved edits for the first dataset");
}

void addedPhotoContinuationAndRetry(TrainerWindow &window, const QString &workspace) {
    clearPendingFixtureEdits(window);
    const QString base = QDir(workspace).filePath("datasets/continuation-base");
    const QString addition = QDir(workspace).filePath("datasets/generated-addition");
    writeFixtureManifest(base, review(base));
    window.requestedReviewPath_ = base; window.populateReview(review(base));
    selectOnly(window.reviewSamples_, window.reviewSamples_->topLevelItem(0)->child(0));
    window.setReviewIncluded(false); window.setReviewSplit("validation");
    const QJsonObject edits = window.selectedReviewEdits();
    require(edits.value("splits").toObject().value("shared-1").toString() == "validation" &&
            edits.value("splits").toObject().value("shared-2").toString() == "validation",
            "a removed teacher lost its linked capture's saved split assignment");
    window.rememberReviewDraft();
    window.appendBase_ = base; window.appendEdits_ = edits; window.appendVersionName_ = "continuation-version";
    {
        QSignalBlocker generationSignals(window.process_), editSignals(window.editProcess_);
        window.job_ = "Generate dataset"; window.refreshAfter_ = true;
        window.stdout_ = QJsonDocument(QJsonObject{{"dataset_path", addition}}).toJson(); window.setBusy(true);
        window.processFinished(0, QProcess::NormalExit);
        window.startNextDatasetSave();
        const QStringList arguments = window.editProcess_->arguments();
        const int operation = arguments.indexOf("update-dataset"), addArgument = arguments.indexOf("--add-dataset");
        const int editsArgument = arguments.indexOf("--edits-json");
        require(window.editingDataset_ == base && operation >= 0 && arguments.value(operation + 1) == base &&
                addArgument >= 0 && arguments.value(addArgument + 1) == addition && !arguments.contains("--output-dir"),
                "adding generated photos created another dataset or restarted existing inference");
        require(editsArgument >= 0 && readJson(arguments.value(editsArgument + 1)) == edits,
                "adding generated photos lost the base dataset's kept entries or split edits");
        require(window.appendBase_.isEmpty() && window.reviewAdditions_.value(base).contains(addition),
                "add-photo continuation lost the generated additions needed for saving or retry");
        selectOnly(window.reviewSamples_, window.reviewSamples_->topLevelItem(1));
        const int snapshotSerial = window.savingDraft_.value("serial").toInt();
        window.setReviewIncluded(false);
        require(window.reviewDrafts_.value(base).value("serial").toInt() > snapshotSerial,
                "a new edit while metadata was saving failed to retain a newer draft");
        stopEditProcess(window);
        window.editStdout_ = "{\"error\":\"isolated save fixture failed\"}";
        window.finishDatasetSave(1, QProcess::NormalExit);
        require(window.reviewAdditions_.value(base).contains(addition) && window.reviewDrafts_.contains(base),
                "a failed in-place save discarded generated additions or photo edits needed to retry");
        window.retryDatasetSaves(); stopEditProcess(window);
        QJsonObject saved = review(base); auto samples = saved.value("samples").toArray();
        const auto keep = window.savingDraft_.value("keep").toArray();
        for (int i=0; i<samples.size(); ++i) { auto sample = samples[i].toObject(); sample.insert("excluded", !keep.contains(sample.value("id"))); sample.insert("split", "validation"); samples[i] = sample; }
        saved.insert("samples", samples); completeMetadataSave(window, base, saved);
        require(!window.reviewAdditions_.contains(base) && !window.reviewDrafts_.contains(base) &&
                !window.reviewEntries().isEmpty() && window.reviewedDataset_ == base &&
                !window.hasUnsavedReviewExclusions(base),
                "a successful in-place save lost the active dataset, retained a stale dirty draft, or hid saved exclusions");
    }
    window.pendingReview_.clear();
}

void explicitSameSplitChoice(TrainerWindow &window, const QString &workspace) {
    const QString first = QDir(workspace).filePath("datasets/explicit-same-split");
    const QString second = QDir(workspace).filePath("datasets/other-split-fixture");
    window.requestedReviewPath_ = first; window.populateReview(review(first));
    selectOnly(window.reviewSamples_, window.reviewSamples_->topLevelItem(0));
    window.setReviewSplit("train");
    require(window.selectedReviewEdits().value("splits").toObject().value("shared-1").toString() == "train" &&
            window.hasUnsavedReviewExclusions(first),
            "choosing the existing split discarded an explicit choice needed to resolve merged group conflicts");
    populateFixtureReview(window, review(second)); populateFixtureReview(window, review(first));
    require(window.selectedReviewEdits().value("splits").toObject().value("shared-1").toString() == "train",
            "switching datasets discarded an explicit choice that matched the source split");
}

void replacingAllPhotos(TrainerWindow &window, const QString &workspace) {
    clearPendingFixtureEdits(window);
    const QString base = QDir(workspace).filePath("datasets/replace-all-base");
    const QString addition = QDir(workspace).filePath("datasets/replace-all-addition");
    window.requestedReviewPath_ = base; window.populateReview(review(base));
    window.reviewSamples_->selectAll();
    window.setReviewIncluded(false);
    require(window.selectedReviewEdits().value("keep").toArray().isEmpty(),
            "remove-all fixture still has included base photos");
    require(window.addPhotos_->isEnabled(), "removing all old photos disabled adding their replacements");
    window.reviewAdditions_.insert(base, {addition}); window.updateReviewCount();
    require(window.saveReviewed_->isEnabled(), "replacement photos could not be saved with every old photo removed");
    {
        QSignalBlocker blocker(window.editProcess_); window.saveReviewedDataset(); window.startNextDatasetSave();
        const QStringList arguments = window.editProcess_->arguments();
        const int editsArgument = arguments.indexOf("--edits-json"), additionArgument = arguments.indexOf("--add-dataset");
        require(window.editingDataset_ == base && arguments.contains("update-dataset") && !arguments.contains("--output-dir") && editsArgument >= 0 &&
                readJson(arguments.value(editsArgument + 1)).value("keep").toArray().isEmpty() &&
                additionArgument >= 0 && arguments.value(additionArgument + 1) == addition,
                "saving replacement photos omitted the addition or restored removed base photos");
        stopEditProcess(window);
        window.editStdout_ = "{\"error\":\"isolated replacement fixture stopped\"}";
        window.finishDatasetSave(1, QProcess::NormalExit);
        require(window.reviewAdditions_.value(base).contains(addition) && window.saveReviewed_->isEnabled(),
                "a failed replacement save lost the new photos or disabled retry with an empty base selection");
        window.saveReviewedDataset();
        window.startNextDatasetSave(); const QStringList retryArguments = window.editProcess_->arguments();
        const int retryEdits = retryArguments.indexOf("--edits-json"), retryAddition = retryArguments.indexOf("--add-dataset");
        require(retryEdits >= 0 && readJson(retryArguments.value(retryEdits + 1)).value("keep").toArray().isEmpty() &&
                retryAddition >= 0 && retryArguments.value(retryAddition + 1) == addition,
                "retrying a replacement save lost its new photos or restored removed base photos");
        stopEditProcess(window);
        window.editStdout_ = "{\"error\":\"isolated replacement retry stopped\"}";
        window.finishDatasetSave(1, QProcess::NormalExit);
    }
}

void saveCompletionPreservesNewerEdits(TrainerWindow &window, const QString &workspace) {
    clearPendingFixtureEdits(window);
    const QString path = QDir(workspace).filePath("datasets/newer-edit-during-save"); writeFixtureManifest(path, review(path));
    populateFixtureReview(window, review(path)); selectOnly(window.reviewSamples_, window.reviewSamples_->topLevelItem(0)->child(0));
    window.setReviewIncluded(false);
    QSignalBlocker editSignals(window.editProcess_); window.startNextDatasetSave();
    const int firstSerial = window.savingDraft_.value("serial").toInt();
    window.setReviewIncluded(true);
    require(window.reviewDrafts_.value(path).value("serial").toInt() > firstSerial,
            "restoring a target while a previous removal saved did not retain the newer change");
    stopEditProcess(window); auto older = review(path); auto olderSamples = older.value("samples").toArray();
    auto removed = olderSamples[0].toObject(); removed.insert("excluded", true); removed.insert("included", false); olderSamples[0] = removed;
    older.insert("samples", olderSamples); completeMetadataSave(window, path, older);
    require(window.hasPendingDatasetEdits() && TrainerWindow::reviewIncluded(window.reviewSamples_->topLevelItem(0)->child(0)) &&
            window.reviewDrafts_.value(path).value("keep").toArray().contains("shared-1") &&
            window.reviewDrafts_.value(path).value("manifest_sha256").toString() == window.manifestHash(path),
            "an earlier completed removal erased the newer restore or kept a stale manifest hash");
    window.startNextDatasetSave(); stopEditProcess(window); completeMetadataSave(window, path, review(path));
    require(!window.hasPendingDatasetEdits() && !window.hasUnsavedReviewExclusions(path) &&
            TrainerWindow::reviewIncluded(window.reviewSamples_->topLevelItem(0)->child(0)),
            "the newer restore did not become the clean saved dataset after its own save");
}

void durableRecoveryAndCloseChoices(const QString &workspace) {
    const QString isolated = QDir(workspace).filePath("recovery-workspace");
    const QString path = QDir(isolated).filePath("datasets/recoverable");
    writeFixtureManifest(path, review(path));
    TrainerWindow owner(true); owner.autosavePaused_ = true;
    { QSignalBlocker blocker(owner.workspace_); owner.workspace_->setText(isolated); }
    owner.loadReviewDrafts(); owner.requestedReviewOpen_ = false; owner.tabs_->setCurrentIndex(2);
    owner.requestedReviewPath_ = path; owner.populateReview(review(path));
    selectOnly(owner.reviewSamples_, owner.reviewSamples_->topLevelItem(0));
    owner.setReviewIncluded(false); owner.setReviewSplit("validation");
    const auto journal = readJson(QDir(isolated).filePath(".dataset-studio-edits.json"));
    const auto diskDraft = journal.value("datasets").toObject().value(path).toObject();
    require(diskDraft.value("dirty").toBool() && !diskDraft.value("keep").toArray().contains("shared-1") &&
            diskDraft.value("all_splits").toObject().value("related").toString() == "validation",
            "pending photo and capture split edits were not durable before the background save");

    bool warningSeen = false;
    QTimer::singleShot(0, &owner, [&] {
        auto *dialog = owner.findChild<QMessageBox *>("pendingDatasetEditsWarning");
        warningSeen = dialog != nullptr;
        if (!dialog) return;
        for (auto *button : dialog->findChildren<QPushButton *>()) if (button->text() == "Cancel") { button->click(); return; }
        dialog->reject();
    });
    QCloseEvent cancelClose; QApplication::sendEvent(&owner, &cancelClose);
    require(warningSeen && !cancelClose.isAccepted() && owner.hasPendingDatasetEdits(),
            "closing silently discarded queued edits instead of letting the user keep editing");
    bool recoveryChoiceSeen = false;
    QTimer::singleShot(0, &owner, [&] {
        auto *dialog = owner.findChild<QMessageBox *>("pendingDatasetEditsWarning"); if (!dialog) return;
        for (auto *button : dialog->findChildren<QPushButton *>()) if (button->text() == "Close keeping recoverable edits") {
            recoveryChoiceSeen = button->isEnabled(); button->click(); return;
        }
        dialog->reject();
    });
    QCloseEvent recoverableClose; QApplication::sendEvent(&owner, &recoverableClose);
    require(recoveryChoiceSeen && recoverableClose.isAccepted(), "explicit recoverable close was unavailable after journal persistence");
    {
        TrainerWindow reopened(true); reopened.autosavePaused_ = true;
        { QSignalBlocker blocker(reopened.workspace_); reopened.workspace_->setText(isolated); }
        reopened.loadReviewDrafts(); reopened.requestedReviewOpen_ = false; reopened.tabs_->setCurrentIndex(2);
        reopened.requestedReviewPath_ = path; reopened.populateReview(review(path));
        require(!TrainerWindow::reviewIncluded(reopened.reviewSamples_->topLevelItem(0)->child(0)) &&
                !TrainerWindow::reviewIncluded(reopened.reviewSamples_->topLevelItem(0)->child(1)) &&
                reopened.reviewSamples_->topLevelItem(1)->child(0)->text(1) == "validation" &&
                reopened.hasPendingDatasetEdits(),
                "a fresh Dataset Studio window failed to recover the previous window's pending edits");
    }

    const QString blocked = QDir(workspace).filePath("unwritable-journal-workspace"); QDir().mkpath(blocked);
    QDir().mkpath(QDir(blocked).filePath(".dataset-studio-edits.json"));
    TrainerWindow failedJournal(true); failedJournal.autosavePaused_ = true;
    { QSignalBlocker blocker(failedJournal.workspace_); failedJournal.workspace_->setText(blocked); }
    failedJournal.loadReviewDrafts(); failedJournal.requestedReviewOpen_ = false; failedJournal.tabs_->setCurrentIndex(2);
    failedJournal.requestedReviewPath_ = path; failedJournal.populateReview(review(path));
    selectOnly(failedJournal.reviewSamples_, failedJournal.reviewSamples_->topLevelItem(0)); failedJournal.setReviewIncluded(false);
    bool refusedRecoveryClose = false;
    QTimer::singleShot(0, &failedJournal, [&] {
        auto *dialog = failedJournal.findChild<QMessageBox *>("pendingDatasetEditsWarning"); if (!dialog) return;
        for (auto *button : dialog->findChildren<QPushButton *>()) {
            if (button->text() == "Close keeping recoverable edits") refusedRecoveryClose = !button->isEnabled();
            if (button->text() == "Cancel") QTimer::singleShot(0, button, &QPushButton::click);
        }
    });
    QCloseEvent failedRecoveryClose; QApplication::sendEvent(&failedJournal, &failedRecoveryClose);
    require(refusedRecoveryClose && !failedRecoveryClose.isAccepted(), "failed journal persistence falsely offered a recoverable close");

    const QString corruptWorkspace = QDir(workspace).filePath("corrupt-journal-workspace"); QDir().mkpath(corruptWorkspace);
    const QString corruptPath = QDir(corruptWorkspace).filePath(".dataset-studio-edits.json");
    const QByteArray corruptBytes("{\"schema\":\"ipde-dataset-studio-edits-v1\",\"datasets\": unfinished recovery draft");
    { QFile file(corruptPath); require(file.open(QIODevice::WriteOnly), "cannot create corrupt recovery fixture"); file.write(corruptBytes); }
    {
        TrainerWindow unreadable(true); unreadable.autosavePaused_ = true;
        { QSignalBlocker blocker(unreadable.workspace_); unreadable.workspace_->setText(corruptWorkspace); }
        unreadable.loadReviewDrafts();
        require(unreadable.statusBar()->currentMessage().contains("could not be read"), "unreadable recovery journal was silently ignored");
    }
    bool originalPreserved = readBytes(corruptPath) == corruptBytes;
    for (const auto &backup : QDir(corruptWorkspace).entryList({".dataset-studio-edits.json.unreadable-*"}, QDir::Files | QDir::Hidden))
        originalPreserved |= readBytes(QDir(corruptWorkspace).filePath(backup)) == corruptBytes;
    require(originalPreserved, "closing after a journal load failure destroyed the original recovery bytes");
}

void selectedValidationIsSavedBeforeTrainer(TrainerWindow &window, const QString &workspace) {
    clearPendingFixtureEdits(window);
    const QString path = QDir(workspace).filePath("datasets/direct-training-split");
    writeFixtureManifest(path, review(path));
    window.tabs_->setCurrentIndex(2); window.populateLibrary({{"datasets", QJsonArray{libraryDataset(path)}}});
    window.requestedReviewPath_ = path; window.populateReview(review(path)); window.tabs_->setCurrentIndex(2);
    require(window.validationFraction_->minimum() <= 5 && window.validationFraction_->maximum() == 100,
            "validation size cannot represent the requested 5 and 100 percent values");
    auto *openTrainer = window.findChild<QPushButton *>("openDatasetInTrainer");
    require(openTrainer != nullptr, "direct Trainer action is missing");
    QSignalBlocker editSignals(window.editProcess_);
    window.lastAppRequest_ = {}; window.validationFraction_->setValue(5); openTrainer->click(); window.startNextDatasetSave();
    const auto arguments = window.editProcess_->arguments(); const int editsArgument = arguments.indexOf("--edits-json");
    const auto edits = readJson(arguments.value(editsArgument + 1));
    require(arguments.contains("update-dataset") && arguments.contains(path) && !arguments.contains("--output-dir") &&
            edits.value("validation_fraction").toDouble() == 0.05 && window.lastAppRequest_.isEmpty(),
            "direct Trainer action ignored validation size or opened before the selected dataset saved");
    stopEditProcess(window); window.editStdout_ = "{\"error\":\"isolated split update failed\"}";
    window.finishDatasetSave(1, QProcess::NormalExit);
    require(window.lastAppRequest_.isEmpty() && window.hasPendingDatasetEdits(),
            "a failed split update opened Trainer on old assignments or lost its recovery draft");

    window.validationFraction_->setValue(100); openTrainer->click(); window.startNextDatasetSave();
    const auto allArguments = window.editProcess_->arguments();
    const auto allEdits = readJson(allArguments.value(allArguments.indexOf("--edits-json") + 1));
    require(allEdits.value("validation_fraction").toDouble() == 1.0 && allEdits.value("splits").toObject().isEmpty(),
            "100 percent validation was clamped or conflicted with old individual split overrides");
    stopEditProcess(window); auto saved = review(path); auto samples = saved.value("samples").toArray();
    for (int i=0; i<samples.size(); ++i) { auto sample = samples[i].toObject(); sample.insert("split", "validation"); samples[i] = sample; }
    saved.insert("samples", samples); saved.insert("validation_fraction", 1.0); completeMetadataSave(window, path, saved);
    require(window.lastAppRequest_.value("role") == "trainer" && window.lastAppRequest_.value("dataset") == path &&
            !window.hasPendingDatasetEdits(), "a completed same-dataset split failed to open Trainer or retained a stale pending draft");
    for (auto *entry : window.reviewEntries()) require(entry->text(1) == "validation", "saved automatic validation was not reflected in teacher rows");
    for (int i=0; i<window.reviewSamples_->topLevelItemCount(); ++i)
        require(window.reviewSamples_->topLevelItem(i)->text(1) == "validation", "saved automatic validation left a stale photo-row split");
    TrainerWindow cleanReopen(true); cleanReopen.autosavePaused_ = true;
    cleanReopen.requestedReviewOpen_ = false; cleanReopen.tabs_->setCurrentIndex(2); cleanReopen.requestedReviewPath_ = path;
    auto excluded = saved; auto excludedSamples = excluded.value("samples").toArray(); auto removed = excludedSamples[0].toObject();
    removed.insert("excluded", true); removed.insert("included", false); excludedSamples[0] = removed; excluded.insert("samples", excludedSamples);
    cleanReopen.populateReview(excluded);
    require(!TrainerWindow::reviewIncluded(cleanReopen.reviewSamples_->topLevelItem(0)->child(0)) &&
            !cleanReopen.hasUnsavedReviewExclusions(path), "a saved excluded entry reopened as included or falsely dirty");
}

QJsonObject realBackendReview(const QString &path, bool createFixture) {
    const QString repository = QFileInfo(scriptPath()).absolutePath();
    const QString script = "import json,sys;from pathlib import Path;"
        "sys.path.insert(0,sys.argv[1]);sys.path.insert(0,str(Path(sys.argv[1])/'src'));"
        "from ipde.dataset_review import review_dataset;"
        "from verification.test_dataset_edit import _dataset;"
        "root=Path(sys.argv[2]);"
        "_dataset(root) if sys.argv[3]=='create' else None;"
        "print(json.dumps(review_dataset(root)))";
    QProcess process; process.setProgram(pythonPath());
    process.setArguments({"-c", script, repository, path, createFixture ? "create" : "review"}); process.start();
    require(process.waitForFinished(15000) && process.exitStatus() == QProcess::NormalExit && process.exitCode() == 0,
            "valid dataset fixture could not be generated or reviewed by the real backend");
    const auto document = QJsonDocument::fromJson(process.readAllStandardOutput());
    require(document.isObject() && document.object().value("dataset_path").toString() == QFileInfo(path).canonicalFilePath(),
            "real backend review did not identify the generated fixture dataset");
    return document.object();
}

void realAutosaveReopenAndSplit(const QString &workspace) {
    const QString isolated = QDir(workspace).filePath("real-autosave-workspace");
    const QString requestedPath = QDir(isolated).filePath("datasets/live"); QDir().mkpath(QFileInfo(requestedPath).absolutePath());
    const auto initial = realBackendReview(requestedPath, true);
    const QString path = initial.value("dataset_path").toString();
    QMap<QString, QByteArray> payloads;
    QDirIterator files(path, {"*.npy", "*.npz"}, QDir::Files, QDirIterator::Subdirectories);
    while (files.hasNext()) { const QString file = files.next(); payloads.insert(file, readBytes(file)); }
    require(!payloads.isEmpty(), "real autosave fixture contains no scientific array files");

    TrainerWindow live(true); live.autosavePaused_ = true;
    { QSignalBlocker blocker(live.workspace_); live.workspace_->setText(isolated); }
    live.loadReviewDrafts(); live.requestedReviewOpen_ = false; live.tabs_->setCurrentIndex(2);
    live.populateLibrary({{"datasets", QJsonArray{libraryDataset(path)}}});
    live.requestedReviewPath_ = path; live.populateReview(initial); live.tabs_->setCurrentIndex(2);
    selectOnly(live.reviewSamples_, live.reviewSamples_->topLevelItem(0)); live.autosavePaused_ = false;
    live.setReviewIncluded(false);
    require(readJson(QDir(isolated).filePath(".dataset-studio-edits.json")).value("datasets").toObject().value(path).toObject().value("dirty").toBool(),
            "real photo removal did not write recovery state before autosave");
    require(await([&] { return !live.hasPendingDatasetEdits(); }), "real metadata autosave did not finish successfully");
    const auto saved = readJson(QDir(path).filePath("dataset.json"));
    require(saved.value("samples").toArray().at(0).toObject().value("excluded").toBool() && saved.value("edit_revision").toInt() >= 1 &&
            live.reviewSaveStatus_->text().contains("saved", Qt::CaseInsensitive),
            "real autosave failed to update the same dataset's saved membership");
    for (auto it=payloads.cbegin(); it!=payloads.cend(); ++it) require(readBytes(it.key()) == it.value(), "real autosave rewrote an image or depth array");
    {
        TrainerWindow reopened(true); reopened.autosavePaused_ = true;
        { QSignalBlocker blocker(reopened.workspace_); reopened.workspace_->setText(isolated); }
        reopened.loadReviewDrafts(); reopened.requestedReviewOpen_ = false; reopened.tabs_->setCurrentIndex(2);
        reopened.requestedReviewPath_ = path; reopened.populateReview(realBackendReview(path, false));
        require(!TrainerWindow::reviewIncluded(reopened.reviewSamples_->topLevelItem(0)->child(0)) &&
                reopened.reviewSamples_->topLevelItem(0)->text(5) == "Removed" && !reopened.hasPendingDatasetEdits() &&
                !reopened.hasUnsavedReviewExclusions(path), "a real saved exclusion was lost or falsely dirty in a fresh window");
    }
    live.lastAppRequest_ = {}; live.validationFraction_->setValue(5); live.splitSeed_->setValue(84785);
    require(await([&] { return !live.hasPendingDatasetEdits(); }), "changing the validation controls did not autosave without opening Trainer");
    require(live.lastAppRequest_.isEmpty() && readJson(QDir(path).filePath("dataset.json")).value("validation_fraction").toDouble() == 0.05 &&
            readJson(QDir(path).filePath("dataset.json")).value("split_seed").toInt() == 84785,
            "validation controls failed to save their percentage and seed independently of the Trainer action");
    live.applySplitAndOpenTrainer();
    require(await([&] { return !live.hasPendingDatasetEdits() && live.lastAppRequest_.value("role") == "trainer"; }),
            "real five-percent split did not save before opening Trainer");
    const auto fivePercent = readJson(QDir(path).filePath("dataset.json"));
    require(fivePercent.value("validation_fraction").toDouble() == 0.05 &&
            fivePercent.value("summary").toObject().value("train_samples").toInt() > 0 &&
            fivePercent.value("summary").toObject().value("validation_samples").toInt() > 0,
            "real five-percent validation was ignored or lost one side of its included group split");
    live.validationFraction_->setValue(100); live.lastAppRequest_ = {}; live.applySplitAndOpenTrainer();
    require(await([&] { return !live.hasPendingDatasetEdits() && live.lastAppRequest_.value("role") == "trainer"; }),
            "real automatic split save did not complete before opening Trainer");
    const auto final = readJson(QDir(path).filePath("dataset.json"));
    require(final.value("validation_fraction").toDouble() == 1.0 && live.lastAppRequest_.value("dataset") == path,
            "real direct Trainer action changed dataset paths or ignored 100 percent validation");
    for (const auto &value : final.value("samples").toArray())
        if (!value.toObject().value("excluded").toBool()) require(value.toObject().value("split").toString() == "validation", "100 percent validation retained an included training sample");
    const auto eligibility = live.datasets_->currentItem()->data(0, Qt::UserRole + 1).toJsonObject().value("training_eligibility").toObject();
    require(!eligibility.value("trainable").toBool(true) && eligibility.value("train_count").toInt(-1) == 0,
            "real metadata save presented all-validation data as trainable");
    for (auto it=payloads.cbegin(); it!=payloads.cend(); ++it) require(readBytes(it.key()) == it.value(), "real split save rewrote scientific array bytes");
    {
        TrainerWindow reopened(true); reopened.autosavePaused_ = true;
        { QSignalBlocker blocker(reopened.workspace_); reopened.workspace_->setText(isolated); }
        reopened.loadReviewDrafts(); reopened.requestedReviewOpen_ = false; reopened.tabs_->setCurrentIndex(2);
        reopened.populateLibrary({{"datasets", QJsonArray{libraryDataset(path)}}});
        require(reopened.validationFraction_->value() == 100.0 && reopened.splitSeed_->value() == 84785 &&
                !reopened.hasPendingDatasetEdits(), "reopening reset the saved validation percentage or split seed, or queued an unintended edit");
    }
}

void realInterruptedAutosaveRecovery(const QString &workspace) {
    for (bool newerEdit : {false, true}) {
        const QString isolated = QDir(workspace).filePath(newerEdit ? "interrupted-newer-save" : "interrupted-completed-save");
        const QString requestedPath = QDir(isolated).filePath("datasets/live"); QDir().mkpath(QFileInfo(requestedPath).absolutePath());
        const auto initial = realBackendReview(requestedPath, true); const QString path = initial.value("dataset_path").toString();
        QMap<QString, QByteArray> payloads; QDirIterator files(path, {"*.npy", "*.npz"}, QDir::Files, QDirIterator::Subdirectories);
        while (files.hasNext()) { const auto file = files.next(); payloads.insert(file, readBytes(file)); }
        {
            TrainerWindow owner(true); owner.autosavePaused_ = true;
            { QSignalBlocker blocker(owner.workspace_); owner.workspace_->setText(isolated); }
            owner.loadReviewDrafts(); owner.requestedReviewOpen_ = false; owner.tabs_->setCurrentIndex(2);
            owner.populateLibrary({{"datasets", QJsonArray{libraryDataset(path)}}});
            owner.requestedReviewPath_ = path; owner.populateReview(initial); owner.tabs_->setCurrentIndex(2);
            selectOnly(owner.reviewSamples_, owner.reviewSamples_->topLevelItem(0)); owner.setReviewIncluded(false);
            QSignalBlocker lostCompletion(owner.editProcess_); owner.startNextDatasetSave();
            const auto attempted = readJson(QDir(isolated).filePath(".dataset-studio-edits.json")).value("datasets").toObject().value(path).toObject().value("in_flight").toObject();
            require(!attempted.value("operation_id").toString().isEmpty() && attempted.value("serial") == owner.savingDraft_.value("serial"),
                    "the in-flight save snapshot was not durable before the backend started");
            if (newerEdit) owner.setReviewIncluded(true);
            require(owner.editProcess_->waitForFinished(15000) && owner.editProcess_->exitCode() == 0,
                    "the interrupted-save fixture did not commit successfully");
            require(readJson(QDir(path).filePath("dataset.json")).value("edit_revision").toInt() == 1 && owner.hasPendingDatasetEdits(),
                    "the fixture did not interrupt a committed save before its GUI acknowledgement");
            if (!newerEdit) {
                owner.editProcess_->readAllStandardOutput(); owner.editStdout_ = "{\"error\":\"cancelled immediately after commit\"}";
                owner.finishDatasetSave(1, QProcess::NormalExit);
                require(!owner.reviewDrafts_.value(path).value("in_flight").toObject().isEmpty(),
                        "an error immediately after commit discarded the operation needed for recovery");
            }
            owner.autosaveTimer_->stop(); require(owner.persistReviewDrafts(), "cannot preserve the interrupted-save journal");
        }
        TrainerWindow reopened(true); reopened.autosavePaused_ = true;
        { QSignalBlocker blocker(reopened.workspace_); reopened.workspace_->setText(isolated); }
        reopened.loadReviewDrafts(); reopened.requestedReviewOpen_ = false; reopened.tabs_->setCurrentIndex(2);
        reopened.populateLibrary({{"datasets", QJsonArray{libraryDataset(path)}}});
        reopened.requestedReviewPath_ = path; reopened.populateReview(realBackendReview(path, false)); reopened.tabs_->setCurrentIndex(2);
        reopened.autosavePaused_ = false; reopened.retryDatasetSaves();
        require(await([&] { return !reopened.hasPendingDatasetEdits(); }),
                "reopening could not recover a save committed before its GUI acknowledgement");
        const auto final = readJson(QDir(path).filePath("dataset.json"));
        require(final.value("edit_revision").toInt() == (newerEdit ? 2 : 1) &&
                final.value("samples").toArray().at(0).toObject().value("excluded").toBool() == !newerEdit,
                "interrupted-save replay duplicated a commit or lost the newer edit");
        require(readJson(QDir(isolated).filePath(".dataset-studio-edits.json")).value("datasets").toObject().isEmpty(),
                "the recovered save left a stale dirty journal");
        for (auto it=payloads.cbegin(); it!=payloads.cend(); ++it) require(readBytes(it.key()) == it.value(), "interrupted-save recovery changed scientific array bytes");
    }
}
} // namespace

int main(int argc, char **argv) {
    Q_UNUSED(argc);
    QTemporaryDir temporary;
    require(temporary.isValid(), "cannot create dataset management fixture");
    QByteArray project = temporary.path().toUtf8();
    char projectOption[] = "--project", smoke[] = "--smoke-test";
    char *arguments[] = {argv[0], projectOption, project.data(), smoke, nullptr}; int count = 4;
    QApplication application(count, arguments); application.setQuitOnLastWindowClosed(false);
    QSettings::setDefaultFormat(QSettings::IniFormat);
    QSettings::setPath(QSettings::IniFormat, QSettings::UserScope, temporary.path());
    try {
        TrainerWindow window(true); window.autosavePaused_ = true;
        const QString workspace = QDir(temporary.path()).filePath("workspace");
        QDir().mkpath(workspace); window.workspace_->setText(workspace);
        librarySelection(window, workspace);
        ordinaryGenerationButtonsAndLinks(window);
        photoEditsAndDrafts(window, workspace);
        filteredNavigation(window, workspace);
        streamingAndUngroupedPhotos(window, workspace);
        asynchronousDatasetSwitch(window, workspace);
        addedPhotoContinuationAndRetry(window, workspace);
        explicitSameSplitChoice(window, workspace);
        replacingAllPhotos(window, workspace);
        saveCompletionPreservesNewerEdits(window, workspace);
        durableRecoveryAndCloseChoices(workspace);
        selectedValidationIsSavedBeforeTrainer(window, workspace);
        realAutosaveReopenAndSplit(workspace);
        realInterruptedAutosaveRecovery(workspace);
        const QString screenshot = qEnvironmentVariable("IPDE_DATASET_SCREENSHOT");
        if (!screenshot.isEmpty()) {
            TrainerWindow shownWindow(true); shownWindow.autosavePaused_ = true;
            const QString shownWorkspace = QDir(workspace).filePath("screenshots"); QDir().mkpath(shownWorkspace);
            { QSignalBlocker blocker(shownWindow.workspace_); shownWindow.workspace_->setText(shownWorkspace); }
            shownWindow.loadReviewDrafts(); shownWindow.requestedReviewOpen_ = false; shownWindow.tabs_->setCurrentIndex(2);
            const QString shown = QDir(shownWorkspace).filePath("datasets/first"), second = QDir(shownWorkspace).filePath("datasets/second");
            auto photos = review(shown); auto samples = photos.value("samples").toArray();
            for (int i=0; i<2; ++i) { auto entry = samples[i].toObject(); entry.insert("excluded", true); entry.insert("included", false); samples[i] = entry; }
            photos.insert("samples", samples); writeFixtureManifest(shown, photos);
            auto secondManifest = review(second); secondManifest.insert("validation_fraction", 0.05); secondManifest.insert("split_seed", 42); writeFixtureManifest(second, secondManifest);
            shownWindow.populateLibrary({{"datasets", QJsonArray{libraryDataset(shown), libraryDataset(second)}}});
            shownWindow.requestedReviewPath_ = shown; shownWindow.populateReview(photos); shownWindow.tabs_->setCurrentIndex(1);
            selectOnly(shownWindow.reviewSamples_, shownWindow.reviewSamples_->topLevelItem(0));
            shownWindow.show(); application.processEvents(); shownWindow.resize(1280, 900); application.processEvents();
            shownWindow.rgbPreview_->reset("Photo preview"); shownWindow.depthPreview_->reset("Depth preview");
            shownWindow.statusBar()->showMessage("Removed photos stay visible and restorable. Changes save to this dataset automatically.");
            require(shownWindow.grab().save(screenshot), "cannot save the dataset management fixture screenshot");
            shownWindow.tabs_->setCurrentIndex(2); shownWindow.statusBar()->showMessage("Validation size saves to the selected dataset automatically.");
            shownWindow.datasets_->setCurrentItem(shownWindow.datasets_->topLevelItem(1));
            shownWindow.setCollectionRole(shownWindow.collectionSources_->topLevelItem(0), "unused");
            shownWindow.setCollectionRole(shownWindow.collectionSources_->topLevelItem(1), "train");
            application.processEvents();
            const QString splitScreenshot = QDir(QFileInfo(screenshot).absolutePath()).filePath("dataset-inplace-training.png");
            require(shownWindow.grab().save(splitScreenshot), "cannot save the training split fixture screenshot");
        }
        std::cout << "Dataset management: second-row and bulk role buttons, checkbox-free controls, in-place metadata saves, durable drafts across fresh windows, explicit close recovery, save retry, and selected validation percentage before Trainer passed\n";
        return 0;
    } catch (const std::exception &error) { std::cerr << error.what() << '\n'; return 1; }
}
