// Exercise dataset selection and photo edits against the real Qt interface.
#define IPDE_STUDIO_REGRESSION
#define main ipde_studio_application_main
#include "../src/gui/trainer.cpp"
#undef main

#include <iostream>
#include <stdexcept>

namespace {
void require(bool condition, const char *message) {
    if (!condition) throw std::runtime_error(message);
}

QJsonObject libraryDataset(const QString &path) {
    return {{"path", path}, {"name", QFileInfo(path).fileName()}, {"sample_count", 4},
            {"train_count", 3}, {"validation_count", 1}};
}

QJsonObject sample(const QString &id, const QString &photo, const QString &group,
                   const QString &teacher, const QString &split = "train") {
    return {{"id", id}, {"source_id", photo}, {"source_path", "/fixture/" + photo + ".HEIC"},
            {"group_id", group}, {"requested_group", group}, {"teacher_id", teacher},
            {"split", split}, {"labels", QJsonArray{}}, {"photo_metadata", QJsonObject{}}};
}

QJsonObject review(const QString &path) {
    // Two teachers for photo one, a related second photo, and one independent photo.
    return {{"dataset_path", path}, {"samples", QJsonArray{
        sample("shared-1", "one", "capture-a", "Teacher A"),
        sample("shared-2", "one", "capture-a", "Teacher B"),
        sample("related", "two", "capture-a", "Teacher A"),
        sample("held-out", "three", "capture-b", "Teacher A", "validation")}}};
}

void selectOnly(QTreeWidget *tree, QTreeWidgetItem *item) {
    tree->clearSelection(); tree->setCurrentItem(item); item->setSelected(true);
}

QByteArray readBytes(const QString &path) {
    QFile file(path); require(file.open(QIODevice::ReadOnly), "cannot read fixture manifest");
    return file.readAll();
}

void librarySelection(TrainerWindow &window, const QString &workspace) {
    const QString first = QDir(workspace).filePath("datasets/first");
    const QString second = QDir(workspace).filePath("datasets/second");
    const QJsonObject library{{"datasets", QJsonArray{libraryDataset(first), libraryDataset(second)}}};
    window.populateLibrary(library);
    auto *lowerSecond = window.collectionSources_->topLevelItem(1);
    window.collectionSources_->setCurrentItem(lowerSecond);
    require(TrainerWindow::selectedPath(window.datasets_) == second,
            "selecting the second preparation dataset left library actions on the first dataset");
    lowerSecond->setCheckState(0, Qt::Checked);
    window.populateLibrary(library);
    require(TrainerWindow::selectedPath(window.datasets_) == second &&
            TrainerWindow::selectedPath(window.collectionSources_) == second,
            "refresh changed the active dataset between library and preparation");
    require(window.collectionSources_->topLevelItem(1)->checkState(0) == Qt::Checked,
            "refresh lost the training role while preserving active selection");
    window.datasets_->setCurrentItem(window.datasets_->topLevelItem(0));
    require(TrainerWindow::selectedPath(window.collectionSources_) == first,
            "selecting the library dataset did not update preparation's active dataset");
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
    window.populateReview(review(first));
    auto *photo = window.reviewSamples_->topLevelItem(0);
    selectOnly(window.reviewSamples_, photo);
    window.setReviewIncluded(false);
    require(photo->child(0)->checkState(0) == Qt::Unchecked &&
            photo->child(1)->checkState(0) == Qt::Unchecked,
            "removing a selected photo retained one of its teacher entries");
    window.setReviewIncluded(true);
    selectOnly(window.reviewSamples_, photo->child(0));
    window.setReviewIncluded(false);
    require(photo->child(0)->checkState(0) == Qt::Unchecked &&
            photo->child(1)->checkState(0) == Qt::Checked,
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
    require(readBytes(manifest) == original, "editing the photo list changed the source manifest");
    window.reviewedName_->setText("my-first-dataset-version");

    window.populateReview(review(second));
    require(window.reviewSamples_->topLevelItem(0)->child(0)->checkState(0) == Qt::Checked &&
            window.reviewSamples_->topLevelItem(0)->child(0)->text(1) == "train",
            "opening another dataset with shared sample IDs inherited the first dataset's draft");
    require(window.reviewDrafts_.contains(first), "switching datasets discarded the first dataset's draft");
    window.populateReview(review(first));
    require(window.reviewSamples_->topLevelItem(0)->child(0)->checkState(0) == Qt::Unchecked &&
            window.reviewSamples_->topLevelItem(0)->child(0)->text(1) == "validation" &&
            window.reviewSamples_->topLevelItem(1)->child(0)->text(1) == "validation",
            "returning to a dataset did not restore its photo and split edits");
    require(window.reviewedName_->text() == "my-first-dataset-version",
            "switching datasets discarded the chosen name for its edited version");
    require(readBytes(manifest) == original, "draft restoration changed the source manifest");

    // A bulk selection includes complete photos, with no need to tick every teacher.
    window.reviewSamples_->clearSelection();
    window.reviewSamples_->topLevelItem(0)->setSelected(true);
    window.reviewSamples_->topLevelItem(1)->setSelected(true);
    window.setReviewIncluded(false);
    require(window.selectedReviewEdits().value("keep").toArray() == QJsonArray{"held-out"},
            "bulk photo removal did not retain only the independent photo");
    window.setReviewIncluded(true);
    require(window.selectedReviewEdits().value("keep").toArray().size() == 4,
            "bulk photo restoration failed to restore every selected teacher");
}

void filteredNavigation(TrainerWindow &window, const QString &workspace) {
    window.populateReview(review(QDir(workspace).filePath("datasets/navigation")));
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
    window.populateReview(generating);
    window.reviewSamples_->topLevelItem(0)->child(0)->setCheckState(0, Qt::Unchecked);
    generating.insert("samples", QJsonArray{sample("initial", "initial-photo", "first-group", "Teacher"),
                                            sample("new", "new-photo", "new-group", "Teacher")});
    window.populateReview(generating);
    require(window.reviewSamples_->topLevelItem(0)->child(0)->checkState(0) == Qt::Unchecked &&
            window.reviewSamples_->topLevelItem(1)->child(0)->checkState(0) == Qt::Checked,
            "streaming refresh either lost an existing exclusion or excluded a newly generated photo");
    const QString ungrouped = QDir(workspace).filePath("datasets/ungrouped");
    window.populateReview({{"dataset_path", ungrouped}, {"samples", QJsonArray{
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
    window.populateReview(review(first));
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
    window.populateReview(review(second));
    require(window.hasUnsavedReviewExclusions(first),
            "loading the second dataset discarded the unsaved edits for the first dataset");
}

void addedPhotoContinuationAndRetry(TrainerWindow &window, const QString &workspace) {
    const QString base = QDir(workspace).filePath("datasets/continuation-base");
    const QString addition = QDir(workspace).filePath("datasets/generated-addition");
    const QString output = QDir(workspace).filePath("datasets/continuation-version");
    window.requestedReviewPath_ = base; window.populateReview(review(base));
    selectOnly(window.reviewSamples_, window.reviewSamples_->topLevelItem(0)->child(0));
    window.setReviewIncluded(false); window.setReviewSplit("validation");
    const QJsonObject edits = window.selectedReviewEdits();
    require(!edits.value("splits").toObject().contains("shared-1") &&
            edits.value("splits").toObject().value("shared-2").toString() == "validation",
            "save instructions contain a split assignment for a removed teacher entry");
    window.rememberReviewDraft();
    window.appendBase_ = base; window.appendEdits_ = edits; window.appendVersionName_ = "continuation-version";
    {
        QSignalBlocker blocker(window.process_);
        window.job_ = "Generate dataset"; window.refreshAfter_ = true;
        window.stdout_ = QJsonDocument(QJsonObject{{"dataset_path", addition}}).toJson(); window.setBusy(true);
        window.processFinished(0, QProcess::NormalExit);
        const QStringList arguments = window.process_->arguments();
        const int operation = arguments.indexOf("edit-dataset"), addArgument = arguments.indexOf("--add-dataset");
        const int editsArgument = arguments.indexOf("--edits-json");
        require(window.job_ == "Save dataset version" && operation >= 0 && arguments.value(operation + 1) == base &&
                addArgument >= 0 && arguments.value(addArgument + 1) == addition && !arguments.contains("dataset"),
                "adding generated photos restarted base inference instead of saving a merged version");
        require(editsArgument >= 0 && readJson(arguments.value(editsArgument + 1)) == edits,
                "adding generated photos lost the base dataset's kept entries or split edits");
        require(window.appendBase_.isEmpty() && window.reviewAdditions_.value(base).contains(addition),
                "add-photo continuation lost the generated additions needed for saving or retry");
        if (window.process_->state() != QProcess::NotRunning) {
            window.process_->kill(); window.process_->waitForFinished(1500);
        }
        window.stdout_ = "{\"error\":\"isolated save fixture failed\"}";
        window.processFinished(1, QProcess::NormalExit);
        require(window.reviewAdditions_.value(base).contains(addition) && window.reviewDrafts_.contains(base),
                "a failed version save discarded generated additions or photo edits needed to retry");
        window.job_ = "Save dataset version"; window.savingVersionSource_ = base; window.refreshAfter_ = false;
        window.stdout_ = QJsonDocument(QJsonObject{{"dataset_path", output}}).toJson(); window.setBusy(true);
        window.processFinished(0, QProcess::NormalExit);
        require(!window.reviewAdditions_.contains(base) && !window.reviewDrafts_.contains(base) &&
                window.reviewEntries().isEmpty() && window.pendingReview_ == output,
                "a successful version save left the applied source draft, additions, or stale photo tree active");
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
    window.populateReview(review(second)); window.populateReview(review(first));
    require(window.selectedReviewEdits().value("splits").toObject().value("shared-1").toString() == "train",
            "switching datasets discarded an explicit choice that matched the source split");
}

void replacingAllPhotos(TrainerWindow &window, const QString &workspace) {
    const QString base = QDir(workspace).filePath("datasets/replace-all-base");
    const QString addition = QDir(workspace).filePath("datasets/replace-all-addition");
    window.requestedReviewPath_ = base; window.populateReview(review(base));
    window.reviewedName_->setText("replaced-photo-version"); window.reviewSamples_->selectAll();
    window.setReviewIncluded(false);
    require(window.selectedReviewEdits().value("keep").toArray().isEmpty(),
            "remove-all fixture still has included base photos");
    require(window.addPhotos_->isEnabled(), "removing all old photos disabled adding their replacements");
    window.reviewAdditions_.insert(base, {addition}); window.updateReviewCount();
    require(window.saveReviewed_->isEnabled(), "replacement photos could not be saved with every old photo removed");
    {
        QSignalBlocker blocker(window.process_); window.saveReviewedDataset();
        const QStringList arguments = window.process_->arguments();
        const int editsArgument = arguments.indexOf("--edits-json"), additionArgument = arguments.indexOf("--add-dataset");
        require(window.job_ == "Save dataset version" && editsArgument >= 0 &&
                readJson(arguments.value(editsArgument + 1)).value("keep").toArray().isEmpty() &&
                additionArgument >= 0 && arguments.value(additionArgument + 1) == addition,
                "saving replacement photos omitted the addition or restored removed base photos");
        if (window.process_->state() != QProcess::NotRunning) {
            window.process_->kill(); window.process_->waitForFinished(1500);
        }
        window.stdout_ = "{\"error\":\"isolated replacement fixture stopped\"}";
        window.processFinished(1, QProcess::NormalExit);
        require(window.reviewAdditions_.value(base).contains(addition) && window.saveReviewed_->isEnabled(),
                "a failed replacement save lost the new photos or disabled retry with an empty base selection");
        window.saveReviewedDataset();
        const QStringList retryArguments = window.process_->arguments();
        const int retryEdits = retryArguments.indexOf("--edits-json"), retryAddition = retryArguments.indexOf("--add-dataset");
        require(retryEdits >= 0 && readJson(retryArguments.value(retryEdits + 1)).value("keep").toArray().isEmpty() &&
                retryAddition >= 0 && retryArguments.value(retryAddition + 1) == addition,
                "retrying a replacement save lost its new photos or restored removed base photos");
        if (window.process_->state() != QProcess::NotRunning) {
            window.process_->kill(); window.process_->waitForFinished(1500);
        }
        window.stdout_ = "{\"error\":\"isolated replacement retry stopped\"}";
        window.processFinished(1, QProcess::NormalExit);
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
        TrainerWindow window(true);
        const QString workspace = QDir(temporary.path()).filePath("workspace");
        QDir().mkpath(workspace); window.workspace_->setText(workspace);
        librarySelection(window, workspace);
        photoEditsAndDrafts(window, workspace);
        filteredNavigation(window, workspace);
        streamingAndUngroupedPhotos(window, workspace);
        asynchronousDatasetSwitch(window, workspace);
        addedPhotoContinuationAndRetry(window, workspace);
        explicitSameSplitChoice(window, workspace);
        replacingAllPhotos(window, workspace);
        const QString screenshot = qEnvironmentVariable("IPDE_DATASET_SCREENSHOT");
        if (!screenshot.isEmpty()) {
            const QString shown = QDir(workspace).filePath("datasets/first");
            window.reviewDrafts_.remove(shown); window.requestedReviewPath_ = shown;
            window.populateReview(review(shown)); window.reviewedName_->setText("first-reviewed");
            selectOnly(window.reviewSamples_, window.reviewSamples_->topLevelItem(0)); window.setReviewIncluded(false);
            window.show(); application.processEvents(); window.resize(1280, 900); application.processEvents();
            window.rgbPreview_->reset("Photo preview"); window.depthPreview_->reset("Depth preview");
            require(window.grab().save(screenshot), "cannot save the dataset management fixture screenshot");
            window.tabs_->setCurrentIndex(2); window.statusBar()->showMessage("Choose validation size, then create a training set.");
            window.datasets_->setCurrentItem(window.datasets_->topLevelItem(1));
            window.collectionSources_->topLevelItem(0)->setCheckState(0, Qt::Unchecked);
            window.collectionSources_->topLevelItem(1)->setCheckState(0, Qt::Checked);
            application.processEvents();
            const QString splitScreenshot = QDir(QFileInfo(screenshot).absolutePath()).filePath("dataset-management-training-split.png");
            require(window.grab().save(splitScreenshot), "cannot save the training split fixture screenshot");
        }
        std::cout << "Dataset management: dataset selection, bulk photo edits, capture-group splits, isolated drafts, streaming/loading states, add-photo continuation, save retry, navigation, and unchanged source manifests passed\n";
        return 0;
    } catch (const std::exception &error) { std::cerr << error.what() << '\n'; return 1; }
}
