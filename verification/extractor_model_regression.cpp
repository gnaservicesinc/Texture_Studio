// Exercise checkpoint-aware output routing in the real extractor interface.
#define IPDE_EXTRACTOR_REGRESSION
#define main ipde_extractor_application_main
#include "../src/gui/main.cpp"
#undef main

#include <QElapsedTimer>
#include <QTemporaryDir>
#include <QThread>
#include <iostream>
#include <stdexcept>

namespace {
void require(bool condition, const char *message) {
    if (!condition) throw std::runtime_error(message);
}

bool await(const std::function<bool()> &condition) {
    QElapsedTimer timer; timer.start();
    while (!condition() && timer.elapsed() < 5000) {
        QCoreApplication::processEvents(QEventLoop::AllEvents, 10);
        QThread::msleep(1);
    }
    return condition();
}

QTreeWidgetItem *product(QTreeWidgetItem *root, const QString &id) {
    for (int index = 0; index < root->childCount(); ++index) {
        if (root->child(index)->data(0, Qt::UserRole + 1).toString() == id) return root->child(index);
    }
    return nullptr;
}

QTreeWidgetItem *addProduct(QTreeWidgetItem *root, const QString &id) {
    auto *item = new QTreeWidgetItem(root, {id});
    item->setData(0, Qt::UserRole + 1, id);
    item->setFlags(item->flags() | Qt::ItemIsUserCheckable);
    item->setCheckState(0, Qt::Checked);
    return item;
}

void response(MainWindow &window, const QString &source, const QString &path,
              const QString &kind, const QStringList &products, bool inspectOnly = true) {
    QJsonArray catalogue;
    for (const auto &id : products)
        catalogue.append(QJsonObject{{"id", id}, {"name", id}, {"width", 6}, {"height", 4}});
    QFile file(path); require(file.open(QIODevice::WriteOnly), "cannot create process response fixture");
    file.write(QJsonDocument(QJsonObject{{"selected_model", kind.isEmpty() ? QJsonValue(QJsonValue::Null) : QJsonValue(QJsonObject{{"kind", kind}})},
        {"available_products", catalogue}, {"assets", QJsonArray{}}, {"source", QJsonObject{}}}).toJson());
    file.close();
    window.current_ = source; window.inspectOnly_ = inspectOnly; window.running_ = true;
    window.queue_.clear(); window.total_ = 1; window.completed_ = 0; window.updateButtons();
    window.process_->start("/bin/cat", {path});
    require(await([&] { return !window.running_; }), "response left the extractor busy");
}

void checkpointRouting(const QString &directory) {
    MainWindow window;
    const QString source = QDir(directory).filePath("sample.HEIC");
    window.raftModel_->setText(QDir(directory).filePath("renamed-checkpoint.pth"));
    window.raftRoot_->setText(" /fixture/RAFT-Stereo ");
    window.raftMember_->setText(" models/display.pth ");
    window.modelSelectionChanged();
    window.current_ = source;
    const auto inspect = window.processArguments();
    for (const auto &option : {QString("--raft-model"), QString("--raft-root"), QString("--raft-model-member")})
        require(inspect.contains(option), "inspection discarded a selected model argument");
    require(inspect.value(inspect.indexOf("--raft-model-member") + 1) == "models/display.pth",
        "inspection changed or failed to trim the ZIP model member");
    require(inspect.contains("--inspect") && !inspect.contains("--select"), "inspection requested an export");
    require(!window.displayStudentSelected(), "an unresolved checkpoint was guessed to be a display model");
    auto *root = new QTreeWidgetItem(window.files_, {"sample.HEIC"}); root->setData(0, Qt::UserRole, source);
    auto *oldMap = addProduct(root, "raft-displacement"); addProduct(root, "raw:0"); addProduct(root, "raft-support");
    window.files_->setCurrentItem(oldMap); window.sources_ = {source};
    window.advancedToggle_->setChecked(true);
    response(window, source, QDir(directory).filePath("display-response.json"), "display_student",
        {"raw:0", "student-display-depth", "student-display-displacement", "student-display-preview"});
    require(window.displayStudentSelected(), "renamed display checkpoint was ignored despite its reported schema");
    require(window.advancedToggle_->isChecked(), "model inspection closed the selected model controls");
    auto *displayMap = product(root, "student-display-displacement");
    require(displayMap && displayMap->checkState(0) == Qt::Checked && window.files_->currentItem() == displayMap,
        "switching to a display checkpoint lost the checked or current displacement output");
    require(product(root, "raw:0")->checkState(0) == Qt::Checked, "model refresh lost an unrelated raw output selection");
    require(!product(root, "raft-displacement") && !product(root, "raft-support") && displayMap->text(1) == "6 × 4",
        "display-model inventory retained a native disparity output or wrong grid dimensions");
    window.goal_->setCurrentIndex(window.goal_->findData("depth-estimation"));
    require(product(root, "student-display-depth")->checkState(0) == Qt::Checked,
        "depth preset ignored the selected display checkpoint");
    product(root, "raw:0")->setCheckState(0, Qt::Checked);
    window.files_->setCurrentItem(product(root, "raw:0"));
    response(window, source, QDir(directory).filePath("display-raw-export-response.json"), {},
        {"raw:0", "student-display-depth", "student-display-displacement", "student-display-preview",
         "raft-depth", "raft-displacement", "raft-preview", "raft-flow", "raft-support"}, false);
    require(window.displayStudentSelected() && product(root, "student-display-depth")->checkState(0) == Qt::Checked,
        "raw-only export discarded the inspected display checkpoint or its depth selection");
    require(!product(root, "raft-depth") && !product(root, "raft-displacement") && !product(root, "raft-preview")
        && !product(root, "raft-flow") && !product(root, "raft-support"),
        "raw-only export exposed outputs incompatible with the selected display checkpoint");
    require(window.files_->currentItem() == product(root, "raw:0") && product(root, "raw:0")->checkState(0) == Qt::Checked,
        "raw-only export lost its current or checked raw output");
    window.files_->setCurrentItem(product(root, "student-display-depth"));
    window.sources_.clear();
    window.raftModel_->setText(QDir(directory).filePath("display-model.pth"));
    window.modelSelectionChanged();
    require(!window.displayStudentSelected(), "checkpoint basename overrode backend model identification");
    window.sources_ = {source};
    response(window, source, QDir(directory).filePath("raft-response.json"), "raft_stereo",
        {"raw:0", "raft-depth", "raft-displacement", "raft-preview", "raft-support"});
    auto *nativeDepth = product(root, "raft-depth");
    require(!window.displayStudentSelected() && nativeDepth && nativeDepth->checkState(0) == Qt::Checked
        && window.files_->currentItem() == nativeDepth, "changing back to RAFT lost the selected depth product");
    response(window, source, QDir(directory).filePath("raft-raw-export-response.json"), {},
        {"raw:0", "raft-depth", "raft-displacement", "raft-preview", "raft-support", "student-display-depth"}, false);
    require(window.selectedModelKind_ == "raft_stereo" && !product(root, "student-display-depth")
        && product(root, "raft-depth")->checkState(0) == Qt::Checked,
        "raw-only export discarded the inspected native checkpoint or exposed display-student products");
    QSettings().setValue("raft/model", QDir(directory).filePath("shared-checkpoint.pth"));
    window.reloadProjectSettings();
    require(window.running_ && window.inspectOnly_ && window.process_->arguments().contains("--inspect")
        && window.process_->arguments().contains(QDir(directory).filePath("shared-checkpoint.pth")),
        "a shared checkpoint change retained the previous model's product inventory");
    window.cancelled_ = true; window.queue_.clear(); window.process_->kill();
    require(await([&] { return !window.running_; }), "cancelled inspection left the extractor busy");
}
}

int main(int argc, char **argv) {
    QApplication application(argc, argv); application.setQuitOnLastWindowClosed(false);
    application.setApplicationName("IPDEExtractorRegression"); application.setOrganizationName("OpenAI");
    QTemporaryDir temporary;
    QSettings::setDefaultFormat(QSettings::IniFormat);
    QSettings::setPath(QSettings::IniFormat, QSettings::UserScope, temporary.path());
    try {
        checkpointRouting(temporary.path());
        std::cout << "Extractor: checkpoint schema, inspection arguments, product selection and shared-model refresh passed\n";
        return 0;
    } catch (const std::exception &error) { std::cerr << error.what() << '\n'; return 1; }
}
