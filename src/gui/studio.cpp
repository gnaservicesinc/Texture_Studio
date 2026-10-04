#include "project_session.h"
#include <QApplication>
#include <QCloseEvent>
#include <QComboBox>
#include <QFileDialog>
#include <QFileSystemWatcher>
#include <QFont>
#include <QInputDialog>
#include <QLabel>
#include <QListWidget>
#include <QLocalServer>
#include <QMainWindow>
#include <QPlainTextEdit>
#include <QProcess>
#include <QProcessEnvironment>
#include <QPushButton>
#include <QSettings>
#include <QStandardPaths>
#include <QTimer>
#include <QUuid>
#include <QVBoxLayout>
#include <map>

namespace {
struct Hub {
    QString project, token;
    std::unique_ptr<QLockFile> lock;
    QLocalServer *server = nullptr;
    QFileSystemWatcher *watcher = nullptr;
    QTimer *debounce = nullptr;
    QMap<QString, QLocalSocket *> clients;
    QMap<QString, QProcess *> processes;
};

class StudioWindow final : public QMainWindow {
public:
    StudioWindow() {
        setWindowTitle("IPDE Studio — Projects"); resize(820, 640);
        auto *central = new QWidget(this); auto *layout = new QVBoxLayout(central);
        layout->setContentsMargins(24, 24, 24, 24);
        auto *title = new QLabel("IPDE Studio", central);
        QFont font = title->font(); font.setPointSize(font.pointSize() + 10); font.setBold(true); title->setFont(font);
        layout->addWidget(title);
        auto *intro = new QLabel("Choose a project and your purpose. Studio shares its settings, datasets and models across the apps.", central);
        intro->setWordWrap(true); layout->addWidget(intro);
        projects_ = new QListWidget(central); layout->addWidget(projects_, 1);
        auto *projectButtons = new QHBoxLayout;
        auto *create = new QPushButton("New project…", central); auto *open = new QPushButton("Open project…", central);
        projectButtons->addWidget(create); projectButtons->addWidget(open); projectButtons->addStretch(); layout->addLayout(projectButtons);
        auto *goalRow = new QHBoxLayout; goalRow->addWidget(new QLabel("Main purpose", central));
        goal_ = new QComboBox(central);
        goal_->addItem("Effect / displacement map — detail first", "effect/map");
        goal_->addItem("Depth estimation — calibrated geometry", "depth-estimation");
        goal_->addItem("Photo effects / masking — portrait depth and mattes", "photo-effects");
        goal_->addItem("Manual — expose all choices", "manual");
        goalRow->addWidget(goal_, 1); layout->addLayout(goalRow);
        description_ = new QLabel(central); description_->setWordWrap(true); layout->addWidget(description_);
        auto *apps = new QHBoxLayout;
        const QList<QPair<QString, QString>> roles{{"Extract maps", "extractor"}, {"Manage & review datasets", "datasets"}, {"Assemble & train", "trainer"}};
        for (const auto &entry : roles) {
            auto *button = new QPushButton(entry.first, central); apps->addWidget(button);
            connect(button, &QPushButton::clicked, this, [this, role=entry.second] { launch(role); });
        }
        layout->addLayout(apps);
        status_ = new QLabel(central); status_->setWordWrap(true); layout->addWidget(status_);
        log_ = new QPlainTextEdit(central); log_->setReadOnly(true); log_->setMaximumHeight(110); log_->setMaximumBlockCount(300); layout->addWidget(log_);
        setCentralWidget(central);
        connect(create, &QPushButton::clicked, this, [this] {
            const QString parent = QFileDialog::getExistingDirectory(this, "Choose a folder for your new project");
            if (parent.isEmpty()) return;
            bool accepted = false;
            const QString name = QInputDialog::getText(this, "New project", "Project name", QLineEdit::Normal, {}, &accepted).trimmed();
            if (!accepted || name.isEmpty()) return;
            if (name.contains('/') || name.contains('\\') || name == "." || name == "..") { status_->setText("Use a project name without path separators."); return; }
            const QString path = QDir(parent).filePath(name);
            if (QFileInfo::exists(path)) { status_->setText("That folder already exists. Use Open project to select it."); return; }
            if (!QDir().mkpath(path)) { status_->setText("Could not create the project folder."); return; }
            addProject(path);
        });
        connect(open, &QPushButton::clicked, this, [this] {
            const QString path = QFileDialog::getExistingDirectory(this, "Open an IPDE project folder");
            if (!path.isEmpty()) addProject(path);
        });
        connect(projects_, &QListWidget::currentItemChanged, this, [this] { loadGoal(); });
        connect(goal_, &QComboBox::currentIndexChanged, this, [this] {
            const QString project = selectedProject();
            if (project.isEmpty() || loading_) return;
            QSettings settings(QDir(project).filePath("project.ini"), QSettings::IniFormat);
            settings.setValue("goal", goal_->currentData()); settings.sync(); updateDescription();
        });
        QSettings recent("IPDE", "Studio");
        for (const auto &path : recent.value("projects").toStringList()) if (QFileInfo(path).isDir()) appendProject(path);
        const QString argumentProject = IPDE::projectRoot();
        if (!argumentProject.isEmpty()) addProject(argumentProject);
        else if (projects_->count()) projects_->setCurrentRow(0);
        updateDescription();
    }
protected:
    void closeEvent(QCloseEvent *event) override {
        bool active = false;
        for (const auto &entry : hubs_) for (auto *process : entry.second->processes) active |= process->state() != QProcess::NotRunning;
        if (active) { event->ignore(); hide(); }
        else event->accept();
    }
private:
    QString selectedProject() const { return projects_->currentItem() ? projects_->currentItem()->data(Qt::UserRole).toString() : QString(); }
    void appendProject(const QString &project) {
        auto *item = new QListWidgetItem(QFileInfo(project).fileName() + "\n" + project, projects_); item->setData(Qt::UserRole, project);
    }
    void addProject(const QString &path) {
        const QString project = QFileInfo(path).canonicalFilePath();
        if (project.isEmpty()) return;
        QDir(project).mkpath("workspace/datasets"); QDir(project).mkpath("workspace/runs");
        QSettings settings(QDir(project).filePath("project.ini"), QSettings::IniFormat);
        if (!settings.contains("goal")) settings.setValue("goal", "effect/map");
        settings.setValue("schema", "ipde-project-v1"); settings.sync();
        int row = -1;
        for (int i = 0; i < projects_->count(); ++i) if (projects_->item(i)->data(Qt::UserRole).toString() == project) row = i;
        if (row < 0) { appendProject(project); row = projects_->count() - 1; }
        projects_->setCurrentRow(row);
        QStringList paths;
        for (int i = 0; i < projects_->count(); ++i) paths << projects_->item(i)->data(Qt::UserRole).toString();
        QSettings("IPDE", "Studio").setValue("projects", paths);
    }
    void loadGoal() {
        loading_ = true;
        QSettings settings(QDir(selectedProject()).filePath("project.ini"), QSettings::IniFormat);
        const int index = goal_->findData(settings.value("goal", "effect/map").toString()); goal_->setCurrentIndex(qMax(0, index));
        loading_ = false; updateDescription();
    }
    void updateDescription() {
        const QString goal = goal_->currentData().toString();
        if (goal == "effect/map") description_->setText("Start with RAFT stereo for detailed displacement. Compare the baseline against a project model before adopting it. Raw arrays stay unchanged.");
        else if (goal == "depth-estimation") description_->setText("Prefer calibrated spatial geometry and meter depth. Teacher models supply estimates; compare held-out captures and independently measured references when available.");
        else if (goal == "photo-effects") description_->setText("Use portrait photos with their embedded Apple depth and matte planes for masking and bokeh. Spatial training needs a calibrated stereo pair.");
        else description_->setText("Manual mode exposes inference, teacher, device, grouping and training settings.");
    }
    Hub *hubFor(const QString &project) {
        const auto existing = hubs_.find(project);
        if (existing != hubs_.end()) return existing->second.get();
        auto hub = std::make_unique<Hub>(); hub->project = project; hub->token = QUuid::createUuid().toString(QUuid::WithoutBraces);
        QDir(project).mkpath(".studio");
        hub->lock = std::make_unique<QLockFile>(QDir(project).filePath(".studio/hub.lock")); hub->lock->setStaleLockTime(0);
        if (!hub->lock->tryLock(0)) { status_->setText("This project is managed by another IPDE Studio window. Open its existing Studio window."); return nullptr; }
        hub->server = new QLocalServer(this);
        const QString name = IPDE::serverName(project); QLocalServer::removeServer(name);
        hub->server->setSocketOptions(QLocalServer::UserAccessOption);
        if (!hub->server->listen(name)) { status_->setText("Could not create project IPC: " + hub->server->errorString()); return nullptr; }
        Hub *pointer = hub.get();
        connect(hub->server, &QLocalServer::newConnection, this, [this, pointer] {
            while (pointer->server->hasPendingConnections()) {
                auto *socket = pointer->server->nextPendingConnection(); socket->setParent(pointer->server);
                auto buffer = std::make_shared<QByteArray>();
                connect(socket, &QLocalSocket::readyRead, this, [this, pointer, socket, buffer] {
                    *buffer += socket->readAll();
                    if (buffer->size() > 65536) { socket->disconnectFromServer(); return; }
                    while (buffer->contains('\n')) {
                        const int end = buffer->indexOf('\n'); const auto message = QJsonDocument::fromJson(buffer->left(end)).object(); buffer->remove(0, end + 1);
                        const QString role = message.value("role").toString();
                        const bool accepted = message.value("command") == "register" && message.value("token") == pointer->token
                            && QStringList{"extractor", "datasets", "trainer"}.contains(role) && !pointer->clients.contains(role);
                        socket->write(QJsonDocument(QJsonObject{{"accepted", accepted}, {"error", accepted ? "" : "An app is already registered, or the project session is invalid."}}).toJson(QJsonDocument::Compact) + '\n');
                        socket->flush();
                        if (!accepted) { socket->disconnectFromServer(); return; }
                        pointer->clients[role] = socket; socket->setProperty("role", role); updateStatus();
                    }
                });
                connect(socket, &QLocalSocket::disconnected, this, [this, pointer, socket] {
                    const QString role = socket->property("role").toString();
                    if (pointer->clients.value(role) == socket) pointer->clients.remove(role);
                    socket->deleteLater(); updateStatus();
                });
            }
        });
        hub->watcher = new QFileSystemWatcher(this); hub->debounce = new QTimer(this); hub->debounce->setSingleShot(true); hub->debounce->setInterval(300);
        connect(hub->watcher, &QFileSystemWatcher::directoryChanged, hub->debounce, qOverload<>(&QTimer::start));
        connect(hub->watcher, &QFileSystemWatcher::fileChanged, hub->debounce, qOverload<>(&QTimer::start));
        connect(hub->debounce, &QTimer::timeout, this, [this, pointer] {
            watchProject(pointer);
            for (auto *client : pointer->clients) client->write("{\"event\":\"project_changed\"}\n");
            if (selectedProject() == pointer->project) loadGoal();
        });
        watchProject(pointer); hubs_[project] = std::move(hub); return pointer;
    }
    void watchProject(Hub *hub) {
        QStringList paths{hub->project, QDir(hub->project).filePath("project.ini"), QDir(hub->project).filePath("workspace"),
            QDir(hub->project).filePath("workspace/datasets"), QDir(hub->project).filePath("workspace/runs")};
        for (const QString &container : {"datasets", "runs"}) {
            QDir dir(QDir(hub->project).filePath("workspace/" + container));
            for (const auto &child : dir.entryInfoList(QDir::Dirs | QDir::NoDotAndDotDot | QDir::NoSymLinks)) paths << child.absoluteFilePath();
        }
        for (const auto &path : paths) if (QFileInfo::exists(path) && !hub->watcher->directories().contains(path) && !hub->watcher->files().contains(path)) hub->watcher->addPath(path);
    }
    QString executable(const QString &role) const {
        const QString bundle = role == "extractor" ? "IPDE.app/Contents/MacOS/IPDE" : "RAFT Studio.app/Contents/MacOS/RAFT Studio";
        const QDir here(QCoreApplication::applicationDirPath());
        const QString packaged = QDir::cleanPath(here.filePath("../Applications/" + bundle));
        if (QFileInfo::exists(packaged)) return packaged;
        return QDir::cleanPath(here.filePath("../../../" + bundle));
    }
    void launch(const QString &role) {
        const QString project = selectedProject();
        if (project.isEmpty()) { status_->setText("Create or open a project first."); return; }
        Hub *hub = hubFor(project); if (!hub) return;
        if (hub->clients.contains(role) || (hub->processes.contains(role) && hub->processes[role]->state() != QProcess::NotRunning)) {
            status_->setText("This app is already running for this project. Select its existing window."); return;
        }
        const QString path = executable(role);
        if (!QFileInfo::exists(path)) { status_->setText("The app is missing from Studio: " + path); return; }
        auto *process = new QProcess(this); process->setProgram(path);
        process->setArguments({"--project", project, "--mode", role});
        auto environment = QProcessEnvironment::systemEnvironment(); environment.insert("IPDE_STUDIO_TOKEN", hub->token); process->setProcessEnvironment(environment);
        hub->processes[role] = process;
        connect(process, &QProcess::readyReadStandardError, this, [this, process] { log_->appendPlainText(QString::fromUtf8(process->readAllStandardError())); });
        connect(process, &QProcess::errorOccurred, this, [this, process] { status_->setText("App launch failed: " + process->errorString()); });
        connect(process, qOverload<int, QProcess::ExitStatus>(&QProcess::finished), this, [this, hub, role, process](int code, QProcess::ExitStatus) {
            hub->processes.remove(role); process->deleteLater(); log_->appendPlainText(role + " closed (" + QString::number(code) + ")."); updateStatus();
            if (!isVisible()) { bool active = false; for (const auto &entry : hubs_) active |= !entry.second->processes.isEmpty(); if (!active) QCoreApplication::quit(); }
        });
        process->start(); status_->setText("Opening " + role + " for " + QFileInfo(project).fileName() + "…");
    }
    void updateStatus() {
        QStringList active;
        for (const auto &entry : hubs_) for (const auto &role : entry.second->clients.keys()) active << QFileInfo(entry.first).fileName() + ": " + role;
        status_->setText(active.isEmpty() ? "Ready" : "Open apps — " + active.join(" · "));
    }
    QListWidget *projects_; QComboBox *goal_; QLabel *description_, *status_; QPlainTextEdit *log_;
    bool loading_ = false;
    std::map<QString, std::unique_ptr<Hub>> hubs_;
};
}

int main(int argc, char **argv) {
    QApplication application(argc, argv); application.setApplicationName("IPDE Studio"); application.setOrganizationName("IPDE");
    StudioWindow window; window.show();
    if (application.arguments().contains("--smoke-test")) QTimer::singleShot(300, &application, &QCoreApplication::quit);
    return application.exec();
}
