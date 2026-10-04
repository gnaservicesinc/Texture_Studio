// Exercise the real project hub with lightweight registered client sockets.
#define IPDE_STUDIO_REGRESSION
#define main ipde_studio_application_main
#include "../src/gui/studio.cpp"
#undef main

#include <QElapsedTimer>
#include <QTemporaryDir>
#include <QThread>
#include <iostream>
#include <stdexcept>

namespace {
void require(bool condition, const char *message) { if (!condition) throw std::runtime_error(message); }
bool await(const std::function<bool()> &condition) {
    QElapsedTimer timer; timer.start();
    while (!condition() && timer.elapsed() < 2000) { QCoreApplication::processEvents(QEventLoop::AllEvents, 10); QThread::msleep(1); }
    return condition();
}
QJsonObject event(QLocalSocket &socket) {
    require(await([&] { return socket.canReadLine(); }), "hub response timed out");
    return QJsonDocument::fromJson(socket.readLine()).object();
}
void send(QLocalSocket &socket, const QJsonObject &message) {
    socket.write(QJsonDocument(message).toJson(QJsonDocument::Compact) + '\n'); socket.flush();
}
void registerClient(QLocalSocket &socket, Hub *hub, const QString &role) {
    socket.connectToServer(IPDE::serverName(hub->project));
    require(await([&] { return socket.state() == QLocalSocket::ConnectedState; }), "fixture client could not connect");
    send(socket, {{"command", "register"}, {"token", hub->token}, {"role", role}});
    require(event(socket).value("accepted").toBool(), "hub rejected fixture registration");
    require(event(socket).value("event") == "project_busy", "new client missed the project busy state");
}
void projectScopedRoutingAndBusy(StudioWindow &window, const QString &first, const QString &second) {
    require(window.addProject(first, "First") && window.addProject(second, "Second"), "fixture projects could not be created");
    Hub *hub = window.hubs_.at(first).get();
    QLocalSocket datasets, trainer;
    registerClient(datasets, hub, "datasets"); registerClient(trainer, hub, "trainer");
    require(event(datasets).value("event") == "project_busy", "existing app missed new client's busy snapshot");
    send(trainer, {{"command", "open_app"}, {"role", "datasets"}, {"section", "review"}, {"dataset", "/fixture/selected"}});
    const auto activation = event(datasets);
    require(activation.value("event") == "activate" && activation.value("dataset") == "/fixture/selected" && activation.value("section") == "review", "existing project app was not activated with selected dataset context");
    require(window.selectedProject() == second && window.hubs_.at(second)->clients.isEmpty(), "request was routed through the currently selected project instead of its owning hub");
    require(hub->processes.isEmpty(), "activating an existing app launched a duplicate process");
    send(trainer, {{"command", "busy"}, {"operation", "train"}});
    require(event(datasets).value("operations").toObject().value("trainer") == "train", "Dataset Studio was not told that training is running");
    require(event(trainer).value("operations").toObject().value("trainer") == "train", "Trainer missed its project operation state");
    trainer.disconnectFromServer();
    require(event(datasets).value("operations").toObject().isEmpty(), "closed Trainer left dataset cleanup permanently disabled");
    QLocalSocket outsider;
    outsider.connectToServer(IPDE::serverName(first)); require(await([&] { return outsider.state() == QLocalSocket::ConnectedState; }), "outsider could not connect");
    send(outsider, {{"command", "open_app"}, {"role", "datasets"}});
    require(!event(outsider).value("accepted").toBool(), "unregistered client could launch or activate project apps");
    require(hub->processes.isEmpty(), "rejected client launched a project app");
    datasets.disconnectFromServer(); require(await([&] { return hub->clients.isEmpty(); }), "fixture clients failed to close");
}
}

int main(int argc, char **argv) {
    Q_UNUSED(argc); char smoke[] = "--smoke-test"; char *arguments[] = {argv[0], smoke, nullptr}; int count = 2;
    QApplication application(count, arguments); application.setQuitOnLastWindowClosed(false);
    QTemporaryDir temporary; QSettings::setDefaultFormat(QSettings::IniFormat); QSettings::setPath(QSettings::IniFormat, QSettings::UserScope, temporary.path());
    QDir().mkpath(temporary.filePath("first")); QDir().mkpath(temporary.filePath("second"));
    const QString first = QFileInfo(temporary.filePath("first")).canonicalFilePath(), second = QFileInfo(temporary.filePath("second")).canonicalFilePath();
    try {
        StudioWindow window; projectScopedRoutingAndBusy(window, first, second);
        std::cout << "Studio: authenticated project-scoped app activation, selection forwarding, duplicate prevention and busy state passed\n";
        return 0;
    } catch (const std::exception &error) { std::cerr << error.what() << '\n'; return 1; }
}
