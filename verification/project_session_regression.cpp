#include "project_session.h"

#include <QApplication>
#include <QElapsedTimer>
#include <QLocalServer>
#include <QMainWindow>
#include <QStatusBar>
#include <QTemporaryDir>
#include <QThread>
#include <atomic>
#include <iostream>
#include <stdexcept>

namespace {
void require(bool condition, const char *message) {
    if (!condition) throw std::runtime_error(message);
}

bool await(const std::function<bool()> &condition, int timeout = 2000) {
    QElapsedTimer timer; timer.start();
    while (!condition() && timer.elapsed() < timeout) {
        QCoreApplication::processEvents(QEventLoop::AllEvents, 10);
        QThread::msleep(1);
    }
    return condition();
}

// A real hub in a separate event loop allows the client's synchronous
// registration handshake to run exactly as it does in the subapps.
class Hub {
public:
    Hub(const QString &name, bool accepted = true, bool malformed = false, bool fragmented = false, bool initialChanged = false) {
        worker_ = new QObject; worker_->moveToThread(&thread_);
        QObject::connect(&thread_, &QThread::finished, worker_, &QObject::deleteLater);
        thread_.start();
        bool listening = false;
        QMetaObject::invokeMethod(worker_, [this, name, accepted, malformed, fragmented, initialChanged, &listening] {
            server_ = new QLocalServer(worker_);
            listening = server_->listen(name);
            QObject::connect(server_, &QLocalServer::newConnection, worker_, [this, accepted, malformed, fragmented, initialChanged] {
                while (server_->hasPendingConnections()) {
                    auto *peer = server_->nextPendingConnection(); peers_.append(peer);
                    const auto buffer = std::make_shared<QByteArray>();
                    QObject::connect(peer, &QLocalSocket::readyRead, peer, [this, peer, buffer, accepted, malformed, fragmented, initialChanged] {
                        *buffer += peer->readAll();
                        if (!buffer->contains('\n')) return;
                        registrations_.fetch_add(1);
                        QByteArray response = malformed ? "not JSON\n" : accepted ? "{\"accepted\":true}\n"
                            : "{\"accepted\":false,\"error\":\"Fixture rejected registration\"}\n";
                        if (initialChanged) response += "{\"event\":\"project_changed\"}\n";
                        if (fragmented) {
                            peer->write(response.left(8));
                            QTimer::singleShot(10, peer, [peer, response] { peer->write(response.mid(8)); peer->flush(); });
                        } else peer->write(response);
                        peer->flush(); buffer->clear();
                    });
                }
            });
        }, Qt::BlockingQueuedConnection);
        require(listening, "fixture hub could not listen");
    }
    ~Hub() {
        QMetaObject::invokeMethod(worker_, [this] { delete server_; server_ = nullptr; }, Qt::BlockingQueuedConnection);
        thread_.quit(); thread_.wait();
    }
    void changed(int count = 1) {
        QMetaObject::invokeMethod(worker_, [this, count] {
            for (const auto &peer : peers_) if (peer && peer->state() == QLocalSocket::ConnectedState) {
                peer->write(QByteArray("{\"event\":\"project_changed\"}\n").repeated(count)); peer->flush();
            }
        }, Qt::BlockingQueuedConnection);
    }
    void disconnectClients() {
        QMetaObject::invokeMethod(worker_, [this] {
            for (const auto &peer : peers_) if (peer) peer->abort();
        }, Qt::BlockingQueuedConnection);
    }
private:
    QObject *worker_ = nullptr;
    QLocalServer *server_ = nullptr;
    QList<QPointer<QLocalSocket>> peers_;
    QThread thread_;
    std::atomic<int> registrations_{0};
};

void liveHandlersAndShutdown(const QString &serverName, const QString &project) {
    Hub hub(serverName);
    int changed = 0, disconnected = 0;
    {
        IPDE::ProjectSession session("trainer", nullptr);
        require(session.start(false), "registered session failed to start");
        QMainWindow window;
        session.setChangedHandler(&window, [&] { ++changed; });
        session.setDisconnectedHandler(&window, [&] { ++disconnected; window.statusBar()->showMessage("disconnected"); });
        hub.changed();
        require(await([&] { return changed == 1; }), "live changed handler did not run");
        hub.disconnectClients();
        require(await([&] { return disconnected == 1; }), "live disconnect handler did not run");
        session.shutdown(); session.shutdown();
        require(disconnected == 1, "shutdown emitted a UI disconnect callback");
        QLockFile competing(QDir(project).filePath(".studio/trainer.lock")); competing.setStaleLockTime(0);
        require(!competing.tryLock(), "shutdown released project lock before session lifetime ended");
    }
    QLockFile released(QDir(project).filePath(".studio/trainer.lock")); released.setStaleLockTime(0);
    require(released.tryLock(), "session destructor left project lock behind");
}

void windowFirstTeardown(const QString &serverName) {
    Hub hub(serverName);
    int changed = 0, disconnected = 0;
    {
        IPDE::ProjectSession session("trainer", nullptr);
        require(session.start(false), "teardown fixture could not register");
        {
            QMainWindow window;
            session.setChangedHandler(&window, [&] { ++changed; window.statusBar()->showMessage("changed"); });
            session.setDisconnectedHandler(&window, [&] { ++disconnected; window.statusBar()->showMessage("disconnected"); });
            hub.changed(); require(await([&] { return changed == 1; }), "window handler fixture did not run");
        }
        // Matches the reported main(): session is declared before the window.
        // Queued hub events and synchronous socket teardown must ignore it.
        hub.changed(); hub.disconnectClients();
        const auto *socket = session.findChild<QLocalSocket *>();
        require(socket && await([&] { return socket->state() == QLocalSocket::UnconnectedState; }), "hub disconnect did not reach client");
    }
    require(changed == 1 && disconnected == 0, "callback ran after captured window was destroyed");
}

void reentrantCallbackDestruction(const QString &serverName) {
    Hub hub(serverName);
    QObject context;
    int callbacks = 0;
    auto session = std::make_unique<IPDE::ProjectSession>("trainer", nullptr);
    require(session->start(false), "reentrant callback fixture could not register");
    session->setChangedHandler(&context, [&] { ++callbacks; session.reset(); });
    hub.changed(2);
    require(await([&] { return !session; }), "callback did not destroy session");
    require(callbacks == 1, "event dispatch used the session after callback destruction");
}

void queuedHandlersCancelled(const QString &serverName) {
    Hub hub(serverName);
    QObject context;
    int callbacks = 0;
    IPDE::ProjectSession session("trainer", nullptr);
    require(session.start(false), "queued handler fixture could not register");
    session.setDisconnectedHandler(&context, [&] { ++callbacks; });
    auto *socket = session.findChild<QLocalSocket *>();
    require(socket && QMetaObject::invokeMethod(socket, "disconnected", Qt::DirectConnection), "fixture could not queue disconnect handler");
    session.shutdown();
    QCoreApplication::processEvents(QEventLoop::AllEvents, 10);
    require(callbacks == 0, "queued UI handler ran after session shutdown");
}

void handshakeFraming(const QString &serverName) {
    {
        Hub hub(serverName, true, false, true);
        IPDE::ProjectSession session("trainer", nullptr);
        require(session.start(false), "fragmented valid registration response was rejected");
    }
    {
        Hub hub(serverName, true, false, false, true);
        IPDE::ProjectSession session("trainer", nullptr);
        require(session.start(false), "coalesced registration fixture could not start");
        QObject context;
        int callbacks = 0;
        session.setChangedHandler(&context, [&] { ++callbacks; });
        require(await([&] { return callbacks == 1; }), "project event bundled with registration response was lost");
    }
}

void connectedDestruction(const QString &serverName) {
    Hub hub(serverName);
    QObject context;
    int disconnected = 0;
    {
        IPDE::ProjectSession session("trainer", nullptr);
        require(session.start(false), "destruction fixture could not register");
        session.setDisconnectedHandler(&context, [&] { ++disconnected; });
    }
    require(disconnected == 0, "connected session destruction called the disconnect UI handler");
}

void failedStartup(const QString &serverName, const QString &project) {
    for (bool malformed : {false, true}) {
        Hub hub(serverName, false, malformed);
        IPDE::ProjectSession session("trainer", nullptr);
        require(!session.start(false), "rejected or malformed registration was accepted");
        require(!session.errorString().isEmpty(), "startup failure was not explained");
        session.shutdown();
    }
    qunsetenv("IPDE_STUDIO_TOKEN");
    {
        IPDE::ProjectSession session("trainer", nullptr);
        require(!session.start(false), "missing hub token was accepted");
    }
    qputenv("IPDE_STUDIO_TOKEN", "verification-token");
    {
        IPDE::ProjectSession session("trainer", nullptr);
        require(!session.start(false), "unavailable hub was accepted");
    }
    // None of these paths constructed a main window; locks still release.
    QLockFile lock(QDir(project).filePath(".studio/trainer.lock")); lock.setStaleLockTime(0);
    require(lock.tryLock(), "failed startup left the project locked");
}
} // namespace

int main(int argc, char **argv) {
    Q_UNUSED(argc);
    QTemporaryDir project;
    require(project.isValid(), "cannot create temporary project");
    const QString projectPath = QFileInfo(project.path()).canonicalFilePath();
    QByteArray projectArgument = projectPath.toUtf8();
    char option[] = "--project";
    char *arguments[] = {argv[0], option, projectArgument.data(), nullptr};
    int argumentCount = 3;
    QApplication application(argumentCount, arguments);
    application.setQuitOnLastWindowClosed(false);
    qputenv("IPDE_STUDIO_TOKEN", "verification-token");
    const auto serverName = IPDE::serverName(projectPath);
    try {
        liveHandlersAndShutdown(serverName, projectPath);
        windowFirstTeardown(serverName);
        connectedDestruction(serverName);
        reentrantCallbackDestruction(serverName);
        queuedHandlersCancelled(serverName);
        handshakeFraming(serverName);
        failedStartup(serverName, projectPath);
        std::cout << "ProjectSession: live callbacks, window-first and connected teardown, repeated shutdown, failed registration, and lock release passed\n";
        return 0;
    } catch (const std::exception &error) {
        std::cerr << error.what() << '\n'; return 1;
    }
}
