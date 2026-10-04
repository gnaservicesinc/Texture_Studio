#pragma once

#include <QCoreApplication>
#include <QApplication>
#include <QCryptographicHash>
#include <QDir>
#include <QElapsedTimer>
#include <QFileInfo>
#include <QJsonDocument>
#include <QJsonObject>
#include <QLocalSocket>
#include <QLockFile>
#include <QMessageBox>
#include <QObject>
#include <QPointer>
#include <QTimer>
#include <QWindow>
#include <functional>
#include <memory>

namespace IPDE {
inline QString projectRoot() {
    const auto args = QCoreApplication::arguments();
    const int index = args.indexOf("--project");
    if (index < 0 || index + 1 >= args.size()) return {};
    const QFileInfo info(args[index + 1]);
    return info.canonicalFilePath();
}
inline QString serverName(const QString &project) {
    return "ipde-" + QString::fromLatin1(QCryptographicHash::hash(project.toUtf8(), QCryptographicHash::Sha256).toHex().left(32));
}

// Every GUI subapp registers with its project hub. The lock covers the process
// lifetime; it also protects a project if the hub crashes before its subapps.
class ProjectSession : public QObject {
public:
    ProjectSession(QString role, QObject *parent) : QObject(parent), role_(std::move(role)), socket_(this) {}
    ~ProjectSession() override { shutdown(); }

    // Bind a handler to the window which owns its captured state. A session can
    // outlive that window, including when the event loop returns during exit.
    void setChangedHandler(QObject *context, std::function<void()> callback) {
        changed_ = {context, std::move(callback)};
    }
    void setDisconnectedHandler(QObject *context, std::function<void()> callback) {
        disconnected_ = {context, std::move(callback)};
    }
    void setActivationHandler(QObject *context, std::function<void(const QJsonObject &)> callback) {
        activationContext_ = context; activation_ = std::move(callback);
    }
    void setStateHandler(QObject *context, std::function<void(const QJsonObject &)> callback) {
        stateContext_ = context; state_ = std::move(callback);
    }
    void setBusy(const QString &operation) {
        if (stopping_ || !started_) return;
        socket_.write(QJsonDocument(QJsonObject{{"command", "busy"}, {"operation", operation}}).toJson(QJsonDocument::Compact) + '\n'); socket_.flush();
    }
    bool openApp(const QString &role, const QJsonObject &request = {}) {
        if (stopping_ || !started_ || socket_.state() != QLocalSocket::ConnectedState) return false;
        QJsonObject message = request;
        message.insert("command", "open_app"); message.insert("role", role);
        socket_.write(QJsonDocument(message).toJson(QJsonDocument::Compact) + '\n');
        socket_.flush(); return true;
    }
    void shutdown() {
        if (stopping_) return;
        stopping_ = true;
        changed_ = {}; disconnected_ = {}; activationContext_.clear(); activation_ = {}; stateContext_.clear(); state_ = {};
        // QLocalSocket::~QLocalSocket can synchronously emit disconnected.
        // Disconnect before any member or the captured window is destroyed.
        socket_.disconnect(this);
        socket_.blockSignals(true);
        socket_.abort();
        buffer_.clear();
    }
    QString errorString() const { return error_; }

    bool start(bool showErrors = true) {
        if (stopping_) return false;
        if (started_) return true;
        if (QCoreApplication::arguments().contains("--smoke-test") && projectRoot().isEmpty()) return true;
        const auto fail = [this, showErrors](const QString &message) {
            error_ = message; shutdown();
            if (showErrors) QMessageBox::information(nullptr, "IPDE Studio", message);
            return false;
        };
        const QString project = projectRoot();
        const QString token = qEnvironmentVariable("IPDE_STUDIO_TOKEN");
        if (project.isEmpty() || token.isEmpty()) return fail("Open this app from IPDE Studio and choose a project.");
        QDir(project).mkpath(".studio");
        lock_ = std::make_unique<QLockFile>(QDir(project).filePath(".studio/" + role_ + ".lock"));
        lock_->setStaleLockTime(0);
        if (!lock_->tryLock(0)) return fail("This app is already open for this project. Use its existing window, or choose another project in Studio.");
        socket_.connectToServer(serverName(project));
        if (!socket_.waitForConnected(1500)) return fail("The IPDE Studio project hub is unavailable. Reopen the project in Studio.");
        socket_.write(QJsonDocument(QJsonObject{{"command", "register"}, {"token", token}, {"role", role_},
            {"pid", QCoreApplication::applicationPid()}}).toJson(QJsonDocument::Compact) + '\n');
        QElapsedTimer handshake; handshake.start();
        while (!socket_.canReadLine()) {
            const int remaining = 1500 - int(handshake.elapsed());
            if (remaining <= 0 || !socket_.waitForReadyRead(remaining))
                return fail("Studio did not accept this project session.");
        }
        QJsonParseError parseError;
        const auto responseDocument = QJsonDocument::fromJson(socket_.readLine(), &parseError);
        if (parseError.error != QJsonParseError::NoError || !responseDocument.isObject())
            return fail("Studio returned an invalid project session response.");
        const auto response = responseDocument.object();
        if (!response.value("accepted").toBool()) return fail(response.value("error").toString("Studio rejected this session."));
        const auto readEvents = [this] {
            if (stopping_) return;
            buffer_ += socket_.readAll();
            const QPointer<ProjectSession> guard(this);
            while (buffer_.contains('\n')) {
                const int end = buffer_.indexOf('\n');
                const auto event = QJsonDocument::fromJson(buffer_.left(end)).object();
                buffer_.remove(0, end + 1);
                if (event.value("event") == "project_changed") queue(changed_);
                if (event.value("event") == "activate") {
                    const QPointer<QObject> context = activationContext_;
                    const auto callback = activation_;
                    if (context && callback) QTimer::singleShot(0, context, [guard, context, callback, event] {
                        if (guard && !guard->stopping_ && context) callback(event);
                    });
                    else QTimer::singleShot(0, this, [guard] {
                        if (!guard || guard->stopping_) return;
                        for (auto *window : QApplication::topLevelWidgets()) {
                            if (window->windowType() != Qt::Window) continue;
                            window->showNormal(); window->raise(); window->activateWindow();
                            if (window->windowHandle()) window->windowHandle()->requestActivate();
                            break;
                        }
                    });
                }
                if (event.value("event") == "project_busy") {
                    const QPointer<QObject> context = stateContext_; const auto callback = state_;
                    if (context && callback) QTimer::singleShot(0, context, [guard, context, callback, event] {
                        if (guard && !guard->stopping_ && context) callback(event);
                    });
                }
                if (!guard || stopping_) return;
            }
        };
        connect(&socket_, &QLocalSocket::readyRead, this, readEvents);
        connect(&socket_, &QLocalSocket::disconnected, this, [this] {
            if (!stopping_) queue(disconnected_);
        });
        started_ = true;
        // The hub can append a project event to its registration response.
        // Its readyRead signal may already have fired during the handshake.
        if (socket_.bytesAvailable()) QTimer::singleShot(0, this, readEvents);
        return true;
    }
private:
    struct Handler {
        QPointer<QObject> context;
        std::function<void()> callback;
    };
    void queue(const Handler &handler) {
        // Run UI work after Qt finishes emitting the socket signal. A handler
        // can then close the session without deleting a socket mid-notification.
        const auto callback = handler.callback;
        const QPointer<ProjectSession> guard(this);
        const QPointer<QObject> context = handler.context;
        if (context && callback) QTimer::singleShot(0, context, [guard, context, callback] {
            if (guard && !guard->stopping_ && context) callback();
        });
    }
    QString role_;
    QLocalSocket socket_;
    QByteArray buffer_;
    std::unique_ptr<QLockFile> lock_;
    Handler changed_, disconnected_;
    QPointer<QObject> activationContext_;
    std::function<void(const QJsonObject &)> activation_;
    QPointer<QObject> stateContext_;
    std::function<void(const QJsonObject &)> state_;
    QString error_;
    bool started_ = false, stopping_ = false;
};
} // namespace IPDE
