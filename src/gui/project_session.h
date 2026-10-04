#pragma once

#include <QCoreApplication>
#include <QCryptographicHash>
#include <QDir>
#include <QFileInfo>
#include <QJsonDocument>
#include <QJsonObject>
#include <QLocalSocket>
#include <QLockFile>
#include <QMessageBox>
#include <QObject>
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
    bool start() {
        if (QCoreApplication::arguments().contains("--smoke-test")) return true;
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
        if (!socket_.waitForReadyRead(1500)) return fail("Studio did not accept this project session.");
        const auto response = QJsonDocument::fromJson(socket_.readLine()).object();
        if (!response.value("accepted").toBool()) return fail(response.value("error").toString("Studio rejected this session."));
        connect(&socket_, &QLocalSocket::readyRead, this, [this] {
            buffer_ += socket_.readAll();
            while (buffer_.contains('\n')) {
                const int end = buffer_.indexOf('\n');
                const auto event = QJsonDocument::fromJson(buffer_.left(end)).object();
                buffer_.remove(0, end + 1);
                if (event.value("event") == "project_changed" && onChanged) onChanged();
            }
        });
        connect(&socket_, &QLocalSocket::disconnected, this, [this] {
            if (onDisconnected) onDisconnected();
        });
        return true;
    }
    std::function<void()> onChanged;
    std::function<void()> onDisconnected;
private:
    bool fail(const QString &message) {
        QMessageBox::information(nullptr, "IPDE Studio", message);
        return false;
    }
    QString role_;
    QLocalSocket socket_;
    QByteArray buffer_;
    std::unique_ptr<QLockFile> lock_;
};
} // namespace IPDE
