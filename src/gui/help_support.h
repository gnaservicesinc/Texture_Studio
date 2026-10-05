#pragma once

#include <QAction>
#include <QApplication>
#include <QClipboard>
#include <QCoreApplication>
#include <QDesktopServices>
#include <QDialog>
#include <QDialogButtonBox>
#include <QDir>
#include <QFile>
#include <QFileInfo>
#include <QHostAddress>
#include <QLabel>
#include <QKeySequence>
#include <QMainWindow>
#include <QMenu>
#include <QMenuBar>
#include <QMessageBox>
#include <QPlainTextEdit>
#include <QPushButton>
#include <QSysInfo>
#include <QTcpServer>
#include <QTcpSocket>
#include <QTimer>
#include <QUrl>
#include <QUuid>
#include <QVBoxLayout>
#include <memory>

#if __has_include("ipde_build_info.h")
#include "ipde_build_info.h"
#endif
#ifndef IPDE_VERSION
#define IPDE_VERSION "0.9.0"
#endif
#ifndef IPDE_BUILD_DATE
#define IPDE_BUILD_DATE "Unavailable in this build"
#endif
#ifndef IPDE_BUILD_REVISION
#define IPDE_BUILD_REVISION "Unavailable in this build"
#endif
#ifndef IPDE_BUILD_CONFIG
#define IPDE_BUILD_CONFIG "Unavailable in this build"
#endif
#ifndef IPDE_BUILD_FLAGS
#define IPDE_BUILD_FLAGS "Unavailable in this build"
#endif
#ifndef IPDE_BUILD_COMPILER
#define IPDE_BUILD_COMPILER "Unavailable in this build"
#endif
#ifndef IPDE_BUILD_QT
#define IPDE_BUILD_QT QT_VERSION_STR
#endif
#ifndef IPDE_REPOSITORY_URL
#define IPDE_REPOSITORY_URL "https://github.com/gnaservicesinc/ipde"
#endif
#ifndef IPDE_PRERELEASE
#define IPDE_PRERELEASE 1
#endif

namespace IPDE {

inline QString documentationRoot() {
    // Prefer deployed resources, including the parent Studio for nested apps.
    QDir directory(QCoreApplication::applicationDirPath());
    for (int depth = 0; depth < 8; ++depth) {
        for (const auto &relative : {QStringLiteral("../Resources/docs"), QStringLiteral("Contents/Resources/docs")}) {
            const auto candidate = QFileInfo(directory.filePath(relative)).canonicalFilePath();
            if (!candidate.isEmpty() && QFileInfo(QDir(candidate).filePath("help/index.html")).isFile()) return candidate;
        }
        if (!directory.cdUp()) break;
    }
#ifdef IPDE_DOCS_DIR
    const auto source = QFileInfo(QString::fromUtf8(IPDE_DOCS_DIR)).canonicalFilePath();
    if (!source.isEmpty() && QFileInfo(QDir(source).filePath("help/index.html")).isFile()) return source;
#endif
    return {};
}

// Qt Network keeps the offline guide available for the lifetime of the app.
// This server accepts only read-only requests for contained documentation files.
class HelpServer final : public QObject {
public:
    explicit HelpServer(const QString &root, QObject *parent = nullptr)
        : QObject(parent), root_(QFileInfo(root).canonicalFilePath()),
          token_(QUuid::createUuid().toString(QUuid::WithoutBraces).toUtf8()), server_(this) {
        connect(&server_, &QTcpServer::newConnection, this, [this] {
            while (server_.hasPendingConnections()) {
                auto *socket = server_.nextPendingConnection();
                const auto request = std::make_shared<QByteArray>();
                connect(socket, &QTcpSocket::disconnected, socket, &QObject::deleteLater);
                QTimer::singleShot(5000, socket, [socket] { socket->disconnectFromHost(); });
                connect(socket, &QTcpSocket::readyRead, socket, [this, socket, request] {
                    if (socket->property("helpResponded").toBool()) return;
                    *request += socket->readAll();
                    if (request->size() > 16384) { respond(socket, 413, "Content Too Large", "text/plain", "Request too large."); return; }
                    if (!request->contains("\r\n\r\n")) return;
                    serve(socket, *request);
                });
            }
        });
    }

    bool start() {
        if (server_.isListening()) return true;
        return !root_.isEmpty() && QFileInfo(root_).isDir() && server_.listen(QHostAddress::LocalHost, 0);
    }
    QString errorString() const { return root_.isEmpty() ? QStringLiteral("Bundled documentation was not found.") : server_.errorString(); }
    QUrl url(const QString &relative = QStringLiteral("help/index.html")) const {
        QUrl result; result.setScheme("http"); result.setHost("127.0.0.1"); result.setPort(server_.serverPort());
        result.setPath("/" + QString::fromLatin1(token_) + "/" + relative); return result;
    }
    quint16 port() const { return server_.serverPort(); }

private:
    static void respond(QTcpSocket *socket, int code, const QByteArray &reason, const QByteArray &type,
                        const QByteArray &body, bool head = false) {
        socket->setProperty("helpResponded", true);
        QByteArray response = "HTTP/1.1 " + QByteArray::number(code) + " " + reason + "\r\nContent-Type: " + type
            + "\r\nContent-Length: " + QByteArray::number(body.size())
            + "\r\nConnection: close\r\nCache-Control: no-store\r\nX-Content-Type-Options: nosniff"
              "\r\nReferrer-Policy: no-referrer\r\nContent-Security-Policy: default-src 'self'; "
              "script-src 'self'; style-src 'self'; img-src 'self'; object-src 'none'; "
              "base-uri 'none'; frame-ancestors 'none'; form-action 'none'\r\n\r\n";
        if (!head) response += body;
        socket->write(response); socket->disconnectFromHost();
    }
    void serve(QTcpSocket *socket, const QByteArray &request) {
        const auto lines = request.left(request.indexOf("\r\n\r\n")).split('\n');
        const auto first = lines.value(0).trimmed().split(' ');
        const bool head = first.value(0) == "HEAD";
        if (first.size() != 3 || (first.value(2) != "HTTP/1.1" && first.value(2) != "HTTP/1.0")) {
            respond(socket, 400, "Bad Request", "text/plain", "Malformed request."); return;
        }
        if (first.value(0) != "GET" && !head) {
            respond(socket, 405, "Method Not Allowed", "text/plain", "Only GET and HEAD are supported."); return;
        }
        QByteArray host;
        int hostCount = 0;
        for (const auto &line : lines) if (line.toLower().startsWith("host:")) { host = line.mid(5).trimmed(); ++hostCount; }
        if (hostCount != 1 || host != "127.0.0.1:" + QByteArray::number(server_.serverPort())) {
            respond(socket, 403, "Forbidden", "text/plain", "Local documentation host required.", head); return;
        }
        const auto rawPath = first.value(1).split('?').value(0);
        const auto prefix = "/" + token_ + "/";
        if (!rawPath.startsWith(prefix)) {
            respond(socket, 404, "Not Found", "text/plain", "Documentation not found.", head); return;
        }
        const auto relative = QUrl::fromPercentEncoding(rawPath.mid(prefix.size()));
        const auto segments = relative.split('/');
        if (relative.contains(QChar(0)) || relative.contains('\\') || relative.startsWith('/')
            || segments.contains("..") || segments.contains(".")) {
            respond(socket, 403, "Forbidden", "text/plain", "Invalid documentation path.", head); return;
        }
        const auto path = QFileInfo(QDir(root_).filePath(relative)).canonicalFilePath();
        if (!path.startsWith(root_ + "/") || !QFileInfo(path).isFile()) {
            respond(socket, 404, "Not Found", "text/plain", "Documentation not found.", head); return;
        }
        const auto extension = QFileInfo(path).suffix().toLower();
        QByteArray type;
        if (extension == "html") type = "text/html; charset=utf-8";
        else if (extension == "css") type = "text/css; charset=utf-8";
        else if (extension == "js") type = "text/javascript; charset=utf-8";
        else if (extension == "svg") type = "image/svg+xml";
        else if (extension == "md") type = "text/plain; charset=utf-8";
        else if (extension == "png") type = "image/png";
        else if (extension == "jpg" || extension == "jpeg") type = "image/jpeg";
        else { respond(socket, 404, "Not Found", "text/plain", "Unsupported documentation file.", head); return; }
        QFile file(path);
        if (!file.open(QIODevice::ReadOnly)) { respond(socket, 404, "Not Found", "text/plain", "Documentation unavailable.", head); return; }
        respond(socket, 200, "OK", type, file.readAll(), head);
    }
    QString root_;
    QByteArray token_;
    QTcpServer server_;
};

inline QString buildInformation(const QString &appName) {
    return QString("%1\nVersion: %2%3\nBuild date (UTC): %4\nRevision: %5\nConfiguration: %6\nCompiler: %7\nQt build: %8\nQt runtime: %9\nBuild flags: %10\nArchitecture: %11\nRepository: %12")
        .arg(appName, QString::fromUtf8(IPDE_VERSION), IPDE_PRERELEASE ? " (pre-release)" : "",
             QString::fromUtf8(IPDE_BUILD_DATE), QString::fromUtf8(IPDE_BUILD_REVISION),
             QString::fromUtf8(IPDE_BUILD_CONFIG), QString::fromUtf8(IPDE_BUILD_COMPILER),
             QString::fromUtf8(IPDE_BUILD_QT), QString::fromUtf8(qVersion()),
             QString::fromUtf8(IPDE_BUILD_FLAGS), QSysInfo::buildCpuArchitecture(), QString::fromUtf8(IPDE_REPOSITORY_URL));
}

inline void showAbout(QWidget *parent, const QString &appName) {
    QDialog dialog(parent); dialog.setObjectName("ipdeAboutDialog"); dialog.setWindowTitle("About " + appName); dialog.resize(630, 440);
    auto *layout = new QVBoxLayout(&dialog);
    auto *heading = new QLabel("<h2>" + appName.toHtmlEscaped() + " " + QString::fromUtf8(IPDE_VERSION).toHtmlEscaped() + "</h2>", &dialog);
    layout->addWidget(heading);
    auto *notice = new QLabel(IPDE_PRERELEASE ? "Development pre-release. Project, dataset and save formats may change before version 1.0.0. Keep original sources and backups." : "Image Precision Data Extractor", &dialog);
    notice->setWordWrap(true); layout->addWidget(notice);
    auto *information = new QPlainTextEdit(buildInformation(appName), &dialog); information->setReadOnly(true); information->setObjectName("ipdeBuildInformation"); layout->addWidget(information, 1);
    auto *links = new QLabel("<a href=\"" + QString::fromUtf8(IPDE_REPOSITORY_URL).toHtmlEscaped() + "\">GitHub repository</a>", &dialog);
    links->setOpenExternalLinks(true); layout->addWidget(links);
    auto *buttons = new QDialogButtonBox(QDialogButtonBox::Close, &dialog);
    auto *copy = buttons->addButton("Copy build information", QDialogButtonBox::ActionRole);
    auto *bug = buttons->addButton("Report Bug", QDialogButtonBox::ActionRole);
    QObject::connect(copy, &QPushButton::clicked, &dialog, [appName] { QApplication::clipboard()->setText(buildInformation(appName)); });
    QObject::connect(bug, &QPushButton::clicked, &dialog, [] { QDesktopServices::openUrl(QUrl("https://github.com/gnaservicesinc/ipde/issues/new")); });
    QObject::connect(buttons, &QDialogButtonBox::rejected, &dialog, &QDialog::reject); layout->addWidget(buttons); dialog.exec();
}

inline void installHelpMenu(QMainWindow *window, const QString &appName, const QString &role) {
    auto *menu = window->menuBar()->addMenu("&Help"); menu->setObjectName("ipdeHelpMenu");
    auto *server = new HelpServer(documentationRoot(), window);
    auto open = [window, server](const QString &relative, const QString &fragment = QString()) {
        if (!server->start()) { QMessageBox::warning(window, "Help unavailable", server->errorString()); return; }
        auto url = server->url(relative); url.setFragment(fragment);
        if (!QDesktopServices::openUrl(url)) QMessageBox::warning(window, "Help unavailable", "The default web browser could not be opened.");
    };
    auto *guide = menu->addAction("Interactive Help"); guide->setObjectName("interactiveHelpAction"); guide->setShortcut(QKeySequence::HelpContents);
    QObject::connect(guide, &QAction::triggered, window, [open, role] { open("help/index.html", role); });
    auto *manual = menu->addMenu("User Manual"); manual->setObjectName("ipdeManualMenu");
    auto *html = manual->addAction("HTML — read or print");
    QObject::connect(html, &QAction::triggered, window, [open] { open("manual/index.html"); });
    auto *markdown = manual->addAction("Markdown — editable source");
    QObject::connect(markdown, &QAction::triggered, window, [open] { open("manual/manual.md"); });
    menu->addSeparator();
    auto *repository = menu->addAction("GitHub Repository"); repository->setObjectName("ipdeRepositoryAction");
    QObject::connect(repository, &QAction::triggered, window, [] { QDesktopServices::openUrl(QUrl(QString::fromUtf8(IPDE_REPOSITORY_URL))); });
    auto *bug = menu->addAction("Report Bug…"); bug->setObjectName("ipdeReportBugAction");
    QObject::connect(bug, &QAction::triggered, window, [] { QDesktopServices::openUrl(QUrl("https://github.com/gnaservicesinc/ipde/issues/new")); });
    menu->addSeparator();
    auto *about = menu->addAction("About " + appName); about->setObjectName("ipdeAboutAction"); about->setMenuRole(QAction::AboutRole);
    QObject::connect(about, &QAction::triggered, window, [window, appName] { showAbout(window, appName); });
}
} // namespace IPDE
