#include "help_support.h"

#include <QElapsedTimer>
#include <QTemporaryDir>
#include <QThread>
#include <iostream>
#include <stdexcept>

namespace {
void require(bool condition, const char *message) { if (!condition) throw std::runtime_error(message); }
void write(const QString &path, const QByteArray &contents) {
    QFile file(path); require(file.open(QIODevice::WriteOnly), "cannot create fixture");
    require(file.write(contents) == contents.size(), "cannot write fixture");
}
QByteArray request(IPDE::HelpServer &server, const QByteArray &method, const QByteArray &path,
                   const QByteArray &host = {}, bool fragmented = false) {
    QTcpSocket socket; socket.connectToHost(QHostAddress::LocalHost, server.port());
    require(socket.waitForConnected(1000), "help client could not connect");
    const auto requestHost = host.isEmpty() ? "127.0.0.1:" + QByteArray::number(server.port()) : host;
    const auto message = method + " " + path + " HTTP/1.1\r\nHost: " + requestHost + "\r\n\r\n";
    if (fragmented) {
        socket.write(message.left(9)); socket.flush(); QCoreApplication::processEvents(QEventLoop::AllEvents, 10);
        socket.write(message.mid(9));
    } else socket.write(message);
    QByteArray response; QElapsedTimer timer; timer.start();
    while (timer.elapsed() < 2000) {
        QCoreApplication::processEvents(QEventLoop::AllEvents, 10);
        response += socket.readAll();
        if (socket.state() == QAbstractSocket::UnconnectedState) break;
        QThread::msleep(1);
    }
    response += socket.readAll(); require(response.startsWith("HTTP/1.1 "), "help response missing or timed out"); return response;
}
}

int main(int argc, char **argv) {
    QApplication app(argc, argv);
    try {
        QTemporaryDir root, outside; require(root.isValid() && outside.isValid(), "temporary docs directory unavailable");
        QDir directory(root.path()); require(directory.mkpath("help"), "cannot create help fixture");
        write(directory.filePath("help/index.html"), "<html>offline guide</html>");
        write(directory.filePath("help/app.js"), "console.log('local');");
        write(directory.filePath("private.txt"), "unsupported file");
        write(QDir(outside.path()).filePath("escape.html"), "outside content");
        require(QFile::link(QDir(outside.path()).filePath("escape.html"), directory.filePath("help/escape.html")), "cannot create containment fixture");
        IPDE::HelpServer server(root.path()); require(server.start(), "help server did not start");
        const auto path = server.url().path().toUtf8();
        const auto prefix = path.left(path.lastIndexOf("help/index.html"));
        auto response = request(server, "GET", path, {}, true);
        require(response.startsWith("HTTP/1.1 200") && response.endsWith("<html>offline guide</html>"), "fragmented help GET failed");
        require(response.contains("Content-Security-Policy:") && response.contains("Referrer-Policy: no-referrer"), "help browser restrictions missing");
        response = request(server, "HEAD", path);
        require(response.startsWith("HTTP/1.1 200") && response.endsWith("\r\n\r\n") && response.contains("Content-Length: 26"), "HEAD did not preserve length without body");
        require(request(server, "GET", prefix + "help/app.js").contains("Content-Type: text/javascript"), "JavaScript MIME type wrong");
        require(request(server, "GET", path, "example.test").startsWith("HTTP/1.1 403"), "nonlocal Host accepted");
        require(request(server, "GET", "/help/index.html").startsWith("HTTP/1.1 404"), "untokened path accepted");
        require(request(server, "POST", path).startsWith("HTTP/1.1 405"), "write method accepted");
        require(request(server, "GET", prefix + "help/%2e%2e/private.txt").startsWith("HTTP/1.1 403"), "encoded traversal accepted");
        require(request(server, "GET", prefix + "help/escape.html").startsWith("HTTP/1.1 404"), "symlink escaped docs root");
        require(request(server, "GET", prefix + "private.txt").startsWith("HTTP/1.1 404"), "unsupported file exposed");
        require(request(server, "GET", prefix + "help/").startsWith("HTTP/1.1 404"), "directory listing enabled");
        QMainWindow window; IPDE::installHelpMenu(&window, "Test app", "studio");
        require(window.findChild<QMenu *>("ipdeHelpMenu") && window.findChild<QAction *>("interactiveHelpAction")
            && window.findChild<QAction *>("ipdeAboutAction") && window.findChild<QAction *>("ipdeReportBugAction"), "common Help actions missing");
        require(IPDE::buildInformation("Test app").contains("Build flags:") && IPDE::buildInformation("Test app").contains("Qt runtime:"), "About build metadata missing");
        std::cout << "Offline help: real HTTP GET/HEAD, fragmented requests, host/token/path containment, MIME, read-only access and common menus passed\n";
        return 0;
    } catch (const std::exception &error) { std::cerr << error.what() << '\n'; return 1; }
}
