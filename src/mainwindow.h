#ifndef MAINWINDOW_H
#define MAINWINDOW_H

#include <QMainWindow>
#include "yolo_infer_thread.h"

QT_BEGIN_NAMESPACE
namespace Ui { class MainWindow; }
QT_END_NAMESPACE

class MainWindow : public QMainWindow
{
    Q_OBJECT

public:
    MainWindow(QWidget *parent = nullptr);
    ~MainWindow();


private slots:
    void on_imButton_clicked();

    void on_inferButton_clicked();
    void postYOLOProcess(cv::Mat bgr, std::vector<DetectResult> results);

private:
    Ui::MainWindow *ui;
    std::string image_path;
    YOLOInferThread *infer_worker{nullptr};
};
#endif // MAINWINDOW_H
