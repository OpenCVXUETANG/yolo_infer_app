#include "mainwindow.h"
#include "ui_mainwindow.h"
#include "QFileDialog"

#pragma execution_character_set("utf-8")

MainWindow::MainWindow(QWidget *parent)
    : QMainWindow(parent)
    , ui(new Ui::MainWindow)
{
    ui->setupUi(this);
    ui->image_label->setText("输入图像显示");
    ui->image_label->setStyleSheet(
        "QLabel {"
        "  background-color: black;"
        "  color: #00FF00;"          // 绿色文字
        "  border: 1px solid #00FF00;" // 1px 绿色细边框
        "  padding: 4px;"
        "}"
    );
    ui->result_label->setStyleSheet(
        "QLabel {"
        "  background-color: black;"
        "  color: #00FF00;"          // 绿色文字
        "  border: 1px solid #00FF00;" // 1px 绿色细边框
        "  padding: 4px;"
        "}"
    );
}

MainWindow::~MainWindow()
{
    delete ui;
}

void MainWindow::on_imButton_clicked()
{
    QString data_file = QFileDialog::getOpenFileName(this, "choose image file", ".", tr("Data(*.png *.jpg *.mp4);"));

    if(data_file.isEmpty()){
        return;
    }
    cv::Mat frame = cv::imread(data_file.toStdString());
    cv::Mat rgb;
    cv::cvtColor(frame, rgb, cv::COLOR_BGR2RGB);
    QImage src(rgb.data, rgb.cols, rgb.rows,
               static_cast<int>(rgb.step), QImage::Format_RGB888);

    auto img = src.scaled(520, 640,
                          Qt::KeepAspectRatio,
                          Qt::SmoothTransformation);

    auto pixmap = QPixmap::fromImage(img.copy());
    ui->image_label->setPixmap(pixmap);

    image_path = data_file.toStdString();
}


void MainWindow::on_inferButton_clicked()
{
//    std::string engine_file = "D:/TensorRT-10.8/bin/yolov8_cu.engine";
    std::string engine_file = "D:/TensorRT-10.8/bin/yolov8_cu_int8.engine";

    this->infer_worker = new YOLOInferThread(engine_file, image_path, 0.25f);
    qRegisterMetaType<cv::Mat>("cv::Mat");
    qRegisterMetaType<std::vector<DetectResult>>("std::vector<DetectResult>");
    QObject::connect(infer_worker,SIGNAL(sendResult(cv::Mat, std::vector<DetectResult>)), this ,SLOT(postYOLOProcess(cv::Mat, std::vector<DetectResult>)));
    connect(infer_worker, &YOLOInferThread::finished, infer_worker, &QObject::deleteLater);
    infer_worker->start();
    ui->inferButton->setEnabled(false);
}

void MainWindow::postYOLOProcess(cv::Mat bgr, std::vector<DetectResult> results) {
    if(bgr.empty() && results.empty()) {
        ui->inferButton->setEnabled(true);
        return;
    }
    cv::Mat rgb;
    cv::cvtColor(bgr, rgb, cv::COLOR_BGR2RGB);
    QImage src(rgb.data, rgb.cols, rgb.rows,
               static_cast<int>(rgb.step), QImage::Format_RGB888);

    auto img = src.scaled(520, 640,
                          Qt::KeepAspectRatio,
                          Qt::SmoothTransformation);

    auto pixmap = QPixmap::fromImage(img.copy());
    ui->result_label->setPixmap(pixmap);

}

