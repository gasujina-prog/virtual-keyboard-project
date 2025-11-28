# main.py

import sys
import time
from PyQt6.QtWidgets import QApplication, QMainWindow, QWidget, QVBoxLayout, QLabel
from PyQt6.QtCore import QThread, QObject, pyqtSignal, QSize, Qt
from PyQt6.QtGui import QImage, QPixmap

import config


# ----------------------------------------------------
# 1. 워커 클래스
# ----------------------------------------------------
class VisionWorker(QObject):
    frame_ready = pyqtSignal(QImage)
    gesture_detected = pyqtSignal(str)
    running = True

    def run(self):
        print("Vision Worker Thread Started.")
        while self.running:
            # 여기에 OpenCV로 카메라 프레임을 읽고 MediaPipe 처리하는 코드를 작성합니다.
            # 여기서 실제 프레임과 제스처 신호가 UI로 emit 되어야 합니다.
            # self.frame_ready.emit(processed_qimage)
            # self.gesture_detected.emit(gesture_name)

            time.sleep(1 / config.FRAME_RATE)
    def stop(self):
        self.running = False
        # ----------------------------------------------------


# 2. 메인 윈도우 (UI/통합 담당)
# ----------------------------------------------------
class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("가상 키보드 및 제스처 인식 시스템")
        self.setMinimumSize(QSize(config.VIDEO_WIDTH + 100, config.VIDEO_HEIGHT + 100))

        central_widget = QWidget()
        self.setCentralWidget(central_widget)

        layout = QVBoxLayout(central_widget)
        self.video_label = QLabel("비전 모듈 대기 중...")
        self.video_label.setFixedSize(config.VIDEO_WIDTH, config.VIDEO_HEIGHT)
        self.video_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(self.video_label)

        self.start_vision_thread()

    def start_vision_thread(self):
        self.thread = QThread()
        self.worker = VisionWorker()

        self.worker.moveToThread(self.thread)

        self.thread.started.connect(self.worker.run)

        # 워커의 신호를 받아 처리할 메서드(슬롯) 연결
        self.worker.frame_ready.connect(self.update_video_frame)
        self.worker.gesture_detected.connect(self.handle_gesture)

        self.thread.start()
        print("Main UI Thread Started.")

    def update_video_frame(self, qimage):
        self.video_label.setPixmap(QPixmap.fromImage(qimage))

    def handle_gesture(self, gesture_name):
        #여기에 매핑 테이블 조회 및 키 입력 실행 로직을 구현합니다.
        pass

    def closeEvent(self, event):
        # 종료 시 쓰레드를 안전하게 정리합니다.
        self.worker.stop()
        self.worker.deleteLater()
        self.thread.quit()
        self.thread.wait()
        super().closeEvent(event)


# ----------------------------------------------------
# 3. 애플리케이션 실행
# ----------------------------------------------------
if __name__ == '__main__':
    app = QApplication(sys.argv)

    window = MainWindow()
    window.show()
    sys.exit(app.exec())