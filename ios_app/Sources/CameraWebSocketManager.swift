import Foundation
import AVFoundation
import UIKit
import Combine

struct InferenceResponse: Decodable {
    let keypoints: [[Double]]?
    let words: String?
    let sentence: String?
}

final class CameraWebSocketManager: NSObject, ObservableObject {
    @Published var isConnected: Bool = false
    @Published var statusText: String = "연결 대기 중"
    @Published var detectedWords: String = "대기 중..."
    @Published var refinedSentence: String = "PC 서버 연결 후 수어 동작을 시작하세요."
    @Published var keypoints: [[Double]] = []

    let captureSession = AVCaptureSession()
    private let videoOutput = AVCaptureVideoDataOutput()
    private let sessionQueue = DispatchQueue(label: "camera.session.queue")
    private let ciContext = CIContext()

    private var webSocketTask: URLSessionWebSocketTask?
    private var urlSession: URLSession?
    private var isWaitingResponse = false
    private var lastSentTime: CFTimeInterval = 0

    override init() {
        super.init()
        setupCamera()
    }

    func setupCamera() {
        sessionQueue.async { [weak self] in
            guard let self = self else { return }
            self.captureSession.beginConfiguration()
            self.captureSession.sessionPreset = .vga640x480

            guard let frontCamera = AVCaptureDevice.default(.builtInWideAngleCamera, for: .video, position: .front),
                  let input = try? AVCaptureDeviceInput(device: frontCamera),
                  self.captureSession.canAddInput(input) else {
                DispatchQueue.main.async {
                    self.statusText = "전면 카메라를 열 수 없습니다"
                }
                self.captureSession.commitConfiguration()
                return
            }
            self.captureSession.addInput(input)

            self.videoOutput.videoSettings = [
                kCVPixelBufferPixelFormatTypeKey as String: kCVPixelFormatType_32BGRA
            ]
            self.videoOutput.alwaysDiscardsLateVideoFrames = true
            self.videoOutput.setSampleBufferDelegate(self, queue: self.sessionQueue)

            if self.captureSession.canAddOutput(self.videoOutput) {
                self.captureSession.addOutput(self.videoOutput)
            }

            if let connection = self.videoOutput.connection(with: .video) {
                if connection.isVideoOrientationSupported {
                    connection.videoOrientation = .portrait
                }
                if connection.isVideoMirroredSupported {
                    connection.isVideoMirrored = true
                }
            }

            self.captureSession.commitConfiguration()
            self.captureSession.startRunning()
        }
    }

    func connect(serverIP: String, port: String = "8000") {
        disconnect()
        let trimmedIP = serverIP.trimmingCharacters(in: .whitespacesAndNewlines)
        guard let url = URL(string: "ws://\(trimmedIP):\(port)/ws") else {
            statusText = "잘못된 서버 IP 형식"
            return
        }

        statusText = "연결 시도 중 (\(trimmedIP))..."
        let config = URLSessionConfiguration.default
        urlSession = URLSession(configuration: config, delegate: self, delegateQueue: OperationQueue())
        webSocketTask = urlSession?.webSocketTask(with: url)
        webSocketTask?.resume()
        receiveLoop()
    }

    func disconnect() {
        webSocketTask?.cancel(with: .goingAway, reason: nil)
        webSocketTask = nil
        urlSession?.invalidateAndCancel()
        urlSession = nil
        isWaitingResponse = false
        DispatchQueue.main.async {
            self.isConnected = false
            self.statusText = "연결 끊김"
            self.keypoints = []
        }
    }

    private func receiveLoop() {
        webSocketTask?.receive { [weak self] result in
            guard let self = self else { return }
            self.isWaitingResponse = false

            switch result {
            case .success(let message):
                if case .string(let text) = message,
                   let data = text.data(using: .utf8),
                   let decoded = try? JSONDecoder().decode(InferenceResponse.self, from: data) {
                    DispatchQueue.main.async {
                        self.keypoints = decoded.keypoints ?? []
                        if let w = decoded.words, !w.isEmpty {
                            self.detectedWords = w
                        }
                        if let s = decoded.sentence, !s.isEmpty {
                            self.refinedSentence = s
                        }
                    }
                }
                self.receiveLoop()

            case .failure(let error):
                DispatchQueue.main.async {
                    self.isConnected = false
                    self.statusText = "연결 종료: \(error.localizedDescription)"
                }
            }
        }
    }
}

extension CameraWebSocketManager: URLSessionWebSocketDelegate {
    func urlSession(_ session: URLSession, webSocketTask: URLSessionWebSocketTask, didOpenWithProtocol protocol: String?) {
        DispatchQueue.main.async {
            self.isConnected = true
            self.statusText = "PC 서버 연결됨 (실시간 분석 중)"
        }
    }

    func urlSession(_ session: URLSession, webSocketTask: URLSessionWebSocketTask, didCloseWith closeCode: URLSessionWebSocketCloseCode, reason: Data?) {
        DispatchQueue.main.async {
            self.isConnected = false
            self.statusText = "서버 연결 닫힘"
        }
    }
}

extension CameraWebSocketManager: AVCaptureVideoDataOutputSampleBufferDelegate {
    func captureOutput(_ output: AVCaptureOutput, didOutput sampleBuffer: CMSampleBuffer, from connection: AVCaptureConnection) {
        guard isConnected, !isWaitingResponse else { return }

        // 최대 ~25 FPS로 전송 빈도 제한
        let now = CACurrentMediaTime()
        guard now - lastSentTime >= 0.04 else { return }

        guard let pixelBuffer = CMSampleBufferGetImageBuffer(sampleBuffer) else { return }
        let ciImage = CIImage(cvPixelBuffer: pixelBuffer)
        guard let cgImage = ciContext.createCGImage(ciImage, from: ciImage.extent) else { return }

        let uiImage = UIImage(cgImage: cgImage)
        guard let jpegData = uiImage.jpegData(compressionQuality: 0.6) else { return }

        lastSentTime = now
        isWaitingResponse = true
        webSocketTask?.send(.data(jpegData)) { [weak self] error in
            if error != nil {
                self?.isWaitingResponse = false
            }
        }
    }
}