import SwiftUI
import AVFoundation

struct CameraPreviewView: UIViewRepresentable {
    let session: AVCaptureSession

    func makeUIView(context: Context) -> PreviewUIView {
        let view = PreviewUIView()
        view.videoPreviewLayer.session = session
        view.videoPreviewLayer.videoGravity = .resize
        return view
    }

    func updateUIView(_ uiView: PreviewUIView, context: Context) {}

    final class PreviewUIView: UIView {
        override class var layerClass: AnyClass { AVCaptureVideoPreviewLayer.self }
        var videoPreviewLayer: AVCaptureVideoPreviewLayer { layer as! AVCaptureVideoPreviewLayer }
    }
}

struct ContentView: View {
    @StateObject private var manager = CameraWebSocketManager()
    @AppStorage("pc_server_ip") private var serverIP: String = "192.168.0.10"

    private let skeletonEdges: [(Int, Int)] = [
        (5, 6), (5, 7), (7, 9), (6, 8), (8, 10), (5, 11), (6, 12), (11, 12)
    ]

    var body: some View {
        ZStack {
            Color(red: 0.06, green: 0.09, blue: 0.16).ignoresSafeArea()

            VStack(spacing: 16) {
                // 상단 서버 연결 바
                HStack(spacing: 12) {
                    Circle()
                        .fill(manager.isConnected ? Color.green : Color.red)
                        .frame(width: 10, height: 10)

                    Text(manager.statusText)
                        .font(.subheadline.bold())
                        .foregroundColor(.white)

                    Spacer()

                    TextField("Windows PC IP (예: 192.168.0.15)", text: $serverIP)
                        .textFieldStyle(.roundedBorder)
                        .keyboardType(.decimalPad)
                        .frame(width: 220)

                    Button(manager.isConnected ? "연결 해제" : "서버 연결") {
                        if manager.isConnected {
                            manager.disconnect()
                        } else {
                            manager.connect(serverIP: serverIP)
                        }
                    }
                    .buttonStyle(.borderedProminent)
                    .tint(manager.isConnected ? .red : .blue)
                }
                .padding(.horizontal, 20)
                .padding(.top, 12)

                // 중앙 카메라 프리뷰 + 실시간 관절 스켈레톤 오버레이 (4:3 비율 고정)
                ZStack {
                    CameraPreviewView(session: manager.captureSession)

                    Canvas { context, size in
                        let kpts = manager.keypoints
                        guard kpts.count >= 17 else { return }

                        // 뼈대 선 그리기
                        for (i, j) in skeletonEdges {
                            let p1 = kpts[i]
                            let p2 = kpts[j]
                            if p1.count >= 3, p2.count >= 3, p1[2] > 0.45, p2[2] > 0.45 {
                                var path = Path()
                                path.move(to: CGPoint(x: p1[0] * size.width, y: p1[1] * size.height))
                                path.addLine(to: CGPoint(x: p2[0] * size.width, y: p2[1] * size.height))
                                context.stroke(path, with: .color(.cyan), lineWidth: 3)
                            }
                        }

                        // 관절 점 그리기 (양손목 9, 10번은 빨간색 강조)
                        for (idx, pt) in kpts.enumerated() where pt.count >= 3 && pt[2] > 0.45 {
                            let isWrist = (idx == 9 || idx == 10)
                            let radius: CGFloat = isWrist ? 7 : 5
                            let rect = CGRect(
                                x: pt[0] * size.width - radius,
                                y: pt[1] * size.height - radius,
                                width: radius * 2,
                                height: radius * 2
                            )
                            context.fill(Path(ellipseIn: rect), with: .color(isWrist ? .pink : .green))
                        }
                    }
                }
                .aspectRatio(3.0 / 4.0, contentMode: .fit)
                .background(Color.black)
                .clipShape(RoundedRectangle(cornerRadius: 20))
                .overlay(
                    RoundedRectangle(cornerRadius: 20)
                        .stroke(Color.white.opacity(0.15), lineWidth: 1)
                )

                // 하단 수어 인식 자막 패널
                VStack(alignment: .leading, spacing: 10) {
                    Text("인식된 수어 단어 (GLOSSES)")
                        .font(.caption.bold())
                        .foregroundColor(.gray)
                    Text(manager.detectedWords)
                        .font(.title3.bold())
                        .foregroundColor(.cyan)

                    Divider().background(Color.white.opacity(0.15))

                    Text("번역된 문장 (REFINED SENTENCE)")
                        .font(.caption.bold())
                        .foregroundColor(.gray)
                    Text(manager.refinedSentence)
                        .font(.title2.bold())
                        .foregroundColor(.green)
                }
                .frame(maxWidth: .infinity, alignment: .leading)
                .padding(20)
                .background(Color(red: 0.12, green: 0.16, blue: 0.23))
                .clipShape(RoundedRectangle(cornerRadius: 18))
                .padding(.horizontal, 20)
                .padding(.bottom, 16)
            }
        }
    }
}