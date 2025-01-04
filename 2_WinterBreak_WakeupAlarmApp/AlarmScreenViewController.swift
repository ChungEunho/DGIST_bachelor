import UIKit
import AVFoundation

protocol AlarmScreenDelegate: AnyObject {
    func deleteAlarm(with id: UUID)
}

class AlarmScreenViewController: UIViewController {
    weak var delegate: AlarmScreenDelegate?
    var alarmID: UUID? // 알람의 고유 ID
    var alarmMessage: String?
    var voiceType: String?
    var synthesizer = AVSpeechSynthesizer()
    var repeatCount = 0
    let maxRepeats = 12 // 2분 동안 10초 간격으로 반복
    var alarmTimer: Timer?

    override func viewDidLoad() {
        super.viewDidLoad()
        startAlarm()
    }

    func startAlarm() {
        guard let message = alarmMessage, let voiceType = voiceType else {
            print("알람 메시지 또는 목소리 타입이 설정되지 않았습니다.")
            return
        }
        print("알람 시작 - 메시지: \(message), 목소리 타입: \(voiceType)") // 디버깅 출력
        let utterance = AVSpeechUtterance(string: message)
        if voiceType == "Female" {
            utterance.voice = AVSpeechSynthesisVoice(identifier: "com.apple.ttsbundle.Samantha-compact")
        } else {
            utterance.voice = AVSpeechSynthesisVoice(identifier: "com.apple.ttsbundle.Daniel-compact")
        }
        utterance.rate = 0.5

        alarmTimer = Timer.scheduledTimer(withTimeInterval: 7, repeats: true) { [weak self] timer in
            guard let self = self else { return }
            if self.repeatCount >= self.maxRepeats {
                timer.invalidate()
                self.dismissAlarm()
            } else {
                self.synthesizer.speak(utterance)
                self.repeatCount += 1
            }
        }
    }

    func stopAlarm() {
        alarmTimer?.invalidate()
        alarmTimer = nil
        synthesizer.stopSpeaking(at: .immediate)
    }

    @IBAction func dismissAlarm() {
        stopAlarm()
        if let alarmID = alarmID {
            delegate?.deleteAlarm(with: alarmID)
        }
        dismiss(animated: true, completion: nil)
    }
}
