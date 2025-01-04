import UIKit
import UserNotifications

class ViewController: UIViewController, UIPickerViewDelegate, UIPickerViewDataSource, UITextFieldDelegate {
    
    override func viewDidLoad() {
        super.viewDidLoad()
        voicePicker.delegate = self
        voicePicker.dataSource = self
        alarmTextField.delegate = self
        requestNotificationPermission()
    }
    
    @IBOutlet weak var voicePicker: UIPickerView!
    @IBOutlet weak var datePicker: UIDatePicker!
    @IBOutlet weak var alarmTextField: UITextField!
    @IBOutlet weak var saveButton: UIButton!
    
    // 앱 내에서 간단한 팝업을 표시하는 함수
    func showAlert(title: String, message: String) {
        let alert = UIAlertController(title: title, message: message, preferredStyle: .alert)
        let okAction = UIAlertAction(title: "확인", style: .default, handler: nil)
        alert.addAction(okAction)
        self.present(alert, animated: true, completion: nil)
    }
    
    var alarms: [Alarm] = []
    
    // 알람 권한 요청
    func requestNotificationPermission() {
        UNUserNotificationCenter.current().requestAuthorization(options: [.alert, .sound, .badge]) {granted, error in
            if granted {
                print("알람 권한 허용")
            } else {
                print("알람 권한 거부")
            }
        }
    }
    
    // 알람을 저장하는 함수
    @IBAction func saveAlarm(_ sender: Any) {
        guard let message = alarmTextField.text, !message.isEmpty else {
            // 텍스트 입력이 없을 경우 Alert 표시
            showAlert(title: "Saving Failed", message: "Please Insert Alarm TextField before you save alarm.")
            return
        }
        
        let selectedDate = datePicker.date
        let selectedVoice = ["Female", "Male"][voicePicker.selectedRow(inComponent: 0)]
        let newAlarm = Alarm(date: selectedDate, message: message, voiceType: selectedVoice)
        alarms.append(newAlarm)
        
        // Local Notification 예약
        scheduleNotification(for: newAlarm)
        
        // 알람 저장 성공 Alert 표시
        showAlert(title: "Saving Success", message: "Your Alarm is successfully saved.")
        print("알람이 저장되었습니다. \(newAlarm)")
    }
    
    // 알림 예약
    func scheduleNotification(for alarm: Alarm) {
        let content = UNMutableNotificationContent() // 알림의 내용을 정의한다.
        content.title = "알람"
        content.body = alarm.message
        content.sound = .default
        content.userInfo = ["id": alarm.id.uuidString, "message": alarm.message, "voiceType": alarm.voiceType]
        
        // 알림이 울릴 날짜와 시간을 설정한다
        let triggerDate = Calendar.current.dateComponents([.year, .month, .day, .hour, .minute], from: alarm.date)
        // 알림이 설정된 시간에 실행되도록 트리거를 만든다
        let trigger = UNCalendarNotificationTrigger(dateMatching: triggerDate, repeats: false)
        // 알림 요청 생성
        let request = UNNotificationRequest(identifier: alarm.id.uuidString, content: content, trigger: trigger)
        // 알림 요청 등록
        UNUserNotificationCenter.current().add(request) { error in
            if let error = error {
                print("알림 예약 실패: \(error.localizedDescription)")
            } else {
                print("알림 예약 성공: \(alarm)")
            }
        }
    }

    override func prepare(for segue: UIStoryboardSegue, sender: Any?) {
        if segue.identifier == "showAlarmList" {
            let destinationVC = segue.destination as! AlarmListViewController
            destinationVC.alarms = alarms
            destinationVC.delegate = self // Delegate 연결
        }
    }
    
    func numberOfComponents(in pickerView: UIPickerView) -> Int {
        return 1
    }

    func pickerView(_ pickerView: UIPickerView, numberOfRowsInComponent component: Int) -> Int {
        return 2
    }
    
    func pickerView(_ pickerView: UIPickerView, titleForRow row: Int, forComponent component: Int) -> String? {
        let voices = ["Female", "Male"]
        return voices[row]
    }
    
    // 최대 40자 입력 제한
    func textField(_ textField: UITextField, shouldChangeCharactersIn range: NSRange, replacementString string: String) -> Bool {
        let currentText = textField.text ?? ""
        guard let stringRange = Range(range, in: currentText) else { return false }
        let updatedText = currentText.replacingCharacters(in: stringRange, with: string)
        
        // 입력 제한
        if updatedText.count > 50 {
            showAlert(title: "Insert Limit", message: "Max TextField Length is 50 letters.")
            return false
        }
        return true
    }

}

// AlarmListViewControllerDelegate 구현
extension ViewController: AlarmListViewControllerDelegate {
    func updateAlarms(_ updatedAlarms: [Alarm]) {
        self.alarms = updatedAlarms
        print("ViewController의 알람 데이터 업데이트됨: \(alarms)")
    }
}

