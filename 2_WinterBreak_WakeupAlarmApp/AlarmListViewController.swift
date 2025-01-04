import UIKit

// Delegate Protocol 정의
protocol AlarmListViewControllerDelegate: AnyObject {
    func updateAlarms(_ updatedAlarms: [Alarm])
}


class AlarmListViewController: UITableViewController {
    
    var alarms: [Alarm] = [] // 전달받은 알람 데이터
    weak var delegate: AlarmListViewControllerDelegate? // Delegate 속성
    
    override func viewDidLoad() {
        super.viewDidLoad()
        print("알람 리스트 로드됨: \(alarms)")
    }

    // 테이블 뷰의 셀 개수
    override func tableView(_ tableView: UITableView, numberOfRowsInSection section: Int) -> Int {
        return alarms.count
    }
    
    // 테이블 뷰의 셀 구성
    override func tableView(_ tableView: UITableView, cellForRowAt indexPath: IndexPath) -> UITableViewCell {
        let cell = tableView.dequeueReusableCell(withIdentifier: "AlarmCell", for: indexPath)
        let alarm = alarms[indexPath.row]

        // 날짜/시간 포맷 설정
        let formatter = DateFormatter()
        formatter.dateFormat = "yyyy-MM-dd HH:mm"
        let formattedDate = formatter.string(from: alarm.date)

        // 텍스트 설정
        cell.textLabel?.text = alarm.message
        cell.detailTextLabel?.text = "\(formattedDate) | Voice: \(alarm.voiceType)"

        return cell
    }
    
    // 스와이프 삭제 기능 추가
    override func tableView(_ tableView: UITableView, commit editingStyle: UITableViewCell.EditingStyle, forRowAt indexPath: IndexPath) {
        if editingStyle == .delete {
            
            let alarmToDelete = alarms[indexPath.row]
            
            // 데이터 소스에서 해당 알람 삭제
            alarms.remove(at: indexPath.row)
            
            // UNUserNotificationCenter에서 알림 삭제
             UNUserNotificationCenter.current().removePendingNotificationRequests(withIdentifiers: [alarmToDelete.id.uuidString])
            
            // 테이블 뷰에서 셀 삭제
            tableView.deleteRows(at: [indexPath], with: .fade)
            
            // Delegate를 통해 ViewController에 업데이트된 alarms 전달
            delegate?.updateAlarms(alarms)
            
            print("알람 삭제됨: \(alarms)")
        }
    }
    
}
