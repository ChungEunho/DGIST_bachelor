import UIKit
import UserNotifications

@main
class AppDelegate: UIResponder, UIApplicationDelegate, UNUserNotificationCenterDelegate {


    func application(_ application: UIApplication, didFinishLaunchingWithOptions launchOptions: [UIApplication.LaunchOptionsKey: Any]?) -> Bool {
        // Notification Center Delegate 설정
        UNUserNotificationCenter.current().delegate = self
        
        // 알림 권한 요청
        requestNotificationPermission()
        
        return true
    }
    
    // 알림 권한 요청
    private func requestNotificationPermission() {
        let center = UNUserNotificationCenter.current()
        center.requestAuthorization(options: [.alert, .sound, .badge]) { granted, error in
            if granted {
                print("알림 권한 허용됨")
            } else {
                print("알림 권한 거부됨")
            }
        }
    }
    
    // 포그라운드에서도 알림 표시
    func userNotificationCenter(_ center: UNUserNotificationCenter, willPresent notification: UNNotification, withCompletionHandler completionHandler: @escaping (UNNotificationPresentationOptions) -> Void) {
        // iOS 14 이상에서는 .banner와 .list를 사용
        if #available(iOS 14.0, *) {
            completionHandler([.banner, .sound]) // 배너로 표시 및 소리 재생
        } else {
            completionHandler([.alert, .sound]) // iOS 14 미만에서는 alert 사용
        }
    }
    
    // 알림 클릭 시 호출
    func userNotificationCenter(_ center: UNUserNotificationCenter, didReceive response: UNNotificationResponse, withCompletionHandler completionHandler: @escaping () -> Void) {
        let userInfo = response.notification.request.content.userInfo
        
        // 알람 데이터 가져오기
        if let message = userInfo["message"] as? String, let voiceType = userInfo["voiceType"] as? String, let idString = userInfo["id"] as? String, let alarmID = UUID(uuidString: idString) {
            showAlarmScreen(message: message, voiceType: voiceType, alarmID: alarmID)
            print("알람 데이터 전달 완료")
        } else {
            print("알람 데이터 전달 실패")
        }
        
        completionHandler()
    }
    
    // Custom Alarm Screen 표시
    private func showAlarmScreen(message: String, voiceType: String, alarmID: UUID) {
        let storyboard = UIStoryboard(name: "Main", bundle: nil)
        if let alarmVC = storyboard.instantiateViewController(withIdentifier: "AlarmScreenViewController") as? AlarmScreenViewController {
            alarmVC.alarmMessage = message
            alarmVC.voiceType = voiceType
            alarmVC.alarmID = alarmID
            alarmVC.delegate = self // Delegate 설정
            
            // 현재 활성화된 UIWindowScene을 사용하여 화면 표시
            if let windowScene = UIApplication.shared.connectedScenes.first as? UIWindowScene,
               let rootVC = windowScene.windows.first?.rootViewController {
                rootVC.present(alarmVC, animated: true, completion: nil)
            } else {
                print("Active UIWindowScene이 없습니다.")
            }
        }
    }
    
    // MARK: UISceneSession Lifecycle

    func application(_ application: UIApplication, configurationForConnecting connectingSceneSession: UISceneSession, options: UIScene.ConnectionOptions) -> UISceneConfiguration {
        // Called when a new scene session is being created.
        // Use this method to select a configuration to create the new scene with.
        return UISceneConfiguration(name: "Default Configuration", sessionRole: connectingSceneSession.role)
    }

    func application(_ application: UIApplication, didDiscardSceneSessions sceneSessions: Set<UISceneSession>) {
        // Called when the user discards a scene session.
        // If any sessions were discarded while the application was not running, this will be called shortly after application:didFinishLaunchingWithOptions.
        // Use this method to release any resources that were specific to the discarded scenes, as they will not return.
    }

}

// dismiss 버튼 누를 시, 알람 삭제 기능 구현
extension AppDelegate: AlarmScreenDelegate {
    func deleteAlarm(with id: UUID) {
        // UIWindowScene에서 rootViewController 가져오기
        if let windowScene = UIApplication.shared.connectedScenes.first(where: { $0.activationState == .foregroundActive }) as? UIWindowScene,
           let rootVC = windowScene.windows.first(where: { $0.isKeyWindow })?.rootViewController as? ViewController {
            rootVC.alarms.removeAll { $0.id == id }
            print("알람 삭제됨: \(id)")
        } else {
            print("Active UIWindowScene을 찾을 수 없습니다.")
        }
    }
}
