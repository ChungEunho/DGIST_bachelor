import Foundation

struct Alarm: Codable { // Codable은 저장할 때 유용
    let id: UUID // 개별 알람 고유 식별자
    let date: Date
    let message: String
    let voiceType: String
    
    init(date: Date, message: String, voiceType: String) {
        self.id = UUID()
        self.date = date
        self.message = message
        self.voiceType = voiceType
    }
}
