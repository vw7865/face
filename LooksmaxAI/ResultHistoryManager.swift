//
//  ResultHistoryManager.swift
//  LooksmaxAI
//
//  Created by Admin on 2026. 01. 07..
//

import Foundation
import UIKit

class ResultHistoryManager {
    static let shared = ResultHistoryManager()
    
    private let historyKey = "com.looksmaxai.resultHistory"
    private let maxHistoryCount = 10
    
    private init() {}
    
    // Get the directory for storing thumbnails
    private var thumbnailsDirectory: URL {
        let documentsPath = FileManager.default.urls(for: .documentDirectory, in: .userDomainMask)[0]
        let thumbnailsPath = documentsPath.appendingPathComponent("ResultThumbnails")
        
        // Create directory if it doesn't exist
        if !FileManager.default.fileExists(atPath: thumbnailsPath.path) {
            try? FileManager.default.createDirectory(at: thumbnailsPath, withIntermediateDirectories: true)
        }
        
        return thumbnailsPath
    }
    
    // Save thumbnail image and return the file path
    private func saveThumbnail(_ image: UIImage, id: String) -> String? {
        // Create a small thumbnail (100x100 points)
        let thumbnailSize = CGSize(width: 100, height: 100)
        UIGraphicsBeginImageContextWithOptions(thumbnailSize, false, 0.0)
        image.draw(in: CGRect(origin: .zero, size: thumbnailSize))
        let thumbnail = UIGraphicsGetImageFromCurrentImageContext()
        UIGraphicsEndImageContext()
        
        guard let thumbnail = thumbnail,
              let imageData = thumbnail.jpegData(compressionQuality: 0.8) else {
            return nil
        }
        
        let filePath = thumbnailsDirectory.appendingPathComponent("\(id).jpg")
        
        do {
            try imageData.write(to: filePath)
            return filePath.path
        } catch {
            print("❌ Failed to save thumbnail: \(error)")
            return nil
        }
    }
    
    // Load thumbnail from file path
    func loadThumbnail(for path: String) -> UIImage? {
        guard FileManager.default.fileExists(atPath: path) else {
            return nil
        }
        
        guard let imageData = try? Data(contentsOf: URL(fileURLWithPath: path)),
              let image = UIImage(data: imageData) else {
            return nil
        }
        
        return image
    }
    
    func saveResult(_ result: FaceAnalysisResults, frontImage: UIImage? = nil) {
        var history = loadHistory()
        
        let id = UUID().uuidString
        var thumbnailPath: String? = nil
        
        // Save thumbnail if front image is provided
        if let frontImage = frontImage {
            thumbnailPath = saveThumbnail(frontImage, id: id)
        }
        
        // Add timestamp
        let historyItem = ResultHistoryItem(
            id: id,
            timestamp: Date(),
            result: result,
            thumbnailPath: thumbnailPath
        )
        
        history.insert(historyItem, at: 0)
        
        // Keep only last N results
        if history.count > maxHistoryCount {
            // Remove old thumbnails for items beyond max count
            let itemsToRemove = history.suffix(from: maxHistoryCount)
            for item in itemsToRemove {
                if let path = item.thumbnailPath {
                    try? FileManager.default.removeItem(atPath: path)
                }
            }
            history = Array(history.prefix(maxHistoryCount))
        }
        
        if let encoded = try? JSONEncoder().encode(history) {
            UserDefaults.standard.set(encoded, forKey: historyKey)
        }
    }
    
    func loadHistory() -> [ResultHistoryItem] {
        guard let data = UserDefaults.standard.data(forKey: historyKey),
              let history = try? JSONDecoder().decode([ResultHistoryItem].self, from: data) else {
            return []
        }
        return history
    }
    
    func getLatestResult() -> FaceAnalysisResults? {
        let history = loadHistory()
        return history.first?.result
    }
    
    func deleteResult(id: String) {
        var history = loadHistory()
        
        // Find and remove thumbnail file
        if let item = history.first(where: { $0.id == id }),
           let path = item.thumbnailPath {
            try? FileManager.default.removeItem(atPath: path)
        }
        
        history.removeAll { $0.id == id }
        
        if let encoded = try? JSONEncoder().encode(history) {
            UserDefaults.standard.set(encoded, forKey: historyKey)
        }
    }
    
    func clearHistory() {
        UserDefaults.standard.removeObject(forKey: historyKey)
    }
}

struct ResultHistoryItem: Codable {
    let id: String
    let timestamp: Date
    let result: FaceAnalysisResults
    let thumbnailPath: String? // Path to saved thumbnail image
}




