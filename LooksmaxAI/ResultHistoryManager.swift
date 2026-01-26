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
    private let maxHistoryCount = 50  // Increased from 10 to keep more results
    
    private init() {}
    
    // Get the directory for storing thumbnails
    private var thumbnailsDirectory: URL {
        let documentsPath = FileManager.default.urls(for: .documentDirectory, in: .userDomainMask)[0]
        let thumbnailsPath = documentsPath.appendingPathComponent("ResultThumbnails")
        
        // Create directory if it doesn't exist
        if !FileManager.default.fileExists(atPath: thumbnailsPath.path) {
            do {
                try FileManager.default.createDirectory(at: thumbnailsPath, withIntermediateDirectories: true)
                
                // Mark directory to not be backed up to iCloud (keeps it local and persistent)
                var resourceValues = URLResourceValues()
                resourceValues.isExcludedFromBackup = true
                var mutablePath = thumbnailsPath
                try? mutablePath.setResourceValues(resourceValues)
                
                print("✅ Created thumbnails directory: \(thumbnailsPath.path)")
            } catch {
                print("❌ Failed to create thumbnails directory: \(error)")
            }
        }
        
        return thumbnailsPath
    }
    
    // Save thumbnail image and return the file path
    private func saveThumbnail(_ image: UIImage, id: String) -> String? {
        // Create a larger thumbnail (200x200 points) for better quality
        let thumbnailSize = CGSize(width: 200, height: 200)
        UIGraphicsBeginImageContextWithOptions(thumbnailSize, false, 0.0)
        image.draw(in: CGRect(origin: .zero, size: thumbnailSize))
        let thumbnail = UIGraphicsGetImageFromCurrentImageContext()
        UIGraphicsEndImageContext()
        
        guard let thumbnail = thumbnail,
              let imageData = thumbnail.jpegData(compressionQuality: 0.9) else {
            print("❌ Failed to create thumbnail image data")
            return nil
        }
        
        let fileName = "\(id).jpg"
        let filePath = thumbnailsDirectory.appendingPathComponent(fileName)
        
        do {
            // Use atomic write to ensure file is fully written before it can be read
            // This prevents race conditions where the file might be read before write completes
            try imageData.write(to: filePath, options: [.atomic])
            
            // Mark file to not be backed up to iCloud (keeps it local and persistent)
            var resourceValues = URLResourceValues()
            resourceValues.isExcludedFromBackup = true
            var mutableFilePath = filePath
            try? mutableFilePath.setResourceValues(resourceValues)
            
            // Return only the filename (relative path) instead of absolute path
            // This ensures the path works even if app container changes
            print("✅ Thumbnail saved: \(fileName)")
            return fileName
        } catch {
            print("❌ Failed to save thumbnail: \(error)")
            return nil
        }
    }
    
    // Load thumbnail from file path
    func loadThumbnail(for path: String) -> UIImage? {
        let fileURL: URL
        
        if path.hasPrefix("/") {
            // Legacy absolute path - try to load it, but also try relative path as fallback
            let absoluteURL = URL(fileURLWithPath: path)
            if FileManager.default.fileExists(atPath: absoluteURL.path) {
                fileURL = absoluteURL
            } else {
                // Absolute path doesn't exist, try relative path (filename only)
                let fileName = (path as NSString).lastPathComponent
                fileURL = thumbnailsDirectory.appendingPathComponent(fileName)
            }
        } else {
            // Relative path (filename) - resolve from thumbnails directory
            fileURL = thumbnailsDirectory.appendingPathComponent(path)
        }
        
        guard FileManager.default.fileExists(atPath: fileURL.path) else {
            print("⚠️ Thumbnail file not found at: \(fileURL.path)")
            // Try to find the file by filename only (in case path changed)
            if !path.hasPrefix("/") {
                // Already tried relative path, file doesn't exist
                return nil
            }
            // Try to find by filename
            let fileName = (path as NSString).lastPathComponent
            let alternativePath = thumbnailsDirectory.appendingPathComponent(fileName)
            if FileManager.default.fileExists(atPath: alternativePath.path) {
                print("✅ Found thumbnail using alternative path: \(alternativePath.path)")
                if let imageData = try? Data(contentsOf: alternativePath),
                   let image = UIImage(data: imageData) {
                    return image
                }
            }
            return nil
        }
        
        guard let imageData = try? Data(contentsOf: fileURL),
              let image = UIImage(data: imageData) else {
            print("⚠️ Failed to load thumbnail data from: \(fileURL.path)")
            return nil
        }
        
        return image
    }
    
    func saveResult(_ result: FaceAnalysisResults, frontImage: UIImage? = nil, gender: String? = nil) {
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
            thumbnailPath: thumbnailPath,
            gender: gender
        )
        
        history.insert(historyItem, at: 0)
        
        // Keep only last N results
        if history.count > maxHistoryCount {
            // Remove old thumbnails for items beyond max count
            let itemsToRemove = history.suffix(from: maxHistoryCount)
            for item in itemsToRemove {
                if let path = item.thumbnailPath {
                    // Handle both absolute and relative paths
                    let fileURL: URL
                    if path.hasPrefix("/") {
                        fileURL = URL(fileURLWithPath: path)
                    } else {
                        fileURL = thumbnailsDirectory.appendingPathComponent(path)
                    }
                    try? FileManager.default.removeItem(at: fileURL)
                }
            }
            history = Array(history.prefix(maxHistoryCount))
        }
        
        do {
            let encoded = try JSONEncoder().encode(history)
            UserDefaults.standard.set(encoded, forKey: historyKey)
            UserDefaults.standard.synchronize()  // Force immediate write to disk
            print("✅ Saved \(history.count) results to UserDefaults")
        } catch {
            print("❌ Failed to encode history: \(error)")
        }
    }
    
    func loadHistory() -> [ResultHistoryItem] {
        guard let data = UserDefaults.standard.data(forKey: historyKey),
              let history = try? JSONDecoder().decode([ResultHistoryItem].self, from: data) else {
            return []
        }
        return history
    }
    
    // Migrate old absolute paths to relative paths
    func migrateThumbnailPaths() {
        var history = loadHistory()
        var needsUpdate = false
        
        for i in 0..<history.count {
            if let path = history[i].thumbnailPath, path.hasPrefix("/") {
                // Convert absolute path to relative path (filename only)
                let fileName = (path as NSString).lastPathComponent
                // Create new item with relative path (preserve gender if it exists)
                let newItem = ResultHistoryItem(
                    id: history[i].id,
                    timestamp: history[i].timestamp,
                    result: history[i].result,
                    thumbnailPath: fileName,
                    gender: history[i].gender
                )
                history[i] = newItem
                needsUpdate = true
                print("🔄 Migrated thumbnail path: \(path) -> \(fileName)")
            }
        }
        
        if needsUpdate {
            do {
                let encoded = try JSONEncoder().encode(history)
                UserDefaults.standard.set(encoded, forKey: historyKey)
                UserDefaults.standard.synchronize()
                print("✅ Migrated \(history.count) thumbnail paths to relative paths")
            } catch {
                print("❌ Failed to migrate thumbnail paths: \(error)")
            }
        }
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
            // Handle both absolute and relative paths
            let fileURL: URL
            if path.hasPrefix("/") {
                fileURL = URL(fileURLWithPath: path)
            } else {
                fileURL = thumbnailsDirectory.appendingPathComponent(path)
            }
            try? FileManager.default.removeItem(at: fileURL)
        }
        
        history.removeAll { $0.id == id }
        
        do {
            let encoded = try JSONEncoder().encode(history)
            UserDefaults.standard.set(encoded, forKey: historyKey)
            UserDefaults.standard.synchronize()  // Force immediate write to disk
        } catch {
            print("❌ Failed to encode history after delete: \(error)")
        }
    }
    
    func clearHistory() {
        // Delete all thumbnail files before clearing history
        let history = loadHistory()
        for item in history {
            if let path = item.thumbnailPath {
                let fileURL: URL
                if path.hasPrefix("/") {
                    fileURL = URL(fileURLWithPath: path)
                } else {
                    fileURL = thumbnailsDirectory.appendingPathComponent(path)
                }
                try? FileManager.default.removeItem(at: fileURL)
            }
        }
        
        UserDefaults.standard.removeObject(forKey: historyKey)
        UserDefaults.standard.synchronize()
    }
}

struct ResultHistoryItem: Codable {
    let id: String
    let timestamp: Date
    let result: FaceAnalysisResults
    var thumbnailPath: String? // Path to saved thumbnail image (now stores relative path/filename)
    var gender: String? // Gender for gender-specific label display (optional for backward compatibility)
}




