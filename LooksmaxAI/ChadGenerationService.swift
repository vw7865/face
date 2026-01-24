//
//  ChadGenerationService.swift
//  LooksmaxAI
//
//  Created by Admin on 2026. 01. 15..
//

import Foundation
import UIKit

class ChadGenerationService {
    static let shared = ChadGenerationService()
    
    // Deployed backend URL on Railway
    private let baseURL = "https://looksmax-backend-production.up.railway.app"
    
    // Custom URLSession with timeout for ML model processing
    private lazy var urlSession: URLSession = {
        let configuration = URLSessionConfiguration.default
        configuration.timeoutIntervalForRequest = 120.0 // 120 seconds for ML processing
        configuration.timeoutIntervalForResource = 120.0
        return URLSession(configuration: configuration)
    }()
    
    private init() {}
    
    // MARK: - Generate Chad Version
    func generateChadVersion(frontImage: UIImage, sideImage: UIImage, gender: String, completion: @escaping (Result<UIImage, Error>) -> Void) {
        guard let frontImageData = frontImage.jpegData(compressionQuality: 0.8) else {
            completion(.failure(NSError(domain: "ChadGeneration", code: -1, userInfo: [NSLocalizedDescriptionKey: "Failed to convert front image"])))
            return
        }
        
        guard let sideImageData = sideImage.jpegData(compressionQuality: 0.8) else {
            completion(.failure(NSError(domain: "ChadGeneration", code: -1, userInfo: [NSLocalizedDescriptionKey: "Failed to convert side image"])))
            return
        }
        
        guard let url = URL(string: "\(baseURL)/api/generate-chad") else {
            completion(.failure(NSError(domain: "ChadGeneration", code: -1, userInfo: [NSLocalizedDescriptionKey: "Invalid URL"])))
            return
        }
        
        var request = URLRequest(url: url)
        request.httpMethod = "POST"
        print("👑 Sending chad generation request to: \(url.absoluteString)")
        print("👤 Gender: \(gender)")
        
        let boundary = UUID().uuidString
        request.setValue("multipart/form-data; boundary=\(boundary)", forHTTPHeaderField: "Content-Type")
        
        var body = Data()
        
        // Add front image (Image 1)
        body.append("--\(boundary)\r\n".data(using: .utf8)!)
        body.append("Content-Disposition: form-data; name=\"front_image\"; filename=\"front.jpg\"\r\n".data(using: .utf8)!)
        body.append("Content-Type: image/jpeg\r\n\r\n".data(using: .utf8)!)
        body.append(frontImageData)
        body.append("\r\n".data(using: .utf8)!)
        
        // Add side image (Image 2)
        body.append("--\(boundary)\r\n".data(using: .utf8)!)
        body.append("Content-Disposition: form-data; name=\"side_image\"; filename=\"side.jpg\"\r\n".data(using: .utf8)!)
        body.append("Content-Type: image/jpeg\r\n\r\n".data(using: .utf8)!)
        body.append(sideImageData)
        body.append("\r\n".data(using: .utf8)!)
        
        // Add gender
        body.append("--\(boundary)\r\n".data(using: .utf8)!)
        body.append("Content-Disposition: form-data; name=\"gender\"\r\n\r\n".data(using: .utf8)!)
        body.append(gender.data(using: .utf8)!)
        body.append("\r\n".data(using: .utf8)!)
        
        body.append("--\(boundary)--\r\n".data(using: .utf8)!)
        
        request.httpBody = body
        
        urlSession.dataTask(with: request) { data, response, error in
            if let error = error {
                print("❌ Chad generation error: \(error.localizedDescription)")
                completion(.failure(error))
                return
            }
            
            guard let httpResponse = response as? HTTPURLResponse else {
                completion(.failure(NSError(domain: "ChadGeneration", code: -1, userInfo: [NSLocalizedDescriptionKey: "Invalid response"])))
                return
            }
            
            print("📥 Chad generation response status: \(httpResponse.statusCode)")
            
            if httpResponse.statusCode != 200 {
                let errorMessage = String(data: data ?? Data(), encoding: .utf8) ?? "Unknown error"
                print("❌ Server returned status \(httpResponse.statusCode): \(errorMessage)")
                completion(.failure(NSError(domain: "ChadGeneration", code: httpResponse.statusCode, userInfo: [NSLocalizedDescriptionKey: "Server returned status \(httpResponse.statusCode)"])))
                return
            }
            
            guard let data = data else {
                completion(.failure(NSError(domain: "ChadGeneration", code: -1, userInfo: [NSLocalizedDescriptionKey: "No data received"])))
                return
            }
            
            // Parse JSON response
            guard let json = try? JSONSerialization.jsonObject(with: data) as? [String: Any] else {
                print("❌ Failed to parse JSON response")
                completion(.failure(NSError(domain: "ChadGeneration", code: -1, userInfo: [NSLocalizedDescriptionKey: "Invalid response format"])))
                return
            }
            
            // Check for error
            if let errorMessage = json["error"] as? String {
                print("❌ Server error: \(errorMessage)")
                completion(.failure(NSError(domain: "ChadGeneration", code: -1, userInfo: [NSLocalizedDescriptionKey: errorMessage])))
                return
            }
            
            // Get base64 image from response
            guard let imageBase64 = json["image"] as? String,
                  let imageData = Data(base64Encoded: imageBase64),
                  let image = UIImage(data: imageData) else {
                print("❌ Failed to decode image from base64 response")
                completion(.failure(NSError(domain: "ChadGeneration", code: -1, userInfo: [NSLocalizedDescriptionKey: "Failed to decode image"])))
                return
            }
            
            print("✅ Chad image generated successfully")
            completion(.success(image))
        }.resume()
    }
}

