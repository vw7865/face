//
//  DatingPhotoService.swift
//  LooksmaxAI
//
//  Created by Admin on 2026. 01. 14..
//

import Foundation
import UIKit

struct DatingPhotoService {
    private static let baseURL = "https://looksmax-backend-production.up.railway.app"
    
    static func generatePhoto(
        userPhoto: UIImage,
        referenceImage: UIImage?,
        prompt: String,
        swapType: String? = nil,
        matchClothingToScene: Bool = false,
        clothingSource: String? = nil,
        comments: String? = nil,
        completion: @escaping (Result<UIImage, Error>) -> Void
    ) {
        guard let url = URL(string: "\(baseURL)/api/dating-photo") else {
            completion(.failure(NSError(domain: "DatingPhotoService", code: -1, userInfo: [NSLocalizedDescriptionKey: "Invalid URL"])))
            return
        }
        
        var request = URLRequest(url: url)
        request.httpMethod = "POST"
        request.setValue("application/json", forHTTPHeaderField: "Content-Type")
        
        // Convert images to base64
        guard let userPhotoData = userPhoto.jpegData(compressionQuality: 0.8) else {
            completion(.failure(NSError(domain: "DatingPhotoService", code: -1, userInfo: [NSLocalizedDescriptionKey: "Failed to encode user photo"])))
            return
        }
        let userPhotoBase64 = userPhotoData.base64EncodedString()
        
        var requestBody: [String: Any] = [
            "userPhoto": userPhotoBase64,
            "prompt": prompt
        ]
        
        // Add optional parameters
        if let swapType = swapType {
            requestBody["swapType"] = swapType
        }
        requestBody["matchClothingToScene"] = matchClothingToScene
        if let clothingSource = clothingSource {
            requestBody["clothingSource"] = clothingSource
        }
        if let comments = comments, !comments.isEmpty {
            requestBody["comments"] = comments
        }
        
        if let refImage = referenceImage,
           let refImageData = refImage.jpegData(compressionQuality: 0.8) {
            let refImageBase64 = refImageData.base64EncodedString()
            requestBody["referenceImage"] = refImageBase64
        }
        
        do {
            request.httpBody = try JSONSerialization.data(withJSONObject: requestBody)
        } catch {
            completion(.failure(error))
            return
        }
        
        print("🚀 Sending dating photo request to: \(url)")
        print("📝 Prompt being sent: \(prompt)")
        print("📝 Prompt length: \(prompt.count) characters")
        print("📸 User photo size: \(userPhotoData.count) bytes")
        if referenceImage != nil {
            print("📸 Reference image provided")
        }
        
        URLSession.shared.dataTask(with: request) { data, response, error in
            if let error = error {
                print("❌ Network error: \(error.localizedDescription)")
                completion(.failure(error))
                return
            }
            
            guard let httpResponse = response as? HTTPURLResponse else {
                completion(.failure(NSError(domain: "DatingPhotoService", code: -1, userInfo: [NSLocalizedDescriptionKey: "Invalid response"])))
                return
            }
            
            print("📥 Dating photo response status: \(httpResponse.statusCode)")
            
            guard httpResponse.statusCode == 200 else {
                if let data = data,
                   let json = try? JSONSerialization.jsonObject(with: data) as? [String: Any],
                   let errorMessage = json["error"] as? String {
                    completion(.failure(NSError(domain: "DatingPhotoService", code: httpResponse.statusCode, userInfo: [NSLocalizedDescriptionKey: errorMessage])))
                } else {
                    completion(.failure(NSError(domain: "DatingPhotoService", code: httpResponse.statusCode, userInfo: [NSLocalizedDescriptionKey: "Server error: \(httpResponse.statusCode)"])))
                }
                return
            }
            
            guard let data = data else {
                completion(.failure(NSError(domain: "DatingPhotoService", code: -1, userInfo: [NSLocalizedDescriptionKey: "No data received"])))
                return
            }
            
            do {
                let json = try JSONSerialization.jsonObject(with: data) as? [String: Any]
                guard let imageBase64 = json?["image"] as? String,
                      let imageData = Data(base64Encoded: imageBase64),
                      let image = UIImage(data: imageData) else {
                    completion(.failure(NSError(domain: "DatingPhotoService", code: -1, userInfo: [NSLocalizedDescriptionKey: "Failed to decode image"])))
                    return
                }
                
                print("✅ Generated image received (size: \(imageData.count) bytes)")
                completion(.success(image))
            } catch {
                completion(.failure(error))
            }
        }.resume()
    }
}

