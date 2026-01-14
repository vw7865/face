//
//  FaceAnalysisService.swift
//  LooksmaxAI
//
//  Created by Admin on 2026. 01. 07..
//

import Foundation
import UIKit

class FaceAnalysisService {
    static let shared = FaceAnalysisService()
    
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
    
    // MARK: - Analyze Faces
    // Gender is now optional - backend will auto-detect if not provided
    func analyzeFaces(frontImage: UIImage, sideImage: UIImage, gender: String? = nil, completion: @escaping (Result<FaceAnalysisResults, Error>) -> Void) {
        guard let frontImageData = frontImage.jpegData(compressionQuality: 0.8),
              let sideImageData = sideImage.jpegData(compressionQuality: 0.8) else {
            completion(.failure(NSError(domain: "FaceAnalysis", code: -1, userInfo: [NSLocalizedDescriptionKey: "Failed to convert images"])))
            return
        }
        
        // Start the analysis request directly
        // The request has a 120-second timeout for ML model processing
        performAPICall(frontImageData: frontImageData, sideImageData: sideImageData, gender: gender, completion: completion)
    }
    
    // MARK: - Actual API Call with Retry Logic
    private func performAPICall(frontImageData: Data, sideImageData: Data, gender: String?, completion: @escaping (Result<FaceAnalysisResults, Error>) -> Void, retryCount: Int = 0) {
        print("🚀 Starting face analysis request (attempt \(retryCount + 1))...")
        
        guard let url = URL(string: "\(baseURL)/api/analyze-face") else {
            completion(.failure(NSError(domain: "FaceAnalysis", code: -1, userInfo: [NSLocalizedDescriptionKey: "Invalid URL"])))
            return
        }
        
        var request = URLRequest(url: url)
        request.httpMethod = "POST"
        print("📡 Sending request to: \(url.absoluteString)")
        
        let boundary = UUID().uuidString
        request.setValue("multipart/form-data; boundary=\(boundary)", forHTTPHeaderField: "Content-Type")
        
        var body = Data()
        
        // Add front image
        body.append("--\(boundary)\r\n".data(using: .utf8)!)
        body.append("Content-Disposition: form-data; name=\"front_image\"; filename=\"front.jpg\"\r\n".data(using: .utf8)!)
        body.append("Content-Type: image/jpeg\r\n\r\n".data(using: .utf8)!)
        body.append(frontImageData)
        body.append("\r\n".data(using: .utf8)!)
        
        // Add side image
        body.append("--\(boundary)\r\n".data(using: .utf8)!)
        body.append("Content-Disposition: form-data; name=\"side_image\"; filename=\"side.jpg\"\r\n".data(using: .utf8)!)
        body.append("Content-Type: image/jpeg\r\n\r\n".data(using: .utf8)!)
        body.append(sideImageData)
        body.append("\r\n".data(using: .utf8)!)
        
        // Add gender (optional - backend will auto-detect if not provided)
        if let gender = gender, !gender.isEmpty {
            body.append("--\(boundary)\r\n".data(using: .utf8)!)
            body.append("Content-Disposition: form-data; name=\"gender\"\r\n\r\n".data(using: .utf8)!)
            body.append(gender.data(using: .utf8)!)
            body.append("\r\n".data(using: .utf8)!)
        }
        
        body.append("--\(boundary)--\r\n".data(using: .utf8)!)
        
        request.httpBody = body
        
        urlSession.dataTask(with: request) { data, response, error in
            if let error = error {
                // Check if it's a timeout error and retry once
                let nsError = error as NSError
                print("❌ Request error: \(error.localizedDescription) (code: \(nsError.code))")
                
                if (nsError.domain == NSURLErrorDomain && nsError.code == NSURLErrorTimedOut) && retryCount < 1 {
                    print("⏳ Request timed out, retrying in 5 seconds... (attempt \(retryCount + 1))")
                    // Wait before retrying (ML processing can take time)
                    DispatchQueue.main.asyncAfter(deadline: .now() + 5.0) {
                        print("🔄 Retrying request...")
                        self.performAPICall(frontImageData: frontImageData, sideImageData: sideImageData, gender: gender, completion: completion, retryCount: retryCount + 1)
                    }
                    return
                }
                
                // For timeout errors, provide a more helpful message
                var errorMessage = error.localizedDescription
                if nsError.domain == NSURLErrorDomain && nsError.code == NSURLErrorTimedOut {
                    if retryCount >= 1 {
                        errorMessage = "The server is taking longer than usual to process your images. This may be due to ML model processing. Please try again in a moment."
                    } else {
                        errorMessage = "Request timed out. Retrying..."
                    }
                }
                
                // Only show error if we've already retried
                if retryCount >= 1 {
                    DispatchQueue.main.async {
                        completion(.failure(NSError(domain: "FaceAnalysis", code: nsError.code, userInfo: [NSLocalizedDescriptionKey: errorMessage])))
                    }
                }
                return
            }
            
            guard let data = data else {
                DispatchQueue.main.async {
                    completion(.failure(NSError(domain: "FaceAnalysis", code: -1, userInfo: [NSLocalizedDescriptionKey: "No data received"])))
                }
                return
            }
            
            if let httpResponse = response as? HTTPURLResponse {
                print("📥 Response received: Status \(httpResponse.statusCode)")
                if httpResponse.statusCode != 200 {
                    let errorMessage = String(data: data, encoding: .utf8) ?? "Server error"
                    print("❌ Server error: \(errorMessage)")
                    DispatchQueue.main.async {
                        completion(.failure(NSError(domain: "FaceAnalysis", code: httpResponse.statusCode, userInfo: [NSLocalizedDescriptionKey: errorMessage])))
                    }
                    return
                }
            }
            
            // Log the raw response for debugging
            if let responseString = String(data: data, encoding: .utf8) {
                print("✅ API Response received (length: \(data.count) bytes)")
                print("API Response: \(responseString.prefix(500))...") // First 500 chars
            }
            
            do {
                let decoder = JSONDecoder()
                // Custom date decoding for ISO8601 with 'Z' suffix
                let formatter = ISO8601DateFormatter()
                formatter.formatOptions = [.withInternetDateTime, .withFractionalSeconds]
                decoder.dateDecodingStrategy = .custom { decoder in
                    let container = try decoder.singleValueContainer()
                    let dateString = try container.decode(String.self)
                    
                    // Try with fractional seconds first
                    formatter.formatOptions = [.withInternetDateTime, .withFractionalSeconds]
                    if let date = formatter.date(from: dateString) {
                        return date
                    }
                    
                    // Try without fractional seconds
                    formatter.formatOptions = [.withInternetDateTime]
                    if let date = formatter.date(from: dateString) {
                        return date
                    }
                    
                    // Try with just date (no time)
                    let dateOnlyFormatter = DateFormatter()
                    dateOnlyFormatter.dateFormat = "yyyy-MM-dd"
                    if let date = dateOnlyFormatter.date(from: dateString) {
                        return date
                    }
                    
                    // If all fail, return a default date (since Date is not optional in the closure)
                    // The actual property is optional, so this will be handled by the decoder
                    return Date()
                }
                let results = try decoder.decode(FaceAnalysisResults.self, from: data)
                print("✅ Successfully decoded results! PSL: \(results.overall.psl ?? 0)")
                print("📱 About to call completion handler on main thread...")
                DispatchQueue.main.async {
                    print("📱 Calling completion handler with success result")
                    completion(.success(results))
                    print("📱 Completion handler called successfully")
                }
            } catch let decodingError as DecodingError {
                // Better error reporting
                let errorMessage: String
                switch decodingError {
                case .typeMismatch(let type, let context):
                    errorMessage = "Type mismatch for \(type) at \(context.codingPath): \(context.debugDescription)"
                case .valueNotFound(let type, let context):
                    errorMessage = "Value not found for \(type) at \(context.codingPath): \(context.debugDescription)"
                case .keyNotFound(let key, let context):
                    errorMessage = "Key '\(key.stringValue)' not found at \(context.codingPath): \(context.debugDescription)"
                case .dataCorrupted(let context):
                    errorMessage = "Data corrupted at \(context.codingPath): \(context.debugDescription)"
                @unknown default:
                    errorMessage = "Unknown decoding error: \(decodingError.localizedDescription)"
                }
                print("Decoding Error: \(errorMessage)")
                if let responseString = String(data: data, encoding: .utf8) {
                    print("Response was: \(responseString)")
                }
                DispatchQueue.main.async {
                    completion(.failure(NSError(domain: "FaceAnalysis", code: -1, userInfo: [NSLocalizedDescriptionKey: "Failed to parse response: \(errorMessage)"])))
                }
            } catch {
                print("Unexpected error: \(error.localizedDescription)")
                if let responseString = String(data: data, encoding: .utf8) {
                    print("Response was: \(responseString)")
                }
                DispatchQueue.main.async {
                    completion(.failure(error))
                }
            }
        }.resume()
    }
}

