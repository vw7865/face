//
//  RizzAdviceService.swift
//  LooksmaxAI
//
//  Created by Admin on 2026. 01. 11..
//

import Foundation

class RizzAdviceService {
    static let shared = RizzAdviceService()
    
    // Deployed backend URL on Railway
    private let baseURL = "https://looksmax-backend-production.up.railway.app"
    
    // Custom URLSession with timeout for AI processing
    private lazy var urlSession: URLSession = {
        let configuration = URLSessionConfiguration.default
        configuration.timeoutIntervalForRequest = 60.0 // 60 seconds for AI processing
        configuration.timeoutIntervalForResource = 60.0
        return URLSession(configuration: configuration)
    }()
    
    private init() {}
    
    // MARK: - Get Rizz Advice
    func getAdvice(input: String, completion: @escaping (Result<String, Error>) -> Void) {
        guard let url = URL(string: "\(baseURL)/api/rizz-advice") else {
            completion(.failure(NSError(domain: "RizzAdvice", code: -1, userInfo: [NSLocalizedDescriptionKey: "Invalid URL"])))
            return
        }
        
        var request = URLRequest(url: url)
        request.httpMethod = "POST"
        request.setValue("application/json", forHTTPHeaderField: "Content-Type")
        
        let requestBody: [String: Any] = [
            "input": input
        ]
        
        do {
            request.httpBody = try JSONSerialization.data(withJSONObject: requestBody)
        } catch {
            completion(.failure(error))
            return
        }
        
        print("🚀 Sending rizz advice request to: \(url.absoluteString)")
        print("📝 Input length: \(input.count) characters")
        
        urlSession.dataTask(with: request) { data, response, error in
            if let error = error {
                print("❌ Rizz advice request error: \(error.localizedDescription)")
                completion(.failure(error))
                return
            }
            
            guard let httpResponse = response as? HTTPURLResponse else {
                completion(.failure(NSError(domain: "RizzAdvice", code: -1, userInfo: [NSLocalizedDescriptionKey: "Invalid response"])))
                return
            }
            
            print("📥 Rizz advice response status: \(httpResponse.statusCode)")
            
            guard httpResponse.statusCode == 200 else {
                let errorMessage = "Server returned status \(httpResponse.statusCode)"
                print("❌ \(errorMessage)")
                completion(.failure(NSError(domain: "RizzAdvice", code: httpResponse.statusCode, userInfo: [NSLocalizedDescriptionKey: errorMessage])))
                return
            }
            
            guard let data = data else {
                completion(.failure(NSError(domain: "RizzAdvice", code: -1, userInfo: [NSLocalizedDescriptionKey: "No data received"])))
                return
            }
            
            do {
                if let json = try JSONSerialization.jsonObject(with: data) as? [String: Any],
                   let advice = json["advice"] as? String {
                    print("✅ Rizz advice received (length: \(advice.count) characters)")
                    completion(.success(advice))
                } else {
                    completion(.failure(NSError(domain: "RizzAdvice", code: -1, userInfo: [NSLocalizedDescriptionKey: "Invalid response format"])))
                }
            } catch {
                print("❌ JSON parsing error: \(error.localizedDescription)")
                completion(.failure(error))
            }
        }.resume()
    }
}

