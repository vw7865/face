//
//  TextRecognitionService.swift
//  LooksmaxAI
//
//  Created by Admin on 2026. 01. 15..
//

import Foundation
import Vision
import UIKit

class TextRecognitionService {
    static let shared = TextRecognitionService()
    
    private init() {}
    
    /// Extracts text from an image using Vision framework's text recognition
    /// - Parameters:
    ///   - image: The UIImage to extract text from
    ///   - completion: Completion handler with the extracted text or error
    func recognizeText(in image: UIImage, completion: @escaping (Result<String, Error>) -> Void) {
        guard let cgImage = image.cgImage else {
            completion(.failure(NSError(domain: "TextRecognition", code: -1, userInfo: [NSLocalizedDescriptionKey: "Failed to get CGImage from UIImage"])))
            return
        }
        
        // Create a request handler
        let requestHandler = VNImageRequestHandler(cgImage: cgImage, options: [:])
        
        // Create a text recognition request
        let request = VNRecognizeTextRequest { request, error in
            if let error = error {
                print("❌ Text recognition error: \(error.localizedDescription)")
                completion(.failure(error))
                return
            }
            
            guard let observations = request.results as? [VNRecognizedTextObservation] else {
                completion(.success("")) // No text found, return empty string
                return
            }
            
            // Extract text from observations
            var recognizedText = ""
            for observation in observations {
                guard let topCandidate = observation.topCandidates(1).first else {
                    continue
                }
                recognizedText += topCandidate.string + "\n"
            }
            
            // Clean up the text (remove extra newlines)
            let cleanedText = recognizedText.trimmingCharacters(in: .whitespacesAndNewlines)
            
            if cleanedText.isEmpty {
                print("⚠️ No text recognized in image")
                completion(.success("")) // Return empty string if no text found
            } else {
                print("✅ Text recognized: \(cleanedText.prefix(100))...")
                completion(.success(cleanedText))
            }
        }
        
        // Configure the request for better accuracy
        request.recognitionLevel = .accurate
        request.recognitionLanguages = ["en-US"] // English recognition
        request.usesLanguageCorrection = true
        
        // Perform the request
        do {
            try requestHandler.perform([request])
        } catch {
            print("❌ Failed to perform text recognition: \(error.localizedDescription)")
            completion(.failure(error))
        }
    }
}


