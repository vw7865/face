//
//  ImageValidationService.swift
//  LooksmaxAI
//
//  Created by Admin on 2026. 01. 07..
//

import Foundation
import UIKit
import Vision

class ImageValidationService {
    static let shared = ImageValidationService()
    
    private init() {}
    
    enum ValidationError: LocalizedError {
        case noFaceDetected
        case multipleFacesDetected
        case imageTooSmall
        case imageTooDark
        case imageTooBright
        case faceTooSmall
        case faceNotCentered
        case notFrontFacing
        case notSideProfile
        case faceAngleIncorrect
        
        var errorDescription: String? {
            switch self {
            case .noFaceDetected:
                return "No face detected. Please ensure your face is clearly visible and well-lit."
            case .multipleFacesDetected:
                return "Multiple faces detected. Please use a photo with only one face."
            case .imageTooSmall:
                return "Image is too small. Please use a higher quality photo (at least 300x300 pixels)."
            case .imageTooDark:
                return "Image is too dark. Please use better lighting - natural light works best."
            case .imageTooBright:
                return "Image is too bright. Please reduce glare or use softer lighting."
            case .faceTooSmall:
                return "Face is too small in the image. Please move closer so your face fills at least 30% of the frame."
            case .faceNotCentered:
                return "Face should be centered in the frame. Please adjust your position."
            case .notFrontFacing:
                return "This doesn't look like a front-facing photo. Please face the camera directly, like a passport photo."
            case .notSideProfile:
                return "This doesn't look like a side profile. Please turn your head 90 degrees to show your profile clearly."
            case .faceAngleIncorrect:
                return "Face angle is incorrect. For front photo: face camera directly. For side photo: turn head 90 degrees."
            }
        }
    }
    
    struct ValidationResult {
        let isValid: Bool
        let error: ValidationError?
        let faceBounds: CGRect?
        let faceCount: Int
    }
    
    // MARK: - Validate Front Image
    func validateFrontImage(_ image: UIImage, gender: String = "Male", completion: @escaping (ValidationResult) -> Void) {
        validateImage(image, isSideProfile: false, gender: gender, completion: completion)
    }
    
    // MARK: - Validate Side Profile Image
    func validateSideImage(_ image: UIImage, gender: String = "Male", completion: @escaping (ValidationResult) -> Void) {
        validateImage(image, isSideProfile: true, gender: gender, completion: completion)
    }
    
    // MARK: - Main Validation Function
    private func validateImage(_ image: UIImage, isSideProfile: Bool, gender: String = "Male", completion: @escaping (ValidationResult) -> Void) {
        // Check image size
        let minDimension = min(image.size.width, image.size.height)
        if minDimension < 300 {
            completion(ValidationResult(isValid: false, error: .imageTooSmall, faceBounds: nil, faceCount: 0))
            return
        }
        
        // Don't check brightness on entire image - it can be misleading with white backgrounds
        // We'll check face brightness after face detection
        
        // Detect faces using Vision framework
        guard let cgImage = image.cgImage else {
            completion(ValidationResult(isValid: false, error: .noFaceDetected, faceBounds: nil, faceCount: 0))
            return
        }
        
        // Use VNDetectFaceLandmarksRequest for better face analysis
        let request = VNDetectFaceLandmarksRequest { request, error in
            if let error = error {
                print("Face detection error: \(error.localizedDescription)")
                completion(ValidationResult(isValid: false, error: .noFaceDetected, faceBounds: nil, faceCount: 0))
                return
            }
            
            guard let observations = request.results as? [VNFaceObservation] else {
                completion(ValidationResult(isValid: false, error: .noFaceDetected, faceBounds: nil, faceCount: 0))
                return
            }
            
            let faceCount = observations.count
            
            // Check number of faces
            if faceCount == 0 {
                completion(ValidationResult(isValid: false, error: .noFaceDetected, faceBounds: nil, faceCount: 0))
                return
            }
            
            if faceCount > 1 {
                completion(ValidationResult(isValid: false, error: .multipleFacesDetected, faceBounds: nil, faceCount: faceCount))
                return
            }
            
            // Get the first (and only) face
            guard let face = observations.first else {
                completion(ValidationResult(isValid: false, error: .noFaceDetected, faceBounds: nil, faceCount: 0))
                return
            }
            
            // Convert normalized coordinates to image coordinates
            let imageSize = CGSize(width: cgImage.width, height: cgImage.height)
            let faceBounds = VNImageRectForNormalizedRect(
                face.boundingBox,
                Int(imageSize.width),
                Int(imageSize.height)
            )
            
            // Check face size (should be at least 30% of image)
            let faceArea = faceBounds.width * faceBounds.height
            let imageArea = imageSize.width * imageSize.height
            let faceRatio = faceArea / imageArea
            
            if faceRatio < 0.15 {
                completion(ValidationResult(isValid: false, error: .faceTooSmall, faceBounds: faceBounds, faceCount: 1))
                return
            }
            
            // Check if face is reasonably centered (within 40% of center)
            let faceCenter = CGPoint(
                x: faceBounds.midX,
                y: faceBounds.midY
            )
            let imageCenter = CGPoint(
                x: imageSize.width / 2,
                y: imageSize.height / 2
            )
            
            let centerDistance = sqrt(
                pow(faceCenter.x - imageCenter.x, 2) +
                pow(faceCenter.y - imageCenter.y, 2)
            )
            let maxDistance = min(imageSize.width, imageSize.height) * 0.4
            
            if centerDistance > maxDistance {
                completion(ValidationResult(isValid: false, error: .faceNotCentered, faceBounds: faceBounds, faceCount: 1))
                return
            }
            
            // Check brightness of face region specifically (not entire image)
            // Only check for extremely dark images - brightness check is too unreliable
            // We'll only reject if the face is so dark that features can't be detected
            if let faceBrightness = self.calculateFaceBrightness(image: image, faceBounds: faceBounds) {
                // Only reject if face is extremely dark (very hard to see features)
                // Most faces are 0.2-0.8, so 0.08 is extremely dark
                if faceBrightness < 0.08 {
                    completion(ValidationResult(isValid: false, error: .imageTooDark, faceBounds: faceBounds, faceCount: 1))
                    return
                }
                // Removed "too bright" check - it's causing false positives
                // Well-lit photos with white backgrounds can have high brightness but still be valid
            }
            
            // Check face angle/orientation using face bounding box aspect ratio
            // For front-facing: face should be roughly square (width ≈ height)
            // For side profile: face should be narrower (width < height)
            let faceAspectRatio = faceBounds.width / faceBounds.height
            
            if isSideProfile {
                // For side profile: face should be narrower (aspect ratio < 1.0)
                // Be very lenient - aspect ratio check alone isn't reliable
                // Allow up to 0.95 for side profiles (some angles can appear wider)
                // For women, be even more lenient (0.98) since hair can affect the bounding box
                let maxRatio = gender == "Female" ? 0.98 : 0.95
                
                // Only reject if it's clearly front-facing (ratio > 1.0)
                // This is a very lenient check - we mainly want to catch obvious front-facing photos
                if faceAspectRatio > maxRatio && faceAspectRatio > 1.0 {
                    completion(ValidationResult(isValid: false, error: .notSideProfile, faceBounds: faceBounds, faceCount: 1))
                    return
                }
            } else {
                // For front photo: face should be roughly square (0.7 < ratio < 1.3)
                // Be lenient to allow slight angles
                if faceAspectRatio < 0.7 || faceAspectRatio > 1.3 {
                    completion(ValidationResult(isValid: false, error: .notFrontFacing, faceBounds: faceBounds, faceCount: 1))
                    return
                }
            }
            
            // All checks passed
            completion(ValidationResult(isValid: true, error: nil, faceBounds: faceBounds, faceCount: 1))
        }
        
        let handler = VNImageRequestHandler(cgImage: cgImage, options: [:])
        do {
            try handler.perform([request])
        } catch {
            print("Failed to perform face detection: \(error.localizedDescription)")
            completion(ValidationResult(isValid: false, error: .noFaceDetected, faceBounds: nil, faceCount: 0))
        }
    }
    
    // MARK: - Calculate Face Region Brightness
    private func calculateFaceBrightness(image: UIImage, faceBounds: CGRect) -> CGFloat? {
        guard let cgImage = image.cgImage else { return nil }
        
        // Crop to face region with minimal padding to avoid including background
        // Use smaller padding to focus more on the actual face
        let padding: CGFloat = 10
        let faceRect = CGRect(
            x: max(0, faceBounds.origin.x - padding),
            y: max(0, faceBounds.origin.y - padding),
            width: min(CGFloat(cgImage.width) - (faceBounds.origin.x - padding), faceBounds.width + padding * 2),
            height: min(CGFloat(cgImage.height) - (faceBounds.origin.y - padding), faceBounds.height + padding * 2)
        )
        
        // Ensure the rect is valid
        guard faceRect.width > 0 && faceRect.height > 0 else { return nil }
        
        guard let croppedImage = cgImage.cropping(to: faceRect) else { return nil }
        
        let context = CIContext()
        let ciImage = CIImage(cgImage: croppedImage)
        
        // Create a small thumbnail for faster processing
        let thumbnailSize = CGSize(width: min(100, Int(faceRect.width)), height: min(100, Int(faceRect.height)))
        let thumbnailRect = CGRect(origin: .zero, size: thumbnailSize)
        
        guard let thumbnail = context.createCGImage(ciImage, from: thumbnailRect) else { return nil }
        
        // Calculate average brightness
        let width = thumbnail.width
        let height = thumbnail.height
        let bytesPerPixel = 4
        let bytesPerRow = bytesPerPixel * width
        let bitsPerComponent = 8
        
        var pixelData = [UInt8](repeating: 0, count: width * height * bytesPerPixel)
        let colorSpace = CGColorSpaceCreateDeviceRGB()
        let context2 = CGContext(
            data: &pixelData,
            width: width,
            height: height,
            bitsPerComponent: bitsPerComponent,
            bytesPerRow: bytesPerRow,
            space: colorSpace,
            bitmapInfo: CGImageAlphaInfo.noneSkipLast.rawValue
        )
        
        context2?.draw(thumbnail, in: CGRect(x: 0, y: 0, width: width, height: height))
        
        var totalBrightness: CGFloat = 0
        let pixelCount = width * height
        
        for i in 0..<pixelCount {
            let index = i * bytesPerPixel
            let r = CGFloat(pixelData[index])
            let g = CGFloat(pixelData[index + 1])
            let b = CGFloat(pixelData[index + 2])
            
            // Calculate brightness using luminance formula
            let brightness = (0.299 * r + 0.587 * g + 0.114 * b) / 255.0
            totalBrightness += brightness
        }
        
        return totalBrightness / CGFloat(pixelCount)
    }
    
}

