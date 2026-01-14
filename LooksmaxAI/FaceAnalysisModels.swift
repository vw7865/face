//
//  FaceAnalysisModels.swift
//  LooksmaxAI
//
//  Created by Admin on 2026. 01. 07..
//

import Foundation

// MARK: - Main Results Model
struct FaceAnalysisResults: Codable {
    let overall: OverallResults
    let eyes: EyesResults
    let midface: MidfaceResults
    let lowerThird: LowerThirdResults
    let upperThird: UpperThirdResults
    let miscellaneous: MiscellaneousResults
    let ascensionDate: Date?
}

// MARK: - Overall Results
struct OverallResults: Codable {
    let psl: Double?
    let potential: Double?
    
    // Provide computed properties with defaults (use 50.0 instead of 0.0 for better UX)
    var pslValue: Double { psl ?? 50.0 }
    var potentialValue: Double { potential ?? 50.0 }
}

// MARK: - Eyes Results
struct EyesResults: Codable {
    let orbitalDepth: Double?
    let canthalTilt: Double?
    let eyebrowDensity: Double?
    let eyelashDensity: Double?
    let eyelidExposure: Double?
    let underEyeHealth: Double?
    
    // Helper to get value or default (use 50.0 instead of 0.0 for better UX)
    func value(for key: String) -> Double {
        switch key {
        case "orbitalDepth": return orbitalDepth ?? 50.0
        case "canthalTilt": return canthalTilt ?? 50.0
        case "eyebrowDensity": return eyebrowDensity ?? 50.0
        case "eyelashDensity": return eyelashDensity ?? 50.0
        case "eyelidExposure": return eyelidExposure ?? 50.0
        case "underEyeHealth": return underEyeHealth ?? 50.0
        default: return 50.0
        }
    }
}

// MARK: - Midface Results
struct MidfaceResults: Codable {
    let cheekbones: Double?
    let maxilla: Double?
    let nose: Double?
    let ipd: Double?
    let fwhr: Double?
    let compactness: Double?
}

// MARK: - Lower Third Results
struct LowerThirdResults: Codable {
    let lips: Double?
    let mandible: Double?
    let gonialAngle: Double?
    let ramus: Double?
    let hyoidSkinTightness: Double?
    let jawWidth: Double?
}

// MARK: - Upper Third Results
struct UpperThirdResults: Codable {
    let norwoodStage: Double?
    let foreheadProjection: Double?
    let hairlineRecession: Double?
    let hairThinning: Double?
    let hairlineDensity: Double?
    let foreheadSlope: Double?
}

// MARK: - Miscellaneous Results
struct MiscellaneousResults: Codable {
    let skin: Double?
    let harmony: Double?
    let symmetry: Double?
    let neckWidth: Double?
    let bloat: Double?
    let boneMass: Double?
}

// MARK: - Metric Card Model
struct MetricCard: Identifiable {
    let id = UUID()
    let title: String
    let score: Double
    let isLocked: Bool
    let rawPSL: Double? // Optional: raw PSL (0-100) for scale conversion, nil for other metrics
}

// MARK: - Results Section
enum ResultsSection: String, CaseIterable {
    case overall = "Overall"
    case eyes = "Eyes"
    case midface = "Midface"
    case lowerThird = "Lower Third"
    case upperThird = "Upper Third"
    case miscellaneous = "Miscellaneous"
}

