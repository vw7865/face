//
//  ScaleManager.swift
//  LooksmaxAI
//
//  Created by Admin on 2026. 01. 11..
//

import Foundation
import SwiftUI

enum DisplayScale: String, CaseIterable {
    case psl = "PSL Scale (0-8)"
    case objective = "1-10 Objective Scale"
    
    var displayName: String {
        return self.rawValue
    }
}

class ScaleManager: ObservableObject {
    static let shared = ScaleManager()
    
    @AppStorage("selectedDisplayScale") private var selectedScaleRaw: String = DisplayScale.psl.rawValue
    
    var selectedScale: DisplayScale {
        get {
            DisplayScale(rawValue: selectedScaleRaw) ?? .psl
        }
        set {
            selectedScaleRaw = newValue.rawValue
        }
    }
    
    // Convert PSL (0-100) to PSL Scale (0-8)
    // Based on Aesthetica Index Wiki: PSL Scale ranges from 0.5 to 7.9 in practice
    // Backend returns 0-100, we convert to 0-8 scale
    func convertToPSLScale(_ psl: Double) -> Double {
        // PSL Scale: 0-8 range (theoretical max 8.0)
        // Map 0-100 to 0-8: psl * 0.08
        return (psl / 100.0) * 8.0
    }
    
    // Convert PSL (0-100) to 1-10 Objective Scale
    // Based on Aesthetica Index Wiki: 1-10 Objective Scale ranges from 0.5 to 9.9 in practice
    // Backend returns 0-100, we convert to 0-10 scale
    func convertToObjectiveScale(_ psl: Double) -> Double {
        // 1-10 Objective Scale: 0-10 range (theoretical max 10.0)
        // Map 0-100 to 0-10: psl * 0.1
        return (psl / 100.0) * 10.0
    }
    
    // Get descriptive label based on PSL Scale (0-8)
    func getPSLScaleLabel(_ pslScale: Double) -> String {
        switch pslScale {
        case ..<0.5:
            return "Low Subhuman"
        case 0.5..<0.8:
            return "Subhuman"
        case 0.8..<1.4:
            return "High Subhuman"
        case 1.4..<1.9:
            return "LTN-/LTB-"
        case 1.9..<2.4:
            return "LTN/LTB"
        case 2.4..<2.8:
            return "LTN+/LTB+"
        case 2.8..<3.3:
            return "MTN-/MTB-"
        case 3.3..<4.6:
            return "MTN/MTB"
        case 4.6..<5.1:
            return "MTN+/MTB+"
        case 5.1..<5.4:
            return "HTN-/HTB-"
        case 5.4..<5.7:
            return "HTN/HTB"
        case 5.7..<6.0:
            return "HTN+/HTB+"
        case 6.0..<6.4:
            return "Low CL/SL"
        case 6.4..<6.6:
            return "CL/SL"
        case 6.6..<6.9:
            return "High CL/SL"
        case 6.9..<7.1:
            return "Low Chad/Stacy"
        case 7.1..<7.3:
            return "Chad/Stacy"
        case 7.3..<7.5:
            return "High Chad/Stacy"
        case 7.5..<7.8:
            return "Adamlite/Evelite"
        case 7.8...8.0:
            return "True Adam/Eve"
        default:
            return "Unknown"
        }
    }
    
    // Get descriptive label based on 1-10 Objective Scale
    func getObjectiveScaleLabel(_ objectiveScale: Double) -> String {
        switch objectiveScale {
        case ..<0.5:
            return "Low Subhuman"
        case 0.5..<1.1:
            return "Subhuman"
        case 1.1..<1.5:
            return "High Subhuman"
        case 1.5..<2.0:
            return "LTN-/LTB-"
        case 2.0..<2.6:
            return "LTN/LTB"
        case 2.6..<3.5:
            return "LTN+/LTB+"
        case 3.5..<4.0:
            return "MTN-/MTB-"
        case 4.0..<4.6:
            return "MTN/MTB"
        case 4.6..<5.2:
            return "MTN+/MTB+"
        case 5.2..<5.8:
            return "HTN-/HTB-"
        case 5.8..<6.1:
            return "HTN/HTB"
        case 6.1..<6.6:
            return "HTN+/HTB+"
        case 6.6..<7.0:
            return "Low CL/SL"
        case 7.0..<7.5:
            return "CL/SL"
        case 7.5..<7.8:
            return "High CL/SL"
        case 7.8..<8.1:
            return "Low Chad/Stacy"
        case 8.1..<8.4:
            return "Chad/Stacy"
        case 8.4..<8.6:
            return "High Chad/True Stacy"
        case 8.6..<9.5:
            return "Terrachad/Terrastacy"
        case 9.5..<9.7:
            return "Adamlite/Evelite"
        case 9.7...:
            return "True Adam/Eve"
        default:
            return "Unknown"
        }
    }
    
    // Format score based on selected scale
    func formatScore(_ psl: Double) -> String {
        switch selectedScale {
        case .psl:
            let pslScale = convertToPSLScale(psl)
            return String(format: "%.2f", pslScale)
        case .objective:
            let objectiveScale = convertToObjectiveScale(psl)
            return String(format: "%.1f", objectiveScale)
        }
    }
    
    // Helper function to convert gender-neutral labels to gender-specific labels
    private func makeGenderSpecific(_ label: String, gender: String?) -> String {
        guard let gender = gender else { return label }
        let isMale = gender.lowercased() == "male"
        
        // Replace gender-neutral labels with gender-specific ones
        var result = label
        
        // HTN/HTB replacements
        result = result.replacingOccurrences(of: "HTN-/HTB-", with: isMale ? "HTN-" : "HTB-")
        result = result.replacingOccurrences(of: "HTN/HTB", with: isMale ? "HTN" : "HTB")
        result = result.replacingOccurrences(of: "HTN+/HTB+", with: isMale ? "HTN+" : "HTB+")
        
        // Chad/Stacy replacements
        result = result.replacingOccurrences(of: "Low Chad/Stacy", with: isMale ? "Low Chad" : "Low Stacy")
        result = result.replacingOccurrences(of: "Chad/Stacy", with: isMale ? "Chad" : "Stacy")
        result = result.replacingOccurrences(of: "High Chad/Stacy", with: isMale ? "High Chad" : "High Stacy")
        result = result.replacingOccurrences(of: "High Chad/True Stacy", with: isMale ? "High Chad" : "True Stacy")
        
        // Chadlite/Stacylite replacements
        result = result.replacingOccurrences(of: "Low CL/SL", with: isMale ? "Low Chadlite" : "Low Stacylite")
        result = result.replacingOccurrences(of: "CL/SL", with: isMale ? "Chadlite" : "Stacylite")
        result = result.replacingOccurrences(of: "High CL/SL", with: isMale ? "High Chadlite" : "High Stacylite")
        result = result.replacingOccurrences(of: "Low Chadlite/Stacylite", with: isMale ? "Low Chadlite" : "Low Stacylite")
        result = result.replacingOccurrences(of: "Chadlite/Stacylite", with: isMale ? "Chadlite" : "Stacylite")
        result = result.replacingOccurrences(of: "High Chadlite/Stacylite", with: isMale ? "High Chadlite" : "High Stacylite")
        
        // Terrachad/Terrastacy replacements
        result = result.replacingOccurrences(of: "Terrachad/Terrastacy", with: isMale ? "Terrachad" : "Terrastacy")
        
        // Adam/Eve replacements
        result = result.replacingOccurrences(of: "Adamlite/Evelite", with: isMale ? "Adamlite" : "Evelite")
        result = result.replacingOccurrences(of: "True Adam/Eve", with: isMale ? "True Adam" : "True Eve")
        
        // LTN/LTB replacements
        result = result.replacingOccurrences(of: "LTN-/LTB-", with: isMale ? "LTN-" : "LTB-")
        result = result.replacingOccurrences(of: "LTN/LTB", with: isMale ? "LTN" : "LTB")
        result = result.replacingOccurrences(of: "LTN+/LTB+", with: isMale ? "LTN+" : "LTB+")
        
        // MTN/MTB replacements
        result = result.replacingOccurrences(of: "MTN-/MTB-", with: isMale ? "MTN-" : "MTB-")
        result = result.replacingOccurrences(of: "MTN/MTB", with: isMale ? "MTN" : "MTB")
        result = result.replacingOccurrences(of: "MTN+/MTB+", with: isMale ? "MTN+" : "MTB+")
        
        return result
    }
    
    // Get full display string (score + label)
    // IMPORTANT: PSL Scale and 1-10 Objective Scale are DIFFERENT scales with DIFFERENT category boundaries
    // A person can be "HTN/HTB" on PSL Scale but "Low CL/SL" on 1-10 Objective Scale
    // Each scale has its own independent category system
    func getFullDisplay(_ psl: Double, gender: String? = nil) -> (score: String, label: String) {
        switch selectedScale {
        case .psl:
            // PSL Scale: Convert 0-100 to 0-8, then use PSL Scale categories
            let pslScale = convertToPSLScale(psl)
            let score = String(format: "%.2f", pslScale)
            let label = getPSLScaleLabel(pslScale)  // Uses PSL Scale categories (0-8 range)
            return (score, makeGenderSpecific(label, gender: gender))
        case .objective:
            // 1-10 Objective Scale: Convert 0-100 to 0-10, then use 1-10 Objective categories
            let objectiveScale = convertToObjectiveScale(psl)
            let score = String(format: "%.1f", objectiveScale)
            let label = getObjectiveScaleLabel(objectiveScale)  // Uses 1-10 Objective categories (0-10 range)
            return (score, makeGenderSpecific(label, gender: gender))
        }
    }
    
    // Get scale range for progress bar
    func getScaleRange() -> (min: Double, max: Double) {
        switch selectedScale {
        case .psl:
            return (0.0, 8.0)
        case .objective:
            return (0.0, 10.0)
        }
    }
    
    // Convert score for progress bar calculation
    func getProgressValue(_ psl: Double) -> Double {
        switch selectedScale {
        case .psl:
            return convertToPSLScale(psl)
        case .objective:
            return convertToObjectiveScale(psl)
        }
    }
}

