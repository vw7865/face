//
//  RizzMaxxingView.swift
//  LooksmaxAI
//
//  Created by Admin on 2026. 01. 11..
//

import SwiftUI
import UIKit

struct RizzMaxxingView: View {
    @Environment(\.dismiss) var dismiss
    @FocusState private var isTextFieldFocused: Bool
    @State private var selectedScreenshot: UIImage? = nil
    @State private var contextText: String = ""
    @State private var showImagePicker = false
    @State private var showImageSourceOptions = false
    @State private var imagePickerSourceType: UIImagePickerController.SourceType = .photoLibrary
    @State private var isAnalyzing = false
    @State private var showResults = false
    @State private var adviceResults: RizzAdviceResults? = nil
    @State private var isExtractingText = false
    @State private var extractedText: String = ""
    @StateObject private var usageTracker = UsageTracker.shared
    @StateObject private var subscriptionManager = SubscriptionManager.shared
    @State private var isShowingUpgrade = false
    @State private var showUpgradeAlert = false
    
    private func dismissKeyboard() {
        UIApplication.shared.sendAction(#selector(UIResponder.resignFirstResponder), to: nil, from: nil, for: nil)
    }
    
    var body: some View {
        NavigationStack {
            ZStack {
                Color.black.ignoresSafeArea()
                
                if !showResults {
                    inputView
                } else {
                    resultsView
                }
            }
            .navigationTitle("RizzMaxxing")
            .navigationBarTitleDisplayMode(.inline)
            .navigationBarBackButtonHidden(true)
            .toolbarVisibility(.hidden, for: .tabBar)
            .toolbar {
                ToolbarItem(placement: .navigationBarLeading) {
                    Button(action: {
                        dismiss()
                    }) {
                        Image(systemName: "xmark.circle.fill")
                            .font(.system(size: 24))
                            .foregroundColor(.white.opacity(0.8))
                    }
                }
                
                ToolbarItemGroup(placement: .keyboard) {
                    Spacer()
                    Button("Done") {
                        dismissKeyboard()
                    }
                    .foregroundColor(.cyan)
                }
            }
            .onDisappear {
                // Show tab bar when view is dismissed
                NotificationCenter.default.post(name: NSNotification.Name("ShowTabBar"), object: nil)
            }
            .sheet(isPresented: $showImagePicker) {
                ImagePicker(selectedImage: $selectedScreenshot, sourceType: imagePickerSourceType)
            }
            .fullScreenCover(isPresented: $isShowingUpgrade) {
                UpgradeView()
            }
            .onChange(of: selectedScreenshot) { oldValue, newValue in
                // Extract text when a new screenshot is selected
                if let screenshot = newValue {
                    extractTextFromImage(screenshot)
                } else {
                    // Clear extracted text if screenshot is removed
                    extractedText = ""
                }
            }
            .confirmationDialog("Select Screenshot Source", isPresented: $showImageSourceOptions, titleVisibility: .visible) {
                Button("Photo Library") {
                    imagePickerSourceType = .photoLibrary
                    showImagePicker = true
                }
                Button("Camera") {
                    imagePickerSourceType = .camera
                    showImagePicker = true
                }
                Button("Cancel", role: .cancel) { }
            }
        }
    }
    
    private var inputView: some View {
        ScrollView {
            VStack(spacing: 24) {
                headerSection
                screenshotUploadSection
                contextTextFieldSection
                analyzeButton
            }
        }
        .onTapGesture {
            dismissKeyboard()
        }
    }
    
    private var headerSection: some View {
        VStack(spacing: 12) {
            Image(systemName: "message.fill")
                .font(.system(size: 50))
                .foregroundColor(.cyan)
            
            Text("RizzMaxxing")
                .font(.title)
                .fontWeight(.bold)
                .foregroundColor(.white)
            
            Text("Upload a screenshot or type out your conversation to get AI-powered advice")
                .font(.subheadline)
                .foregroundColor(.gray)
                .multilineTextAlignment(.center)
                .padding(.horizontal)
        }
        .padding(.top, 20)
    }
    
    private var screenshotUploadSection: some View {
        VStack(alignment: .leading, spacing: 12) {
            Text("Conversation Screenshot (Optional)")
                .font(.headline)
                .foregroundColor(.white)
            
            Text("Upload a screenshot if you have one. Skip this if you're worried about screenshot notifications (e.g., Snapchat)")
                .font(.caption)
                .foregroundColor(.gray)
            
            screenshotButton
        }
        .padding()
        .background(sectionBackground)
        .padding(.horizontal)
    }
    
    private var screenshotButton: some View {
        Button(action: {
            showImageSourceOptions = true
        }) {
            if let screenshot = selectedScreenshot {
                screenshotPreviewView(screenshot: screenshot)
            } else {
                screenshotPlaceholderView
            }
        }
        .buttonStyle(PlainButtonStyle())
    }
    
    private func screenshotPreviewView(screenshot: UIImage) -> some View {
        VStack(spacing: 12) {
            Image(uiImage: screenshot)
                .resizable()
                .aspectRatio(contentMode: .fit)
                .frame(maxHeight: 300)
                .cornerRadius(12)
                .overlay(
                    RoundedRectangle(cornerRadius: 12)
                        .stroke(Color.white.opacity(0.2), lineWidth: 1)
                )
            
            if isExtractingText {
                HStack {
                    ProgressView()
                        .progressViewStyle(CircularProgressViewStyle(tint: .cyan))
                    Text("Extracting text...")
                        .font(.caption)
                        .foregroundColor(.gray)
                }
            } else if !extractedText.isEmpty {
                HStack {
                    Image(systemName: "text.bubble.fill")
                        .foregroundColor(.green)
                    Text("Text extracted")
                        .font(.caption)
                        .foregroundColor(.green)
                }
            }
            
            Button(action: {
                showImageSourceOptions = true
            }) {
                Text("Change Screenshot")
                    .font(.subheadline)
                    .foregroundColor(.cyan)
            }
        }
    }
    
    private var screenshotPlaceholderView: some View {
        VStack(spacing: 12) {
            Image(systemName: "photo.badge.plus")
                .font(.system(size: 50))
                .foregroundColor(.gray)
            
            Text("Tap to upload screenshot")
                .font(.subheadline)
                .foregroundColor(.gray)
            
            Text("From Snapchat, Messages, or any dating app")
                .font(.caption)
                .foregroundColor(.gray.opacity(0.7))
        }
        .frame(maxWidth: .infinity)
        .frame(height: 200)
        .background(placeholderBackground)
    }
    
    private var placeholderBackground: some View {
        RoundedRectangle(cornerRadius: 12)
            .fill(Color.white.opacity(0.05))
            .overlay(
                RoundedRectangle(cornerRadius: 12)
                    .stroke(Color.white.opacity(0.2), style: StrokeStyle(lineWidth: 2, dash: [10]))
            )
    }
    
    private var sectionBackground: some View {
        RoundedRectangle(cornerRadius: 16)
            .fill(Color.white.opacity(0.1))
            .overlay(
                RoundedRectangle(cornerRadius: 16)
                    .stroke(Color.white.opacity(0.2), lineWidth: 1)
            )
    }
    
    private var contextTextFieldSection: some View {
        VStack(alignment: .leading, spacing: 12) {
            Text("Conversation Details")
                .font(.headline)
                .foregroundColor(.white)
            
            Text("Type out your conversation, add context, or explain the situation. Text from screenshots will be automatically extracted and shown here (you can edit it).")
                .font(.caption)
                .foregroundColor(.gray)
            
            ZStack(alignment: .topLeading) {
                TextEditor(text: $contextText)
                    .frame(height: 150)
                    .padding(8)
                    .background(Color.white.opacity(0.05))
                    .cornerRadius(8)
                    .overlay(
                        RoundedRectangle(cornerRadius: 8)
                            .stroke(Color.white.opacity(0.2), lineWidth: 1)
                    )
                    .foregroundColor(.white)
                    .scrollContentBackground(.hidden)
                    .focused($isTextFieldFocused)
                
                if contextText.isEmpty {
                    Text("Paste your conversation here or describe the situation...\n\nE.g., \"We just met last week, trying to get a first date. Here's our conversation:\nHer: Hey! How's it going?\nMe: Good! Just got back from the gym. What about you?\"")
                        .font(.caption)
                        .foregroundColor(.gray.opacity(0.5))
                        .padding(.leading, 12)
                        .padding(.top, 8)
                        .allowsHitTesting(false)
                }
            }
        }
        .padding()
        .background(sectionBackground)
        .padding(.horizontal)
    }
    
    private var analyzeButton: some View {
        VStack(spacing: 12) {
            // Show usage limit for free users
            if !subscriptionManager.isPro && usageTracker.hasUsedRizzCoach() {
                VStack(spacing: 8) {
                    Text("You've used your free Rizz Coach analysis")
                        .font(.headline)
                        .foregroundColor(.white)
                    Text("Upgrade to Pro for unlimited advice")
                        .font(.subheadline)
                        .foregroundColor(.gray)
                    Button(action: {
                        isShowingUpgrade = true
                    }) {
                        Text("Upgrade to Pro")
                            .font(.headline)
                            .foregroundColor(.white)
                            .padding(.horizontal, 24)
                            .padding(.vertical, 12)
                            .background(Color.red)
                            .cornerRadius(10)
                    }
                }
                .padding()
                .background(
                    RoundedRectangle(cornerRadius: 12)
                        .fill(Color.white.opacity(0.1))
                )
                .padding(.horizontal)
            }
            
            Button(action: {
                if usageTracker.canUseRizzCoach() {
                    analyzeConversation()
                } else {
                    isShowingUpgrade = true
                }
            }) {
                HStack {
                    if isAnalyzing {
                        ProgressView()
                            .progressViewStyle(CircularProgressViewStyle(tint: .white))
                    } else {
                        Image(systemName: "sparkles")
                        Text("Get Advice")
                    }
                }
                .font(.headline)
                .foregroundColor(.white)
                .frame(maxWidth: .infinity)
                .padding()
                .background(buttonBackground)
            }
            .disabled((selectedScreenshot == nil && contextText.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty) || isAnalyzing || (!subscriptionManager.isPro && usageTracker.hasUsedRizzCoach()))
            .padding(.horizontal)
            .padding(.bottom, 30)
        }
    }
    
    private var buttonBackground: some View {
        let hasContent = selectedScreenshot != nil || !contextText.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty
        return RoundedRectangle(cornerRadius: 12)
            .fill(hasContent ? Color.cyan : Color.gray.opacity(0.3))
    }
    
    private var resultsView: some View {
        Group {
            if let results = adviceResults {
                RizzAdviceResultsView(results: results, onBack: {
                    showResults = false
                    adviceResults = nil
                })
            } else {
                RizzAdviceResultsView(results: RizzAdviceResults.placeholder, onBack: {
                    showResults = false
                    adviceResults = nil
                })
            }
        }
    }
    
    private func extractTextFromImage(_ image: UIImage) {
        isExtractingText = true
        extractedText = ""
        
        TextRecognitionService.shared.recognizeText(in: image) { result in
            DispatchQueue.main.async {
                self.isExtractingText = false
                
                switch result {
                case .success(let text):
                    self.extractedText = text
                    // Pre-fill contextText with extracted text, but append if there's already text
                    if self.contextText.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty {
                        self.contextText = text
                    } else {
                        // If user already typed something, append extracted text
                        self.contextText = self.contextText + "\n\n" + text
                    }
                case .failure(let error):
                    print("❌ Failed to extract text: \(error.localizedDescription)")
                    // Don't show error to user, just continue without extracted text
                    self.extractedText = ""
                }
            }
        }
    }
    
    private func analyzeConversation() {
        // Require at least one input method
        let hasScreenshot = selectedScreenshot != nil
        let hasText = !contextText.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty
        
        guard hasScreenshot || hasText else { return }
        
        isAnalyzing = true
        
        // Use the contextText which should already contain extracted text if available
        var inputText = contextText.trimmingCharacters(in: .whitespacesAndNewlines)
        
        // If we have a screenshot but no text was extracted or entered, provide a fallback
        if hasScreenshot && inputText.isEmpty {
            inputText = "Please analyze this conversation screenshot and provide blackpill dating advice."
        }
        
        // Call the API
        RizzAdviceService.shared.getAdvice(input: inputText) { result in
            DispatchQueue.main.async {
                isAnalyzing = false
                
                switch result {
                case .success(let advice):
                    // Increment usage counter (only for free users)
                    usageTracker.incrementRizzCoach()
                    
                    // Try to parse as structured JSON first
                    if let jsonData = advice.data(using: .utf8),
                       let json = try? JSONSerialization.jsonObject(with: jsonData) as? [String: Any],
                       let analysisText = json["analysis"] as? String,
                       let suggestionsArray = json["suggestions"] as? [[String: Any]] {
                        // Structured format - parse properly
                        let analysis = ConversationAnalysis(
                            tone: "Analyzed",
                            vibe: "Blackpill perspective",
                            interestLevel: .unclear,
                            redFlags: [],
                            greenFlags: [],
                            overallAssessment: analysisText
                        )
                        
                        // Parse structured suggestions
                        let suggestions = suggestionsArray.compactMap { suggestionDict -> ResponseSuggestion? in
                            guard let reply = suggestionDict["reply"] as? String,
                                  let tone = suggestionDict["tone"] as? String,
                                  let why = suggestionDict["why"] as? String else {
                                return nil
                            }
                            return ResponseSuggestion(
                                text: reply,
                                tone: tone,
                                reasoning: why
                            )
                        }
                        
                        self.adviceResults = RizzAdviceResults(
                            analysis: analysis,
                            responseSuggestions: suggestions.isEmpty ? parseAdviceToSuggestions(analysisText) : suggestions,
                            tips: extractTips(analysisText),
                            interestLevel: .unclear
                        )
                    } else {
                        // Fallback to old format (plain text)
                        let analysis = ConversationAnalysis(
                            tone: "Analyzed",
                            vibe: "Blackpill perspective",
                            interestLevel: .unclear,
                            redFlags: [],
                            greenFlags: [],
                            overallAssessment: advice
                        )
                        
                        // Create response suggestions from the advice
                        let suggestions = parseAdviceToSuggestions(advice)
                        
                        self.adviceResults = RizzAdviceResults(
                            analysis: analysis,
                            responseSuggestions: suggestions,
                            tips: extractTips(advice),
                            interestLevel: .unclear
                        )
                    }
                    self.showResults = true
                    
                case .failure(let error):
                    print("Error getting advice: \(error.localizedDescription)")
                    // Show error - for now just use placeholder
                    self.adviceResults = RizzAdviceResults.placeholder
                    self.showResults = true
                }
            }
        }
    }
    
    private func parseAdviceToSuggestions(_ advice: String) -> [ResponseSuggestion] {
        // Simple parsing - extract potential response suggestions from advice
        // In production, the AI could return structured suggestions
        let sentences = advice.components(separatedBy: ". ")
        var suggestions: [ResponseSuggestion] = []
        
        // Look for sentences that might be response suggestions
        for (index, sentence) in sentences.prefix(3).enumerated() {
            if sentence.count > 20 && sentence.count < 150 {
                suggestions.append(ResponseSuggestion(
                    text: sentence.trimmingCharacters(in: .whitespacesAndNewlines),
                    tone: "Direct",
                    reasoning: "Based on blackpill analysis"
                ))
            }
        }
        
        // If no good suggestions found, create generic ones
        if suggestions.isEmpty {
            suggestions = [
                ResponseSuggestion(
                    text: "Keep it short and direct",
                    tone: "Direct",
                    reasoning: "Based on analysis"
                )
            ]
        }
        
        return suggestions
    }
    
    private func extractTips(_ advice: String) -> [String] {
        // Extract tips from advice text
        var tips: [String] = []
        let lines = advice.components(separatedBy: "\n")
        
        for line in lines {
            let trimmed = line.trimmingCharacters(in: .whitespacesAndNewlines)
            if trimmed.hasPrefix("-") || trimmed.hasPrefix("•") || trimmed.hasPrefix("*") {
                tips.append(trimmed.replacingOccurrences(of: "- ", with: "")
                    .replacingOccurrences(of: "• ", with: "")
                    .replacingOccurrences(of: "* ", with: ""))
            }
        }
        
        // If no tips found, create generic ones
        if tips.isEmpty {
            tips = [
                "Focus on looksmaxxing",
                "Be realistic about your SMV",
                "Don't bluepill cope"
            ]
        }
        
        return Array(tips.prefix(4))
    }
}

// MARK: - Results Models
struct RizzAdviceResults {
    let analysis: ConversationAnalysis
    let responseSuggestions: [ResponseSuggestion]
    let tips: [String]
    let interestLevel: InterestLevel
    
    static var placeholder: RizzAdviceResults {
        RizzAdviceResults(
            analysis: ConversationAnalysis(
                tone: "Friendly but slightly distant",
                vibe: "She's responding but not initiating",
                interestLevel: .moderate,
                redFlags: ["Taking long to respond", "Short replies"],
                greenFlags: ["Still engaging", "Asking questions"],
                overallAssessment: "She seems interested but might be playing it cool. Keep the conversation light and fun, but don't overdo it."
            ),
            responseSuggestions: [
                ResponseSuggestion(
                    text: "Hey! Just saw your story, that looks amazing 😍",
                    tone: "Playful and engaging",
                    reasoning: "Shows you're paying attention without being too intense"
                ),
                ResponseSuggestion(
                    text: "Haha that's hilarious 😂 What are you up to today?",
                    tone: "Casual and curious",
                    reasoning: "Keeps the conversation flowing naturally"
                ),
                ResponseSuggestion(
                    text: "Sounds fun! Want to grab coffee this weekend?",
                    tone: "Direct but not pushy",
                    reasoning: "Moves toward meeting up without being too forward"
                )
            ],
            tips: [
                "Don't double text - wait for her response",
                "Keep messages light and fun",
                "Ask open-ended questions to keep conversation going",
                "Don't overuse emojis - one or two is enough"
            ],
            interestLevel: .moderate
        )
    }
}

struct ConversationAnalysis {
    let tone: String
    let vibe: String
    let interestLevel: InterestLevel
    let redFlags: [String]
    let greenFlags: [String]
    let overallAssessment: String
}

struct ResponseSuggestion {
    let text: String
    let tone: String
    let reasoning: String
}

enum InterestLevel: String {
    case high = "High"
    case moderate = "Moderate"
    case low = "Low"
    case unclear = "Unclear"
    
    var color: Color {
        switch self {
        case .high: return .green
        case .moderate: return .yellow
        case .low: return .orange
        case .unclear: return .gray
        }
    }
}

// MARK: - Results View
struct RizzAdviceResultsView: View {
    let results: RizzAdviceResults
    let onBack: () -> Void
    
    var body: some View {
        ScrollView {
            VStack(spacing: 24) {
                // Header
                VStack(spacing: 8) {
                    Image(systemName: "checkmark.circle.fill")
                        .font(.system(size: 50))
                        .foregroundColor(.green)
                    
                    Text("Analysis Complete")
                        .font(.title2)
                        .fontWeight(.bold)
                        .foregroundColor(.white)
                }
                .padding(.top, 20)
                
                // Interest Level Badge
                HStack {
                    Text("Interest Level:")
                        .foregroundColor(.gray)
                    Text(results.interestLevel.rawValue)
                        .foregroundColor(results.interestLevel.color)
                        .fontWeight(.bold)
                }
                .padding()
                .background(
                    RoundedRectangle(cornerRadius: 12)
                        .fill(Color.white.opacity(0.1))
                )
                .padding(.horizontal)
                
                // Main Advice Section
                VStack(alignment: .leading, spacing: 16) {
                    Text("Chad's advice")
                        .font(.headline)
                        .foregroundColor(.white)
                    
                    Text(results.analysis.overallAssessment)
                        .font(.body)
                        .foregroundColor(.white)
                        .padding()
                        .frame(maxWidth: .infinity, alignment: .leading)
                        .background(
                            RoundedRectangle(cornerRadius: 12)
                                .fill(Color.white.opacity(0.05))
                        )
                    
                    if !results.analysis.tone.isEmpty && results.analysis.tone != "Analyzed" {
                        AnalysisCard(title: "Tone", value: results.analysis.tone)
                    }
                    if !results.analysis.vibe.isEmpty && results.analysis.vibe != "Blackpill perspective" {
                        AnalysisCard(title: "Vibe", value: results.analysis.vibe)
                    }
                    
                    // Flags
                    if !results.analysis.greenFlags.isEmpty {
                        VStack(alignment: .leading, spacing: 8) {
                            Text("✅ Green Flags")
                                .font(.subheadline)
                                .fontWeight(.semibold)
                                .foregroundColor(.green)
                            
                            ForEach(results.analysis.greenFlags, id: \.self) { flag in
                                Text("• \(flag)")
                                    .font(.caption)
                                    .foregroundColor(.gray)
                            }
                        }
                    }
                    
                    if !results.analysis.redFlags.isEmpty {
                        VStack(alignment: .leading, spacing: 8) {
                            Text("⚠️ Red Flags")
                                .font(.subheadline)
                                .fontWeight(.semibold)
                                .foregroundColor(.orange)
                            
                            ForEach(results.analysis.redFlags, id: \.self) { flag in
                                Text("• \(flag)")
                                    .font(.caption)
                                    .foregroundColor(.gray)
                            }
                        }
                    }
                }
                .padding()
                .background(
                    RoundedRectangle(cornerRadius: 16)
                        .fill(Color.white.opacity(0.1))
                        .overlay(
                            RoundedRectangle(cornerRadius: 16)
                                .stroke(Color.white.opacity(0.2), lineWidth: 1)
                        )
                )
                .padding(.horizontal)
                
                // Response Suggestions
                VStack(alignment: .leading, spacing: 16) {
                    Text("Response Suggestions")
                        .font(.headline)
                        .foregroundColor(.white)
                    
                    ForEach(Array(results.responseSuggestions.enumerated()), id: \.offset) { index, suggestion in
                        ResponseSuggestionCard(suggestion: suggestion, index: index + 1)
                    }
                }
                .padding()
                .background(
                    RoundedRectangle(cornerRadius: 16)
                        .fill(Color.white.opacity(0.1))
                        .overlay(
                            RoundedRectangle(cornerRadius: 16)
                                .stroke(Color.white.opacity(0.2), lineWidth: 1)
                        )
                )
                .padding(.horizontal)
                
                // Tips
                VStack(alignment: .leading, spacing: 16) {
                    Text("Tips")
                        .font(.headline)
                        .foregroundColor(.white)
                    
                    ForEach(results.tips, id: \.self) { tip in
                        HStack(alignment: .top, spacing: 12) {
                            Image(systemName: "lightbulb.fill")
                                .foregroundColor(.yellow)
                                .font(.caption)
                            Text(tip)
                                .font(.subheadline)
                                .foregroundColor(.gray)
                        }
                    }
                }
                .padding()
                .background(
                    RoundedRectangle(cornerRadius: 16)
                        .fill(Color.white.opacity(0.1))
                        .overlay(
                            RoundedRectangle(cornerRadius: 16)
                                .stroke(Color.white.opacity(0.2), lineWidth: 1)
                        )
                )
                .padding(.horizontal)
                
                // Back Button
                Button(action: onBack) {
                    Text("Analyze Another Conversation")
                        .font(.headline)
                        .foregroundColor(.white)
                        .frame(maxWidth: .infinity)
                        .padding()
                        .background(
                            RoundedRectangle(cornerRadius: 12)
                                .fill(Color.cyan)
                        )
                }
                .padding(.horizontal)
                .padding(.bottom, 30)
            }
        }
    }
}

struct AnalysisCard: View {
    let title: String
    let value: String
    
    var body: some View {
        VStack(alignment: .leading, spacing: 4) {
            Text(title)
                .font(.caption)
                .foregroundColor(.gray)
            Text(value)
                .font(.subheadline)
                .foregroundColor(.white)
        }
    }
}

struct ResponseSuggestionCard: View {
    let suggestion: ResponseSuggestion
    let index: Int
    
    var body: some View {
        VStack(alignment: .leading, spacing: 12) {
            HStack {
                Text("Option \(index)")
                    .font(.caption)
                    .fontWeight(.semibold)
                    .foregroundColor(.cyan)
                Spacer()
            }
            
            Text(suggestion.text)
                .font(.body)
                .foregroundColor(.white)
                .padding()
                .frame(maxWidth: .infinity, alignment: .leading)
                .background(
                    RoundedRectangle(cornerRadius: 8)
                        .fill(Color.white.opacity(0.05))
                )
            
            VStack(alignment: .leading, spacing: 4) {
                Text("Tone: \(suggestion.tone)")
                    .font(.caption)
                    .foregroundColor(.gray)
                Text("Why: \(suggestion.reasoning)")
                    .font(.caption)
                    .foregroundColor(.gray)
            }
            
            Button(action: {
                UIPasteboard.general.string = suggestion.text
            }) {
                HStack {
                    Image(systemName: "doc.on.doc")
                    Text("Copy")
                }
                .font(.caption)
                .foregroundColor(.cyan)
                .padding(.horizontal, 12)
                .padding(.vertical, 6)
                .background(
                    RoundedRectangle(cornerRadius: 6)
                        .fill(Color.cyan.opacity(0.2))
                )
            }
        }
        .padding()
        .background(
            RoundedRectangle(cornerRadius: 12)
                .fill(Color.white.opacity(0.05))
                .overlay(
                    RoundedRectangle(cornerRadius: 12)
                        .stroke(Color.white.opacity(0.1), lineWidth: 1)
                )
        )
    }
}

