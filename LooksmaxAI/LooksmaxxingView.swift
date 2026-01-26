//
//  LooksmaxxingView.swift
//  LooksmaxAI
//
//  Created by Admin on 2026. 01. 11..
//

import SwiftUI

struct LooksmaxxingView: View {
    @Environment(\.dismiss) var dismiss
    @State private var selectedResult: FaceAnalysisResults? = nil
    @State private var showResultPicker = false
    @State private var isAnalyzing = false
    @State private var showResults = false
    @State private var adviceText: String = ""
    @StateObject private var usageTracker = UsageTracker.shared
    @StateObject private var subscriptionManager = SubscriptionManager.shared
    @State private var isShowingUpgrade = false
    
    // User input fields
    @State private var height: HeightOption = .notSelected
    @State private var race: RaceEthnicity = .notSelected
    @State private var weight: WeightBracket = .notSelected
    @State private var bodyFat: BodyFatPercentage = .notSelected
    @State private var norwood: NorwoodScale = .nw0
    @State private var hairDensity: HairDensity = .dense
    @State private var gymStatus: GymStatus = .notStarted
    @State private var skinCondition: SkinCondition = .clear
    @State private var frame: FrameType = .average
    @State private var bodyType: BodyType = .lean
    @State private var voice: VoiceType = .average
    @State private var socialStatus: SocialStatus = .average
    
    private var history: [ResultHistoryItem] {
        ResultHistoryManager.shared.loadHistory()
    }
    
    var body: some View {
        NavigationStack {
            ZStack {
                Color.black.ignoresSafeArea()
                
                if !showResults {
                    formView
                } else {
                    resultsView
                }
            }
            .navigationTitle("Looksmaxxing")
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
            }
            .fullScreenCover(isPresented: $isShowingUpgrade) {
                UpgradeView()
            }
            .onDisappear {
                // Show tab bar when view is dismissed
                NotificationCenter.default.post(name: NSNotification.Name("ShowTabBar"), object: nil)
            }
        }
    }
    
    private var formView: some View {
        ScrollView {
            VStack(spacing: 24) {
                // Header
                headerSection
                
                // Face Rating Selection
                faceRatingSection
                
                // User Inputs
                userInputsSection
                
                // Submit Button
                submitButton
            }
            .padding()
        }
    }
    
    private var headerSection: some View {
        VStack(spacing: 12) {
            Text("Looksmaxxing Advisor")
                .font(.title)
                .fontWeight(.bold)
                .foregroundColor(.white)
            
            Text("Get proper looksmaxxing advice on how to ascend and improve your looks")
                .font(.subheadline)
                .foregroundColor(.gray)
                .multilineTextAlignment(.center)
        }
        .padding(.top, 20)
    }
    
    private var faceRatingSection: some View {
        VStack(alignment: .leading, spacing: 12) {
            Text("Face Rating (Optional)")
                .font(.headline)
                .foregroundColor(.white)
            
            Text("Select a previous face analysis to include PSL and facial metrics")
                .font(.caption)
                .foregroundColor(.gray)
            
            Button(action: {
                showResultPicker = true
            }) {
                HStack {
                    if let result = selectedResult {
                        VStack(alignment: .leading, spacing: 4) {
                            Text("Selected: PSL \(result.overall.psl ?? 0, specifier: "%.1f")")
                                .foregroundColor(.white)
                            Text("Tap to change")
                                .font(.caption)
                                .foregroundColor(.gray)
                        }
                    } else {
                        Text("Tap to select face rating")
                            .foregroundColor(.gray)
                    }
                    Spacer()
                    Image(systemName: "chevron.right")
                        .foregroundColor(.gray)
                }
                .padding()
                .background(
                    RoundedRectangle(cornerRadius: 12)
                        .fill(Color.white.opacity(0.1))
                        .overlay(
                            RoundedRectangle(cornerRadius: 12)
                                .stroke(Color.white.opacity(0.2), lineWidth: 1)
                        )
                )
            }
        }
        .padding()
        .background(sectionBackground)
        .sheet(isPresented: $showResultPicker) {
            ResultPickerView(selectedResult: $selectedResult)
        }
    }
    
    private var userInputsSection: some View {
        VStack(spacing: 16) {
            // Height
            PickerField(title: "Height", selection: $height)
            
            // Race/Ethnicity
            PickerField(title: "Race/Ethnicity", selection: $race)
            
            // Weight
            PickerField(title: "Weight", selection: $weight)
            
            // Body Fat %
            PickerField(title: "Body Fat %", selection: $bodyFat)
            
            // Norwood Scale
            PickerField(title: "Norwood Scale", selection: $norwood)
            
            // Hair Density
            PickerField(title: "Hair Density/Thickness", selection: $hairDensity)
            
            // Gym Status
            PickerField(title: "Current Gym Status", selection: $gymStatus)
            
            // Skin Condition
            PickerField(title: "Skin Condition", selection: $skinCondition)
            
            // Frame
            PickerField(title: "Frame", selection: $frame)
            
            // Body Type
            PickerField(title: "Body Type", selection: $bodyType)
            
            // Voice
            PickerField(title: "Voice", selection: $voice)
            
            // Social Status
            PickerField(title: "Social Status/Inhib", selection: $socialStatus)
        }
        .padding()
        .background(sectionBackground)
    }
    
    private var sectionBackground: some View {
        RoundedRectangle(cornerRadius: 16)
            .fill(Color.white.opacity(0.1))
            .overlay(
                RoundedRectangle(cornerRadius: 16)
                    .stroke(Color.white.opacity(0.2), lineWidth: 1)
            )
    }
    
    private var submitButton: some View {
        VStack(spacing: 12) {
            // Show usage limit for free users
            if !subscriptionManager.isPro && usageTracker.hasUsedLooksmaxxing() {
                VStack(spacing: 8) {
                    Text("You've used your free Looksmaxxing advice")
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
                if usageTracker.canUseLooksmaxxing() {
                    getLooksmaxAdvice()
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
                        Text("Get Looksmaxxing Advice")
                    }
                }
                .font(.headline)
                .foregroundColor(.white)
                .frame(maxWidth: .infinity)
                .padding()
                .background(
                    RoundedRectangle(cornerRadius: 12)
                        .fill(isFormValid ? Color.purple : Color.gray.opacity(0.3))
                )
            }
            .disabled(!isFormValid || isAnalyzing || (!subscriptionManager.isPro && usageTracker.hasUsedLooksmaxxing()))
            .padding(.bottom, 30)
        }
    }
    
    private var isFormValid: Bool {
        // At least some inputs should be filled
        height != .notSelected || race != .notSelected || weight != .notSelected || bodyFat != .notSelected || selectedResult != nil
    }
    
    private var resultsView: some View {
        ScrollView {
            VStack(spacing: 24) {
                // Header
                VStack(spacing: 8) {
                    Image(systemName: "checkmark.circle.fill")
                        .font(.system(size: 50))
                        .foregroundColor(.green)
                    
                    Text("Looksmaxxing Analysis")
                        .font(.title2)
                        .fontWeight(.bold)
                        .foregroundColor(.white)
                }
                .padding(.top, 20)
                
                // Advice Text
                VStack(alignment: .leading, spacing: 12) {
                    Text("Looksmaxxing Advice")
                        .font(.headline)
                        .foregroundColor(.white)
                    
                    Text(adviceText)
                        .font(.body)
                        .foregroundColor(.white)
                        .padding()
                        .frame(maxWidth: .infinity, alignment: .leading)
                        .background(
                            RoundedRectangle(cornerRadius: 12)
                                .fill(Color.white.opacity(0.05))
                        )
                }
                .padding()
                .background(sectionBackground)
                .padding(.horizontal)
                
                // Disclaimer
                VStack(alignment: .leading, spacing: 8) {
                    HStack(spacing: 8) {
                        Image(systemName: "exclamationmark.triangle.fill")
                            .foregroundColor(.orange)
                            .font(.caption)
                        Text("Medical Disclaimer")
                            .font(.caption)
                            .fontWeight(.semibold)
                            .foregroundColor(.orange)
                    }
                    
                    Text("We are not medical professionals or healthcare providers. All suggestions, recommendations, and advice provided in this analysis are for informational purposes only and should not be considered medical, health, or professional advice. Before implementing any suggestions related to health, fitness, supplements, medications, or medical procedures, please consult with a qualified healthcare provider, physician, or licensed medical professional. Do not disregard professional medical advice or delay seeking it because of information provided here.")
                        .font(.caption2)
                        .foregroundColor(.gray)
                        .lineSpacing(2)
                }
                .padding()
                .background(
                    RoundedRectangle(cornerRadius: 12)
                        .fill(Color.orange.opacity(0.1))
                        .overlay(
                            RoundedRectangle(cornerRadius: 12)
                                .stroke(Color.orange.opacity(0.3), lineWidth: 1)
                        )
                )
                .padding(.horizontal)
                
                // Back Button
                Button(action: {
                    showResults = false
                    adviceText = ""
                }) {
                    Text("New Analysis")
                        .font(.headline)
                        .foregroundColor(.white)
                        .frame(maxWidth: .infinity)
                        .padding()
                        .background(
                            RoundedRectangle(cornerRadius: 12)
                                .fill(Color.purple)
                        )
                }
                .padding(.horizontal)
                .padding(.bottom, 30)
            }
        }
    }
    
    private func getLooksmaxAdvice() {
        isAnalyzing = true
        
        // Build PSL data from selected result
        var pslData: [String: Any] = [:]
        if let result = selectedResult {
            pslData["psl"] = result.overall.psl ?? 0
            pslData["rating"] = getRatingCategory(psl: result.overall.psl ?? 0)
            pslData["eyes"] = [
                "canthalTilt": result.eyes.canthalTilt,
                "orbitalDepth": result.eyes.orbitalDepth,
                "eyelidExposure": result.eyes.eyelidExposure
            ]
            pslData["midface"] = [
                "ipd": result.midface.ipd,
                "fwhr": result.midface.fwhr,
                "compactness": result.midface.compactness,
                "cheekbones": result.midface.cheekbones,
                "nose": result.midface.nose
            ]
            pslData["lowerThird"] = [
                "mandible": result.lowerThird.mandible,
                "jawWidth": result.lowerThird.jawWidth,
                "lips": result.lowerThird.lips,
                "ramus": result.lowerThird.ramus
            ]
        }
        
        // Build user inputs
        var userInputs: [String: String] = [:]
        if height != .notSelected { userInputs["height"] = height.rawValue }
        if race != .notSelected { userInputs["race"] = race.rawValue }
        if weight != .notSelected { userInputs["weight"] = weight.rawValue }
        if bodyFat != .notSelected { userInputs["bodyFat"] = bodyFat.rawValue }
        userInputs["norwood"] = norwood.rawValue
        userInputs["hairDensity"] = hairDensity.rawValue
        userInputs["gymStatus"] = gymStatus.rawValue
        userInputs["skinCondition"] = skinCondition.rawValue
        userInputs["frame"] = frame.rawValue
        userInputs["bodyType"] = bodyType.rawValue
        userInputs["voice"] = voice.rawValue
        userInputs["socialStatus"] = socialStatus.rawValue
        
        // Call API
        LooksmaxAdviceService.shared.getAdvice(pslData: pslData, userInputs: userInputs) { result in
            DispatchQueue.main.async {
                isAnalyzing = false
                
                switch result {
                case .success(let advice):
                    // Increment usage counter (only for free users)
                    usageTracker.incrementLooksmaxxing()
                    adviceText = advice
                    showResults = true
                case .failure(let error):
                    print("Error getting advice: \(error.localizedDescription)")
                    adviceText = "Error: \(error.localizedDescription)"
                    showResults = true
                }
            }
        }
    }
    
    private func getRatingCategory(psl: Double) -> String {
        let scaleManager = ScaleManager.shared
        let pslScale = scaleManager.convertToPSLScale(psl)
        
        if pslScale >= 7.0 { return "Chad/Adam" }
        if pslScale >= 5.5 { return "Chadlite" }
        if pslScale >= 4.0 { return "HTN (High Tier Normie)" }
        if pslScale >= 2.5 { return "MTN (Mid Tier Normie)" }
        if pslScale >= 1.0 { return "LTN (Low Tier Normie)" }
        return "Subhuman"
    }
}

// MARK: - Input Components
struct InputField: View {
    let title: String
    @Binding var value: String
    let placeholder: String
    
    var body: some View {
        VStack(alignment: .leading, spacing: 8) {
            Text(title)
                .font(.subheadline)
                .fontWeight(.semibold)
                .foregroundColor(.white)
            
            TextField(placeholder, text: $value)
                .textFieldStyle(.plain)
                .padding()
                .background(Color.white.opacity(0.05))
                .cornerRadius(8)
                .foregroundColor(.white)
                .overlay(
                    RoundedRectangle(cornerRadius: 8)
                        .stroke(Color.white.opacity(0.2), lineWidth: 1)
                )
        }
    }
}

struct PickerField<T: RawRepresentable & CaseIterable & Hashable>: View where T.RawValue == String {
    let title: String
    @Binding var selection: T
    
    var body: some View {
        VStack(alignment: .leading, spacing: 8) {
            Text(title)
                .font(.subheadline)
                .fontWeight(.semibold)
                .foregroundColor(.white)
            
            Picker(title, selection: $selection) {
                ForEach(Array(T.allCases), id: \.self) { option in
                    Text(option.rawValue).tag(option)
                }
            }
            .pickerStyle(.menu)
            .padding()
            .background(Color.white.opacity(0.05))
            .cornerRadius(8)
            .overlay(
                RoundedRectangle(cornerRadius: 8)
                    .stroke(Color.white.opacity(0.2), lineWidth: 1)
            )
            .foregroundColor(.white)
        }
    }
}

// MARK: - Enums
enum HeightOption: String, CaseIterable {
    case notSelected = "Not Selected"
    case under56 = "<5'6"
    case five6 = "5'6"
    case five7 = "5'7"
    case five8 = "5'8"
    case five9 = "5'9"
    case five10 = "5'10"
    case five11 = "5'11"
    case six0 = "6'0"
    case six1 = "6'1"
    case six2 = "6'2"
    case six3 = "6'3"
    case six4 = "6'4"
    case six5 = "6'5"
    case over65 = ">6'5"
}

enum WeightBracket: String, CaseIterable {
    case notSelected = "Not Selected"
    case under100 = "<100 lbs"
    case w100_105 = "100-105 lbs"
    case w105_110 = "105-110 lbs"
    case w110_115 = "110-115 lbs"
    case w115_120 = "115-120 lbs"
    case w120_125 = "120-125 lbs"
    case w125_130 = "125-130 lbs"
    case w130_135 = "130-135 lbs"
    case w135_140 = "135-140 lbs"
    case w140_145 = "140-145 lbs"
    case w145_150 = "145-150 lbs"
    case w150_155 = "150-155 lbs"
    case w155_160 = "155-160 lbs"
    case w160_165 = "160-165 lbs"
    case w165_170 = "165-170 lbs"
    case w170_175 = "170-175 lbs"
    case w175_180 = "175-180 lbs"
    case w180_185 = "180-185 lbs"
    case w185_190 = "185-190 lbs"
    case w190_195 = "190-195 lbs"
    case w195_200 = "195-200 lbs"
    case w200_205 = "200-205 lbs"
    case w205_210 = "205-210 lbs"
    case w210_215 = "210-215 lbs"
    case w215_220 = "215-220 lbs"
    case w220_225 = "220-225 lbs"
    case w225_230 = "225-230 lbs"
    case w230_235 = "230-235 lbs"
    case w235_240 = "235-240 lbs"
    case w240_245 = "240-245 lbs"
    case w245_250 = "245-250 lbs"
    case over250 = ">250 lbs"
}

enum BodyFatPercentage: String, CaseIterable {
    case notSelected = "Not Selected"
    case veryLow = "<8% (Very Low - Competitive)"
    case low = "8-12% (Low - Shredded)"
    case lean = "12-15% (Lean - Athletic)"
    case average = "15-18% (Average - Fit)"
    case moderate = "18-22% (Moderate - Normal)"
    case high = "22-25% (High - Overweight)"
    case veryHigh = "25-30% (Very High - Obese)"
    case extremelyHigh = ">30% (Extremely High - Morbidly Obese)"
}

enum RaceEthnicity: String, CaseIterable {
    case notSelected = "Not Selected"
    case white = "White/Caucasian"
    case black = "Black"
    case africanAmerican = "African American"
    case asian = "Asian"
    case hispanic = "Hispanic/Latino"
    case middleEastern = "Middle Eastern"
    case indian = "Indian"
    case pakistani = "Pakistani"
    case bangladeshi = "Bangladeshi"
    case somali = "Somali"
    case ethiopian = "Ethiopian"
    case nigerian = "Nigerian"
    case slav = "Slav"
    case russian = "Russian"
    case eastAsian = "East Asian (Chinese, Japanese, Korean)"
    case southeastAsian = "Southeast Asian (Filipino, Vietnamese, Thai)"
    case mixedBlackWhite = "Mixed: Half Black Half White"
    case mixedAsianBlack = "Mixed: Half Asian Half Black"
    case mixedWhiteAsian = "Mixed: Half White Half Asian"
    case mixedOther = "Mixed Race (Other)"
    case nativeAmerican = "Native American"
    case pacificIslander = "Pacific Islander"
    case jewish = "Jewish"
    case arab = "Arab"
    case persian = "Persian/Iranian"
    case turkish = "Turkish"
    case greek = "Greek"
    case italian = "Italian"
    case scandinavian = "Scandinavian"
    case other = "Other"
}

enum NorwoodScale: String, CaseIterable {
    case nw0 = "NW0 (Perfect)"
    case nw1 = "NW1"
    case nw2 = "NW2"
    case nw3 = "NW3+"
    case balding = "Balding Aggressively"
}

enum HairDensity: String, CaseIterable {
    case dense = "Dense"
    case average = "Average"
    case thinning = "Thinning"
}

enum GymStatus: String, CaseIterable {
    case notStarted = "Not Started"
    case beginner = "Beginner (< 6 months)"
    case intermediate = "Intermediate (6-24 months)"
    case advanced = "Advanced (2+ years)"
    case competitive = "Competitive/Elite"
}

enum SkinCondition: String, CaseIterable {
    case clear = "Clear/Good"
    case mild = "Mild Acne/Issues"
    case moderate = "Moderate Acne/Issues"
    case severe = "Severe Acne/Scarring"
    case excellent = "Excellent (Glowing)"
}

enum FrameType: String, CaseIterable {
    case narrow = "Narrow Shoulders"
    case average = "Average Shoulders"
    case broad = "Broad Shoulders"
    case shortClavicle = "Short Clavicle"
    case longClavicle = "Long Clavicle"
}

enum BodyType: String, CaseIterable {
    case skinnyFat = "Skinny-Fat"
    case lean = "Lean"
    case muscular = "Muscular"
    case fat = "Fat/Overweight"
    case shredded = "Shredded (10-15% BF)"
}

enum VoiceType: String, CaseIterable {
    case deep = "Deep"
    case average = "Average"
    case highPitched = "High-Pitched"
    case nasal = "Nasal"
}

enum SocialStatus: String, CaseIterable {
    case highInhib = "High Inhib (Shy/Anxious)"
    case average = "Average"
    case lowInhib = "Low Inhib (Confident)"
    case veryLowInhib = "Very Low Inhib (Alpha)"
}

// MARK: - Result Picker
struct ResultPickerView: View {
    @Binding var selectedResult: FaceAnalysisResults?
    @Environment(\.dismiss) var dismiss
    @StateObject private var scaleManager = ScaleManager.shared
    
    private var history: [ResultHistoryItem] {
        ResultHistoryManager.shared.loadHistory()
    }
    
    var body: some View {
        NavigationStack {
            ZStack {
                Color.black.ignoresSafeArea()
                
                if history.isEmpty {
                    VStack(spacing: 20) {
                        Image(systemName: "chart.bar.fill")
                            .font(.system(size: 50))
                            .foregroundColor(.gray)
                        Text("No Results Yet")
                            .font(.title2)
                            .foregroundColor(.white)
                        Text("Complete a face rating first")
                            .foregroundColor(.gray)
                    }
                } else {
                    ScrollView {
                        LazyVStack(spacing: 16) {
                            ForEach(history, id: \.id) { item in
                                ResultPickerCard(
                                    item: item,
                                    isSelected: selectedResult?.overall.psl == item.result.overall.psl,
                                    scaleManager: scaleManager,
                                    onTap: {
                                        selectedResult = item.result
                                        dismiss()
                                    }
                                )
                            }
                        }
                        .padding()
                    }
                }
            }
            .navigationTitle("Select Face Rating")
            .navigationBarTitleDisplayMode(.inline)
            .toolbar {
                ToolbarItem(placement: .navigationBarTrailing) {
                    Button("Cancel") {
                        dismiss()
                    }
                    .foregroundColor(.white)
                }
            }
        }
    }
}

struct ResultPickerCard: View {
    let item: ResultHistoryItem
    let isSelected: Bool
    let scaleManager: ScaleManager
    let onTap: () -> Void
    @State private var thumbnailImage: UIImage? = nil
    
    private var psl: Double {
        item.result.overall.psl ?? 0
    }
    
    private var display: (score: String, label: String) {
        scaleManager.getFullDisplay(psl, gender: item.gender)
    }
    
    private var ratingCategory: String {
        getRatingCategory(psl: psl)
    }
    
    var body: some View {
        Button(action: onTap) {
            HStack(spacing: 16) {
                thumbnailView
                infoSection
                Spacer()
                selectionIndicator
            }
            .padding()
            .background(cardBackground)
        }
        .buttonStyle(PlainButtonStyle())
        .onAppear {
            loadThumbnail()
        }
    }
    
    private var thumbnailView: some View {
        Group {
            if let thumbnail = thumbnailImage {
                Image(uiImage: thumbnail)
                    .resizable()
                    .aspectRatio(contentMode: .fill)
            } else {
                Image(systemName: "person.fill")
                    .foregroundColor(.gray.opacity(0.5))
                    .font(.system(size: 30))
            }
        }
        .frame(width: 80, height: 80)
        .clipShape(RoundedRectangle(cornerRadius: 12))
        .overlay(thumbnailOverlay)
        .background(thumbnailBackground)
    }
    
    private var thumbnailOverlay: some View {
        RoundedRectangle(cornerRadius: 12)
            .stroke(isSelected ? Color.cyan : Color.white.opacity(0.2), lineWidth: isSelected ? 2 : 1)
    }
    
    private var thumbnailBackground: some View {
        RoundedRectangle(cornerRadius: 12)
            .fill(Color.white.opacity(0.05))
    }
    
    private var infoSection: some View {
        VStack(alignment: .leading, spacing: 8) {
            HStack(spacing: 8) {
                Text(display.score)
                    .font(.system(size: 28, weight: .bold))
                    .foregroundColor(.white)
                
                Text(display.label)
                    .font(.system(size: 12, weight: .semibold))
                    .foregroundColor(.cyan)
                    .padding(.horizontal, 8)
                    .padding(.vertical, 4)
                    .background(
                        Capsule()
                            .fill(Color.cyan.opacity(0.2))
                    )
            }
            
            Text(ratingCategory)
                .font(.system(size: 14, weight: .medium))
                .foregroundColor(.purple)
            
            Text(item.timestamp, style: .date)
                .font(.system(size: 12))
                .foregroundColor(.gray)
        }
    }
    
    private var selectionIndicator: some View {
        Group {
            if isSelected {
                Image(systemName: "checkmark.circle.fill")
                    .font(.system(size: 24))
                    .foregroundColor(.cyan)
            } else {
                Image(systemName: "chevron.right")
                    .font(.system(size: 16))
                    .foregroundColor(.gray.opacity(0.5))
            }
        }
    }
    
    private var cardBackground: some View {
        RoundedRectangle(cornerRadius: 16)
            .fill(isSelected ? Color.cyan.opacity(0.1) : Color.white.opacity(0.05))
            .overlay(cardBorder)
    }
    
    private var cardBorder: some View {
        RoundedRectangle(cornerRadius: 16)
            .stroke(isSelected ? Color.cyan.opacity(0.5) : Color.white.opacity(0.2), lineWidth: isSelected ? 2 : 1)
    }
    
    private func loadThumbnail() {
        guard let path = item.thumbnailPath else { return }
        thumbnailImage = ResultHistoryManager.shared.loadThumbnail(for: path)
    }
    
    private func getRatingCategory(psl: Double) -> String {
        let pslScale = scaleManager.convertToPSLScale(psl)
        
        if pslScale >= 7.0 { return "Chad/Adam" }
        if pslScale >= 5.5 { return "Chadlite" }
        if pslScale >= 4.0 { return "HTN (High Tier Normie)" }
        if pslScale >= 2.5 { return "MTN (Mid Tier Normie)" }
        if pslScale >= 1.0 { return "LTN (Low Tier Normie)" }
        return "Subhuman"
    }
}

// MARK: - Service
class LooksmaxAdviceService {
    static let shared = LooksmaxAdviceService()
    
    private let baseURL = "https://looksmax-backend-production.up.railway.app"
    
    private lazy var urlSession: URLSession = {
        let configuration = URLSessionConfiguration.default
        configuration.timeoutIntervalForRequest = 60.0
        configuration.timeoutIntervalForResource = 60.0
        return URLSession(configuration: configuration)
    }()
    
    private init() {}
    
    func getAdvice(pslData: [String: Any], userInputs: [String: String], completion: @escaping (Result<String, Error>) -> Void) {
        guard let url = URL(string: "\(baseURL)/api/looksmax-advice") else {
            completion(.failure(NSError(domain: "LooksmaxAdvice", code: -1, userInfo: [NSLocalizedDescriptionKey: "Invalid URL"])))
            return
        }
        
        var request = URLRequest(url: url)
        request.httpMethod = "POST"
        request.setValue("application/json", forHTTPHeaderField: "Content-Type")
        
        let requestBody: [String: Any] = [
            "pslData": pslData,
            "userInputs": userInputs
        ]
        
        do {
            request.httpBody = try JSONSerialization.data(withJSONObject: requestBody)
        } catch {
            completion(.failure(error))
            return
        }
        
        print("🚀 Sending looksmax advice request")
        
        urlSession.dataTask(with: request) { data, response, error in
            if let error = error {
                completion(.failure(error))
                return
            }
            
            guard let httpResponse = response as? HTTPURLResponse else {
                completion(.failure(NSError(domain: "LooksmaxAdvice", code: -1, userInfo: [NSLocalizedDescriptionKey: "Invalid response"])))
                return
            }
            
            guard httpResponse.statusCode == 200 else {
                let errorMessage = "Server returned status \(httpResponse.statusCode)"
                completion(.failure(NSError(domain: "LooksmaxAdvice", code: httpResponse.statusCode, userInfo: [NSLocalizedDescriptionKey: errorMessage])))
                return
            }
            
            guard let data = data else {
                completion(.failure(NSError(domain: "LooksmaxAdvice", code: -1, userInfo: [NSLocalizedDescriptionKey: "No data received"])))
                return
            }
            
            do {
                if let json = try JSONSerialization.jsonObject(with: data) as? [String: Any],
                   let advice = json["advice"] as? String {
                    completion(.success(advice))
                } else {
                    completion(.failure(NSError(domain: "LooksmaxAdvice", code: -1, userInfo: [NSLocalizedDescriptionKey: "Invalid response format"])))
                }
            } catch {
                completion(.failure(error))
            }
        }.resume()
    }
}

