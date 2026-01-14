//
//  ResultsView.swift
//  LooksmaxAI
//
//  Created by Admin on 2026. 01. 07..
//

import SwiftUI

struct ResultsView: View {
    let results: FaceAnalysisResults
    let onClose: (() -> Void)?
    @StateObject private var subscriptionManager = SubscriptionManager.shared
    @StateObject private var scaleManager = ScaleManager.shared
    @State private var selectedDate = Date()
    @State private var currentSection: ResultsSection = .overall
    @State private var showUnlockSheet = false
    @State private var showDatePicker = false
    @Environment(\.dismiss) var dismiss
    
    init(results: FaceAnalysisResults, onClose: (() -> Void)? = nil) {
        self.results = results
        self.onClose = onClose
    }
    
    var body: some View {
        NavigationStack {
            ZStack {
                Color.black.ignoresSafeArea()
                
                VStack(spacing: 0) {
                    // Header
                    VStack(spacing: 20) {
                        Text("You Will Ascend By")
                            .font(.system(size: 32, weight: .bold))
                            .foregroundColor(.white)
                            .padding(.top, 20)
                    
                    // Date Picker Button
                    Button(action: {
                        showDatePicker = true
                    }) {
                        HStack {
                            Text(selectedDate, style: .date)
                                .font(.system(size: 20, weight: .semibold))
                                .foregroundColor(.orange)
                            Image(systemName: "chevron.down")
                                .foregroundColor(.orange)
                                .font(.system(size: 14))
                        }
                        .padding(.horizontal, 24)
                        .padding(.vertical, 12)
                        .background(
                            RoundedRectangle(cornerRadius: 25)
                                .stroke(Color.orange, lineWidth: 2)
                                .background(
                                    RoundedRectangle(cornerRadius: 25)
                                        .fill(Color.orange.opacity(0.1))
                                )
                        )
                    }
                    .sheet(isPresented: $showDatePicker) {
                        DatePickerSheet(selectedDate: $selectedDate)
                    }
                    .sheet(isPresented: .constant(false)) {
                        // Date picker sheet would go here
                    }
                    
                    // Section Title
                    Text(currentSection.rawValue)
                        .font(.system(size: 18, weight: .medium))
                        .foregroundColor(.cyan)
                }
                .padding(.top, 10)
                
                // Metrics Grid
                ScrollView {
                    LazyVGrid(columns: [
                        GridItem(.flexible(), spacing: 12),
                        GridItem(.flexible(), spacing: 12)
                    ], spacing: 12) {
                        ForEach(getMetricsForSection(currentSection), id: \.id) { metric in
                            MetricCardView(metric: metric)
                        }
                    }
                    .padding(.horizontal, 16)
                    .padding(.top, 20)
                }
                
                // Pagination Dots
                HStack(spacing: 8) {
                    ForEach(ResultsSection.allCases, id: \.self) { section in
                        Circle()
                            .fill(section == currentSection ? Color.cyan : Color.gray.opacity(0.3))
                            .frame(width: 8, height: 8)
                    }
                }
                .padding(.vertical, 16)
                
                    // Unlock button removed - all results are free
                }
            }
            .navigationBarTitleDisplayMode(.inline)
            .toolbar {
                ToolbarItem(placement: .navigationBarTrailing) {
                    Button(action: {
                        print("🔄 Close button tapped in ResultsView")
                        if let onClose = onClose {
                            print("🔄 Calling onClose callback")
                            onClose()
                        } else {
                            print("🔄 No onClose callback, using dismiss")
                            dismiss()
                        }
                    }) {
                        Image(systemName: "xmark.circle.fill")
                            .font(.system(size: 24))
                            .foregroundColor(.white.opacity(0.8))
                    }
                }
            }
        }
        .onAppear {
            // Set initial date from results
            if let date = results.ascensionDate {
                selectedDate = date
            }
        }
        .gesture(
            DragGesture()
                .onEnded { value in
                    if value.translation.width > 50 {
                        // Swipe right - previous section
                        if let currentIndex = ResultsSection.allCases.firstIndex(of: currentSection),
                           currentIndex > 0 {
                            withAnimation {
                                currentSection = ResultsSection.allCases[currentIndex - 1]
                            }
                        }
                    } else if value.translation.width < -50 {
                        // Swipe left - next section
                        if let currentIndex = ResultsSection.allCases.firstIndex(of: currentSection),
                           currentIndex < ResultsSection.allCases.count - 1 {
                            withAnimation {
                                currentSection = ResultsSection.allCases[currentIndex + 1]
                            }
                        }
                    }
                }
        )
    }
    
    private func getMetricsForSection(_ section: ResultsSection) -> [MetricCard] {
        let isLocked = !subscriptionManager.isUnlocked
        
        switch section {
        case .overall:
            // Use the selected scale for PSL display
            let psl = results.overall.psl ?? 50.0
            let potential = results.overall.potential ?? 50.0
            
            // Get display values based on selected scale
            let pslDisplay = scaleManager.getProgressValue(psl)
            let potentialDisplay = scaleManager.getProgressValue(potential)
            
            return [
                MetricCard(title: "PSL", score: pslDisplay, isLocked: isLocked, rawPSL: psl),
                MetricCard(title: "POTENTIAL", score: potentialDisplay, isLocked: isLocked, rawPSL: potential)
            ]
        case .eyes:
            return [
                MetricCard(title: "ORBITAL DEPTH", score: results.eyes.orbitalDepth ?? 50.0, isLocked: isLocked, rawPSL: nil),
                MetricCard(title: "CANTHAL TILT", score: results.eyes.canthalTilt ?? 50.0, isLocked: isLocked, rawPSL: nil),
                MetricCard(title: "EYEBROW DENSI...", score: results.eyes.eyebrowDensity ?? 50.0, isLocked: isLocked, rawPSL: nil),
                MetricCard(title: "EYELASH DENSITY", score: results.eyes.eyelashDensity ?? 50.0, isLocked: isLocked, rawPSL: nil),
                MetricCard(title: "EYELID EXPOSURE", score: results.eyes.eyelidExposure ?? 50.0, isLocked: isLocked, rawPSL: nil),
                MetricCard(title: "UNDER EYE HEAL...", score: results.eyes.underEyeHealth ?? 50.0, isLocked: isLocked, rawPSL: nil)
            ]
        case .midface:
            return [
                MetricCard(title: "CHEEKBONES", score: results.midface.cheekbones ?? 50.0, isLocked: isLocked, rawPSL: nil),
                MetricCard(title: "MAXILLA", score: results.midface.maxilla ?? 50.0, isLocked: isLocked, rawPSL: nil),
                MetricCard(title: "NOSE", score: results.midface.nose ?? 50.0, isLocked: isLocked, rawPSL: nil),
                MetricCard(title: "IPD", score: results.midface.ipd ?? 50.0, isLocked: isLocked, rawPSL: nil),
                MetricCard(title: "FWHR", score: results.midface.fwhr ?? 50.0, isLocked: isLocked, rawPSL: nil),
                MetricCard(title: "COMPACTNESS", score: results.midface.compactness ?? 50.0, isLocked: isLocked, rawPSL: nil)
            ]
        case .lowerThird:
            return [
                MetricCard(title: "LIPS", score: results.lowerThird.lips ?? 50.0, isLocked: isLocked, rawPSL: nil),
                MetricCard(title: "MANDIBLE", score: results.lowerThird.mandible ?? 50.0, isLocked: isLocked, rawPSL: nil),
                MetricCard(title: "GONIAL ANGLE", score: results.lowerThird.gonialAngle ?? 50.0, isLocked: isLocked, rawPSL: nil),
                MetricCard(title: "RAMUS", score: results.lowerThird.ramus ?? 50.0, isLocked: isLocked, rawPSL: nil),
                MetricCard(title: "HYOID SKIN TIG...", score: results.lowerThird.hyoidSkinTightness ?? 50.0, isLocked: isLocked, rawPSL: nil),
                MetricCard(title: "JAW WIDTH", score: results.lowerThird.jawWidth ?? 50.0, isLocked: isLocked, rawPSL: nil)
            ]
        case .upperThird:
            return [
                MetricCard(title: "NORWOOD STAGE", score: results.upperThird.norwoodStage ?? 50.0, isLocked: isLocked, rawPSL: nil),
                MetricCard(title: "FOREHEAD PROP...", score: results.upperThird.foreheadProjection ?? 50.0, isLocked: isLocked, rawPSL: nil),
                MetricCard(title: "HAIRLINE RECES...", score: results.upperThird.hairlineRecession ?? 50.0, isLocked: isLocked, rawPSL: nil),
                MetricCard(title: "HAIR THINNING", score: results.upperThird.hairThinning ?? 50.0, isLocked: isLocked, rawPSL: nil),
                MetricCard(title: "HAIRLINE DENSITY", score: results.upperThird.hairlineDensity ?? 50.0, isLocked: isLocked, rawPSL: nil),
                MetricCard(title: "FOREHEAD SLOPE", score: results.upperThird.foreheadSlope ?? 50.0, isLocked: isLocked, rawPSL: nil)
            ]
        case .miscellaneous:
            return [
                MetricCard(title: "SKIN", score: results.miscellaneous.skin ?? 50.0, isLocked: isLocked, rawPSL: nil),
                MetricCard(title: "HARMONY", score: results.miscellaneous.harmony ?? 50.0, isLocked: isLocked, rawPSL: nil),
                MetricCard(title: "SYMMETRY", score: results.miscellaneous.symmetry ?? 50.0, isLocked: isLocked, rawPSL: nil),
                MetricCard(title: "NECK WIDTH", score: results.miscellaneous.neckWidth ?? 50.0, isLocked: isLocked, rawPSL: nil),
                MetricCard(title: "BLOAT", score: results.miscellaneous.bloat ?? 50.0, isLocked: isLocked, rawPSL: nil),
                MetricCard(title: "BONE MASS", score: results.miscellaneous.boneMass ?? 50.0, isLocked: isLocked, rawPSL: nil)
            ]
        }
    }
}

struct MetricCardView: View {
    let metric: MetricCard
    @StateObject private var scaleManager = ScaleManager.shared
    
    var body: some View {
        VStack(alignment: .leading, spacing: 8) {
            HStack {
                Text(metric.title)
                    .font(.system(size: 12, weight: .medium))
                    .foregroundColor(.cyan)
                    .lineLimit(1)
                
                Spacer()
                
                if metric.isLocked {
                    Image(systemName: "lock.fill")
                        .font(.system(size: 10))
                        .foregroundColor(.white.opacity(0.6))
                }
            }
            
            // Score display (blurred if locked)
            // For PSL/POTENTIAL, use scale conversion; for other metrics, use raw score
            ZStack {
                if let rawPSL = metric.rawPSL {
                    // PSL/POTENTIAL: Use scale conversion
                    let display = scaleManager.getFullDisplay(rawPSL)
                    VStack(alignment: .leading, spacing: 2) {
                        Text(display.score)
                            .font(.system(size: 24, weight: .bold))
                            .foregroundColor(.white)
                        Text(display.label)
                            .font(.system(size: 10, weight: .medium))
                            .foregroundColor(.cyan.opacity(0.8))
                            .lineLimit(1)
                    }
                    .blur(radius: metric.isLocked ? 8 : 0)
                } else {
                    // Other metrics: Show raw score (0-100)
                    Text(String(format: "%.0f", metric.score))
                        .font(.system(size: 24, weight: .bold))
                        .foregroundColor(.white)
                        .blur(radius: metric.isLocked ? 8 : 0)
                }
            }
            .frame(height: 40)
            
            // Progress bar
            GeometryReader { geometry in
                ZStack(alignment: .leading) {
                    Rectangle()
                        .fill(Color.gray.opacity(0.3))
                        .frame(height: 4)
                    
                    Rectangle()
                        .fill(
                            LinearGradient(
                                colors: [Color.yellow, Color.green],
                                startPoint: .leading,
                                endPoint: .trailing
                            )
                        )
                        .frame(
                            width: geometry.size.width * CGFloat(
                                metric.rawPSL != nil 
                                    ? scaleManager.getProgressValue(metric.rawPSL!) / scaleManager.getScaleRange().max
                                    : metric.score / 100
                            ),
                            height: 4
                        )
                }
            }
            .frame(height: 4)
        }
        .padding(12)
        .background(
            RoundedRectangle(cornerRadius: 12)
                .fill(Color.white.opacity(0.1))
                .overlay(
                    RoundedRectangle(cornerRadius: 12)
                        .stroke(Color.white.opacity(0.2), lineWidth: 1)
                )
        )
        .frame(height: 140)
    }
}

// MARK: - Unlock Sheet
struct UnlockSheet: View {
    @Binding var isPresented: Bool
    @StateObject private var subscriptionManager = SubscriptionManager.shared
    
    var body: some View {
        NavigationView {
            VStack(spacing: 30) {
                Image(systemName: "lock.open.fill")
                    .font(.system(size: 60))
                    .foregroundColor(.orange)
                
                Text("Unlock All Results")
                    .font(.title)
                    .fontWeight(.bold)
                
                Text("Get access to all your detailed facial analysis metrics and personalized recommendations.")
                    .font(.body)
                    .multilineTextAlignment(.center)
                    .foregroundColor(.gray)
                    .padding(.horizontal)
                
                VStack(spacing: 15) {
                    Button(action: {
                        subscriptionManager.unlockResults()
                        isPresented = false
                    }) {
                        Text("Subscribe - $3.99/month")
                            .font(.headline)
                            .foregroundColor(.white)
                            .frame(maxWidth: .infinity)
                            .padding()
                            .background(Color.blue)
                            .cornerRadius(12)
                    }
                    
                    Button(action: {
                        isPresented = false
                    }) {
                        Text("Cancel")
                            .font(.headline)
                            .foregroundColor(.gray)
                            .frame(maxWidth: .infinity)
                            .padding()
                            .background(Color.gray.opacity(0.2))
                            .cornerRadius(12)
                    }
                }
                .padding(.horizontal)
                
                Spacer()
            }
            .padding()
            .navigationTitle("Unlock Results")
            .navigationBarTitleDisplayMode(.inline)
        }
    }
}

// MARK: - Date Picker Sheet
struct DatePickerSheet: View {
    @Binding var selectedDate: Date
    @Environment(\.dismiss) var dismiss
    
    var body: some View {
        NavigationView {
            VStack {
                DatePicker("Ascension Date", selection: $selectedDate, displayedComponents: .date)
                    .datePickerStyle(.graphical)
                    .padding()
                
                Spacer()
            }
            .navigationTitle("Select Date")
            .navigationBarTitleDisplayMode(.inline)
            .toolbar {
                ToolbarItem(placement: .navigationBarTrailing) {
                    Button("Done") {
                        dismiss()
                    }
                }
            }
        }
    }
}

