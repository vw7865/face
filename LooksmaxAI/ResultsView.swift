//
//  ResultsView.swift
//  LooksmaxAI
//
//  Created by Admin on 2026. 01. 07..
//

import SwiftUI
import Photos
import UIKit

struct ResultsView: View {
    let results: FaceAnalysisResults
    let onClose: (() -> Void)?
    let gender: String? // Optional gender for gender-specific labels
    let thumbnailPath: String? // Optional thumbnail path for front image
    let thumbnailImage: UIImage? // Optional thumbnail image for instant display
    @StateObject private var subscriptionManager = SubscriptionManager.shared
    @StateObject private var scaleManager = ScaleManager.shared
    @State private var currentSection: ResultsSection = .overall
    @State private var showUnlockSheet = false
    @State private var frontImage: UIImage? = nil
    @State private var showShareSheet = false
    @State private var shareImage: UIImage? = nil
    @State private var showSaveConfirmation = false
    
    // Internal resolved state to handle nil values on first render
    @State private var resolvedGender: String? = nil
    @State private var resolvedThumbnailPath: String? = nil
    
    @Environment(\.dismiss) var dismiss
    
    // Detect if device is iPhone (not iPad)
    private var isIPhone: Bool {
        UIDevice.current.userInterfaceIdiom == .phone
    }
    
    init(results: FaceAnalysisResults, onClose: (() -> Void)? = nil, gender: String? = nil, thumbnailPath: String? = nil, thumbnailImage: UIImage? = nil) {
        self.results = results
        self.onClose = onClose
        self.gender = gender
        self.thumbnailPath = thumbnailPath
        self.thumbnailImage = thumbnailImage
        
        // Initialize frontImage with thumbnailImage for instant display
        _frontImage = State(initialValue: thumbnailImage)
        
        // Initialize resolvedGender with gender for instant display
        _resolvedGender = State(initialValue: gender)
        _resolvedThumbnailPath = State(initialValue: thumbnailPath)
    }
    
    var body: some View {
        NavigationStack {
            ZStack {
                Color.black.ignoresSafeArea()
                
                VStack(spacing: 0) {
                    // Header
                    VStack(spacing: isIPhone ? 4 : 8) {
                        // Section Title
                        Text(getSectionTitle())
                            .font(.system(size: isIPhone ? 22 : 28, weight: .semibold))
                            .foregroundColor(.white)
                            .padding(.top, isIPhone ? 12 : 20)
                        
                        // Percentile ranking (only show on Overall section)
                        if currentSection == .overall {
                            Text(getPercentileText())
                                .font(.system(size: isIPhone ? 11 : 14, weight: .medium))
                                .foregroundColor(.white.opacity(0.7))
                        }
                    }
                    .padding(.top, isIPhone ? 4 : 10)
                
                // Metrics Grid
                if currentSection == .overall {
                    // Overall section - optimized layout for sharing
                    VStack(spacing: 0) {
                        // Top section: Image and Overall/Potential side by side
                        HStack(spacing: isIPhone ? 8 : 12) {
                            // Larger profile image on the left
                            if let frontImage = frontImage {
                                Image(uiImage: frontImage)
                                    .resizable()
                                    .aspectRatio(contentMode: .fill)
                                    .frame(width: isIPhone ? 120 : 180, height: isIPhone ? 120 : 180)
                                    .clipShape(RoundedRectangle(cornerRadius: isIPhone ? 12 : 16))
                                    .overlay(
                                        RoundedRectangle(cornerRadius: isIPhone ? 12 : 16)
                                            .stroke(Color.white.opacity(0.3), lineWidth: 2)
                                    )
                                    .shadow(color: Color.black.opacity(0.4), radius: 10, x: 0, y: 5)
                            }
                            
                            // Overall and Potential cards stacked on the right
                            VStack(spacing: isIPhone ? 6 : 10) {
                                ForEach(getOverallMetrics(), id: \.id) { metric in
                                    MetricCardView(metric: metric, gender: resolvedGender, isCompact: true, isIPhone: isIPhone)
                                }
                            }
                        }
                        .padding(.horizontal, isIPhone ? 12 : 16)
                        .padding(.top, isIPhone ? 6 : 10)
                        
                        // Key metrics - 2x2 grid below
                        LazyVGrid(columns: [
                            GridItem(.flexible(), spacing: isIPhone ? 6 : 10),
                            GridItem(.flexible(), spacing: isIPhone ? 6 : 10)
                        ], spacing: isIPhone ? 6 : 10) {
                            ForEach(getKeyMetrics(), id: \.id) { metric in
                                MetricCardView(metric: metric, gender: resolvedGender, isSmall: true, isIPhone: isIPhone)
                            }
                        }
                        .padding(.horizontal, isIPhone ? 12 : 16)
                        .padding(.top, isIPhone ? 8 : 12)
                    }
                } else {
                    // Other sections - use ScrollView
                    ScrollView {
                        VStack(spacing: 20) {
                            // For other sections, use 2-column grid
                            LazyVGrid(columns: [
                                GridItem(.flexible(), spacing: 12),
                                GridItem(.flexible(), spacing: 12)
                            ], spacing: 12) {
                                ForEach(getMetricsForSection(currentSection), id: \.id) { metric in
                                    MetricCardView(metric: metric, gender: resolvedGender)
                                }
                            }
                        }
                        .padding(.horizontal, 16)
                        .padding(.top, 20)
                    }
                }
                
                // Pagination Dots
                HStack(spacing: 8) {
                    ForEach(ResultsSection.allCases, id: \.self) { section in
                        Circle()
                            .fill(section == currentSection ? Color.white : Color.gray.opacity(0.3))
                            .frame(width: isIPhone ? 6 : 8, height: isIPhone ? 6 : 8)
                    }
                }
                .padding(.vertical, isIPhone ? 8 : 16)
                
                // Unlock button removed - all results are free
            }
            }
            .navigationBarTitleDisplayMode(.inline)
            .toolbar {
                // Close button - always visible on leading side
                ToolbarItem(placement: .navigationBarLeading) {
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
                
                // Scale toggle button - always visible
                ToolbarItem(placement: .navigationBarTrailing) {
                    Button(action: {
                        // Toggle between PSL and Objective scales
                        scaleManager.selectedScale = scaleManager.selectedScale == .psl ? .objective : .psl
                    }) {
                        HStack(spacing: 4) {
                            Image(systemName: "arrow.left.arrow.right")
                                .font(.system(size: 12, weight: .medium))
                            Text(scaleManager.selectedScale == .psl ? "PSL" : "1-10")
                                .font(.system(size: 12, weight: .semibold))
                        }
                        .foregroundColor(.white)
                        .padding(.horizontal, 8)
                        .padding(.vertical, 6)
                        .background(
                            RoundedRectangle(cornerRadius: 8)
                                .fill(Color.white.opacity(0.15))
                                .overlay(
                                    RoundedRectangle(cornerRadius: 8)
                                        .stroke(Color.white.opacity(0.4), lineWidth: 1)
                                )
                        )
                    }
                }
                
                // Share/Save button (only show on Overall section) - separate item
                if currentSection == .overall {
                    ToolbarItem(placement: .navigationBarTrailing) {
                        Menu {
                            Button(action: {
                                captureAndSaveOverallScreen()
                            }) {
                                Label("Save to Photos", systemImage: "square.and.arrow.down")
                            }
                            
                            Button(action: {
                                captureAndShareOverallScreen()
                            }) {
                                Label("Share", systemImage: "square.and.arrow.up")
                            }
                        } label: {
                            Image(systemName: "square.and.arrow.up")
                                .font(.system(size: 20))
                                .foregroundColor(.white.opacity(0.8))
                        }
                    }
                }
            }
        }
        .onAppear {
            // Resolve gender and thumbnail path on appear
            print("📋 ResultsView appeared - gender=\(gender ?? "nil"), thumbnailPath=\(thumbnailPath ?? "nil"), thumbnailImage=\(thumbnailImage != nil ? "provided" : "nil"), resolvedGender=\(resolvedGender ?? "nil")")
            
            // Update resolved values (they're already set in init, but update in case they changed)
            resolvedGender = gender ?? resolvedGender
            resolvedThumbnailPath = thumbnailPath ?? resolvedThumbnailPath
            
            // If we already have the image from init, we're done
            // Otherwise, try to load from disk
            if frontImage == nil, let path = resolvedThumbnailPath {
                print("📸 ResultsView: No image from init, loading from disk...")
                loadFrontImage(from: path)
            } else if frontImage != nil {
                print("📸 ResultsView: Image already loaded from init")
            }
        }
        .onChange(of: gender) { newGender in
            resolvedGender = newGender
            print("📋 ResultsView gender updated: \(newGender ?? "nil")")
        }
        .onChange(of: thumbnailPath) { newPath in
            resolvedThumbnailPath = newPath
            print("📋 ResultsView thumbnailPath updated: \(newPath ?? "nil")")
            if let path = newPath {
                loadFrontImage(from: path)
            }
        }
        .onChange(of: currentSection) { newSection in
            // Reload image when switching to Overall section, especially if it's nil
            if newSection == .overall && frontImage == nil {
                if let path = resolvedThumbnailPath {
                    loadFrontImage(from: path)
                }
            }
        }
        .sheet(isPresented: Binding(
            get: { showShareSheet && shareImage != nil },
            set: { showShareSheet = $0 }
        )) {
            if let shareImage = shareImage, shareImage.size.width > 0, shareImage.size.height > 0 {
                ShareSheet(activityItems: [shareImage])
                    .onAppear {
                        print("✅ ShareSheet appeared with valid image: \(shareImage.size)")
                    }
            } else {
                ShareSheet(activityItems: [])
                    .onAppear {
                        print("⚠️ ShareSheet appeared but shareImage is nil or invalid")
                    }
            }
        }
        .alert("Saved!", isPresented: $showSaveConfirmation) {
            Button("OK", role: .cancel) { }
        } message: {
            Text("Your result has been saved to Photos")
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
    
    private func loadFrontImage(from path: String) {
        // Debug: Check if file exists
        let fileManager = FileManager.default
        let documentsPath = fileManager.urls(for: .documentDirectory, in: .userDomainMask)[0]
        let thumbnailsPath = documentsPath.appendingPathComponent("ResultThumbnails")
        let fileURL = thumbnailsPath.appendingPathComponent(path)
        let fileExists = fileManager.fileExists(atPath: fileURL.path)
        print("📸 ResultsView: Loading from path: \(path), file exists: \(fileExists)")
        
        // Try synchronous load first (fast path)
        if let image = ResultHistoryManager.shared.loadThumbnail(for: path) {
            print("📸 ResultsView: Image loaded synchronously")
            frontImage = image
            return
        }
        
        // If failed, try async with retry loop
        print("📸 ResultsView: Synchronous load failed, trying async with retry...")
        Task {
            await loadFrontImageAsync(from: path)
        }
    }
    
    private func loadFrontImageAsync(from path: String) async {
        print("📸 ResultsView: Loading image from path: \(path)")
        
        // Retry loop with exponential backoff (up to 5 attempts)
        for attempt in 0..<5 {
            if let image = ResultHistoryManager.shared.loadThumbnail(for: path) {
                print("📸 ResultsView: Image loaded successfully on attempt \(attempt + 1)")
                await MainActor.run {
                    self.frontImage = image
                }
                return
            }
            
            // Exponential backoff: 0.2s, 0.4s, 0.6s, 0.8s, 1.0s
            let delay = UInt64(200_000_000 * (attempt + 1))
            print("📸 ResultsView: Attempt \(attempt + 1) failed, retrying after \(delay / 1_000_000)ms...")
            try? await Task.sleep(nanoseconds: delay)
        }
        
        print("📸 ResultsView: Image load failed after all retry attempts")
    }
    
    private func getSectionTitle() -> String {
        if currentSection == .overall {
            // Get the user's rating label based on their PSL score
            let psl = results.overall.psl ?? 50.0
            let display = scaleManager.getFullDisplay(psl, gender: resolvedGender)
            // Format as "You're a [label]" - capitalize first letter
            let label = display.label
            let capitalizedLabel = label.prefix(1).uppercased() + label.dropFirst()
            return "You're a \(capitalizedLabel)"
        } else {
            return currentSection.rawValue
        }
    }
    
    private func getPercentileText() -> String {
        guard currentSection == .overall else { return "" }
        
        let psl = results.overall.psl ?? 50.0
        let genderText = (resolvedGender?.lowercased() == "male" || resolvedGender?.lowercased() == "man") ? "men" : "women"
        
        switch scaleManager.selectedScale {
        case .psl:
            let pslScale = scaleManager.convertToPSLScale(psl)
            
            if pslScale >= 7.0 {
                return "Top ~0.01% of \(genderText)"
            } else if pslScale >= 6.5 {
                return "Top ~0.1–0.5% of \(genderText)"
            } else if pslScale >= 6.0 {
                return "Top ~1–3% of \(genderText)"
            } else if pslScale >= 5.5 {
                return "Top ~5–10% of \(genderText)"
            } else if pslScale >= 4.5 {
                return "Top ~35–45% of \(genderText)"
            } else if pslScale >= 3.0 {
                return "Solid foundation with room to optimize"
            } else {
                return "Developing facial structure"
            }
            
        case .objective:
            let objectiveScale = scaleManager.convertToObjectiveScale(psl)
            
            if objectiveScale >= 9.0 {
                return "Top ~0.1–1% of \(genderText)"
            } else if objectiveScale >= 8.0 {
                return "Top ~1–5% of \(genderText)"
            } else if objectiveScale >= 7.0 {
                return "Top ~5–15% of \(genderText)"
            } else if objectiveScale >= 6.0 {
                return "Top ~15–30% of \(genderText)"
            } else if objectiveScale >= 4.0 {
                return "Solid foundation with room to optimize"
            } else {
                return "Developing facial structure"
            }
        }
    }
    
    private func captureOverallScreen() -> UIImage? {
        // Ensure we're on the Overall section
        guard currentSection == .overall else {
            print("❌ captureOverallScreen: Not on overall section, currentSection = \(currentSection)")
            return nil
        }
        
        print("📸 captureOverallScreen: Starting capture...")
        
        // Create a view for the Overall section
        let overallView = ZStack {
            Color.black
            
            VStack(spacing: 0) {
                // Title
                Text(getSectionTitle())
                    .font(.system(size: 28, weight: .semibold))
                    .foregroundColor(.white)
                    .padding(.top, 80)  // Increased from 40 to 80 to position below notch
                
                // Percentile ranking
                Text(getPercentileText())
                    .font(.system(size: 14, weight: .medium))
                    .foregroundColor(.white.opacity(0.7))
                    .padding(.bottom, 10)
                
                // Top section: Image and Overall/Potential side by side
                HStack(spacing: 12) {
                    // Larger profile image on the left
                    if let frontImage = frontImage {
                        Image(uiImage: frontImage)
                            .resizable()
                            .aspectRatio(contentMode: .fill)
                            .frame(width: 180, height: 180)
                            .clipShape(RoundedRectangle(cornerRadius: 16))
                            .overlay(
                                RoundedRectangle(cornerRadius: 16)
                                    .stroke(Color.white.opacity(0.3), lineWidth: 2)
                            )
                            .shadow(color: Color.black.opacity(0.4), radius: 10, x: 0, y: 5)
                    }
                    
                    // Overall and Potential cards stacked on the right
                    VStack(spacing: 10) {
                        ForEach(getOverallMetrics(), id: \.id) { metric in
                            MetricCardView(metric: metric, gender: resolvedGender, isCompact: true)
                        }
                    }
                }
                .padding(.horizontal, 16)
                .padding(.top, 10)
                
                // Key metrics - 2x2 grid below
                LazyVGrid(columns: [
                    GridItem(.flexible(), spacing: 10),
                    GridItem(.flexible(), spacing: 10)
                ], spacing: 10) {
                    ForEach(getKeyMetrics(), id: \.id) { metric in
                        MetricCardView(metric: metric, gender: resolvedGender, isSmall: true)
                    }
                }
                .padding(.horizontal, 16)
                .padding(.top, 12)
            }
        }
        .frame(width: UIScreen.main.bounds.width, height: UIScreen.main.bounds.height)
        .background(Color.black)
        
        // Use ImageRenderer on iOS 16+ for better SwiftUI rendering
        if #available(iOS 16.0, *) {
            let renderer = ImageRenderer(content: overallView)
            renderer.scale = UIScreen.main.scale
            renderer.proposedSize = .init(width: UIScreen.main.bounds.width, height: UIScreen.main.bounds.height)
            
            if let image = renderer.uiImage {
                // Convert to opaque image to avoid alpha channel warning
                let opaqueImage = makeOpaque(image: image)
                print("📸 captureOverallScreen: Image captured with ImageRenderer, size = \(opaqueImage.size)")
                return opaqueImage
            }
        }
        
        // Fallback: Use UIView snapshot method
        let hostingController = UIHostingController(rootView: overallView)
        hostingController.view.backgroundColor = .black
        let targetSize = CGSize(width: UIScreen.main.bounds.width, height: UIScreen.main.bounds.height)
        hostingController.view.frame = CGRect(origin: .zero, size: targetSize)
        
        // Add to window temporarily - make it visible but off-screen
        var containerView: UIView?
        if let windowScene = UIApplication.shared.connectedScenes.first as? UIWindowScene,
           let window = windowScene.windows.first {
            containerView = UIView(frame: CGRect(x: -targetSize.width, y: 0, width: targetSize.width, height: targetSize.height))
            containerView?.addSubview(hostingController.view)
            window.addSubview(containerView!)
        }
        
        // Force layout
        hostingController.view.setNeedsLayout()
        hostingController.view.layoutIfNeeded()
        
        // Render immediately - the view should be laid out
        // Use opaque format since background is black (no transparency needed)
        let format = UIGraphicsImageRendererFormat()
        format.scale = UIScreen.main.scale
        format.opaque = true  // Set to true since we have a black opaque background
        let renderer = UIGraphicsImageRenderer(size: targetSize, format: format)
        
        let image = renderer.image { context in
            hostingController.view.drawHierarchy(in: hostingController.view.bounds, afterScreenUpdates: true)
        }
        
        // Clean up
        containerView?.removeFromSuperview()
        
        // Convert to opaque image to avoid alpha channel warning
        let opaqueImage = makeOpaque(image: image)
        print("📸 captureOverallScreen: Image captured with UIView snapshot, size = \(opaqueImage.size)")
        return opaqueImage
    }
    
    private func makeOpaque(image: UIImage) -> UIImage {
        // Convert image to opaque format to avoid alpha channel warning
        let size = image.size
        let scale = image.scale
        
        let format = UIGraphicsImageRendererFormat()
        format.scale = scale
        format.opaque = true
        
        let renderer = UIGraphicsImageRenderer(size: size, format: format)
        return renderer.image { context in
            // Fill with black background first
            context.cgContext.setFillColor(UIColor.black.cgColor)
            context.cgContext.fill(CGRect(origin: .zero, size: size))
            // Then draw the image on top
            image.draw(at: .zero)
        }
    }
    
    private func captureAndSaveOverallScreen() {
        print("💾 Starting save process...")
        
        // Add a delay to ensure UI is ready and rendered
        DispatchQueue.main.asyncAfter(deadline: .now() + 0.5) {
            print("💾 Capturing image for save...")
            guard let image = self.captureOverallScreen(), image.size.width > 0, image.size.height > 0 else {
                print("❌ Failed to capture image for saving")
                return
            }
            
            print("💾 Image captured successfully, size: \(image.size)")
            
            print("💾 Image captured, requesting photo library access...")
            // Request photo library access
            PHPhotoLibrary.requestAuthorization { status in
                guard status == .authorized else {
                    print("❌ Photo library access denied")
                    return
                }
                
                print("💾 Saving image to photo library...")
                // Save to photo library
                PHPhotoLibrary.shared().performChanges({
                    PHAssetChangeRequest.creationRequestForAsset(from: image)
                }) { success, error in
                    DispatchQueue.main.async {
                        if let error = error {
                            print("❌ Error saving image: \(error.localizedDescription)")
                        } else if success {
                            print("✅ Image saved successfully")
                            self.showSaveConfirmation = true
                        }
                    }
                }
            }
        }
    }
    
    private func captureAndShareOverallScreen() {
        print("📸 Starting image capture for sharing...")
        print("   currentSection: \(currentSection)")
        print("   frontImage is nil: \(frontImage == nil)")
        
        // Add a delay to ensure UI is ready and rendered
        DispatchQueue.main.asyncAfter(deadline: .now() + 0.5) {
            print("📸 Attempting to capture image...")
            guard let image = self.captureOverallScreen() else {
                print("❌ Failed to capture image - captureOverallScreen returned nil")
                return
            }
            
            guard image.size.width > 0, image.size.height > 0 else {
                print("❌ Captured image has invalid size: \(image.size)")
                return
            }
            
            print("✅ Image captured successfully: \(image.size)")
            self.shareImage = image
            print("✅ shareImage set to: \(image.size)")
            
            // Ensure state is updated before showing sheet
            DispatchQueue.main.async {
                print("✅ Showing share sheet...")
                self.showShareSheet = true
            }
        }
    }
    
    private func getOverallMetrics() -> [MetricCard] {
        // Overall and Potential metrics (large cards)
        let isLocked = false
        let psl = results.overall.psl ?? 50.0
        let potential = results.overall.potential ?? 50.0
        
        // Get display values based on selected scale
        let pslDisplay = scaleManager.getProgressValue(psl)
        let potentialDisplay = scaleManager.getProgressValue(potential)
        
        // Determine PSL title based on selected scale
        let pslTitle = scaleManager.selectedScale == .psl ? "Overall (PSL Scale 0-8)" : "Overall (1-10 Scale)"
        
        return [
            MetricCard(title: pslTitle, score: pslDisplay, isLocked: isLocked, rawPSL: psl),
            MetricCard(title: "POTENTIAL", score: potentialDisplay, isLocked: isLocked, rawPSL: potential)
        ]
    }
    
    private func getKeyMetrics() -> [MetricCard] {
        // Key metrics for Overall section (smaller cards)
        let isLocked = false
        
        // Get raw scores
        let rawJawline = results.lowerThird.jawWidth ?? 50.0
        let rawCheekbones = results.midface.cheekbones ?? 50.0
        
        // Calculate Eye Area as average of eye metrics
        let orbitalDepth = results.eyes.orbitalDepth ?? 50.0
        let canthalTilt = results.eyes.canthalTilt ?? 50.0
        let eyelidExposure = results.eyes.eyelidExposure ?? 50.0
        let rawEyeArea = (orbitalDepth + canthalTilt + eyelidExposure) / 3.0
        
        let rawHarmony = results.miscellaneous.harmony ?? 50.0
        
        // Get overall PSL (0-100 from backend)
        let overallPSL = results.overall.psl ?? 50.0
        
        // Check if we should apply boost/floor: overall_PSL >= 6.5 (PSL scale) OR > 8 (objective scale)
        let pslScale = scaleManager.convertToPSLScale(overallPSL)  // Convert to 0-8 scale
        let objectiveScale = scaleManager.convertToObjectiveScale(overallPSL)  // Convert to 0-10 scale
        let shouldApplyBoost = pslScale >= 6.5 || objectiveScale > 8.0
        
        // Apply boost/floor logic only for Chadlite+ territory
        let (displayedJawline, displayedCheekbones, displayedEyeArea, displayedHarmony): (Double, Double, Double, Double)
        
        if shouldApplyBoost {
            // Calculate dynamic floors based on PSL scale (0-8 range)
            // Jawline & Cheekbones floor = 70 + (pslScale - 6.5) × 10
            // Eye Area & Harmony floor = 75 + (pslScale - 6.5) × 8
            let jawCheekFloor = 70.0 + (pslScale - 6.5) * 10.0
            let eyeHarmonyFloor = 75.0 + (pslScale - 6.5) * 8.0
            
            // Apply max() to each sub-score, but only if raw is not already very high (85+)
            // If raw is already 85+, let it stay raw (no floor needed)
            let jawlineScore = rawJawline >= 85.0 ? rawJawline : max(rawJawline, jawCheekFloor)
            let cheekbonesScore = rawCheekbones >= 85.0 ? rawCheekbones : max(rawCheekbones, jawCheekFloor)
            let eyeAreaScore = rawEyeArea >= 85.0 ? rawEyeArea : max(rawEyeArea, eyeHarmonyFloor)
            let harmonyScore = rawHarmony >= 85.0 ? rawHarmony : max(rawHarmony, eyeHarmonyFloor)
            
            // Cap at 95 and round to nearest whole number
            displayedJawline = min(95.0, round(jawlineScore))
            displayedCheekbones = min(95.0, round(cheekbonesScore))
            displayedEyeArea = min(95.0, round(eyeAreaScore))
            displayedHarmony = min(95.0, round(harmonyScore))
        } else {
            // For everyone below Chadlite, use raw scores (blackpill honesty for normies)
            displayedJawline = round(rawJawline)
            displayedCheekbones = round(rawCheekbones)
            displayedEyeArea = round(rawEyeArea)
            displayedHarmony = round(rawHarmony)
        }
        
        return [
            MetricCard(title: "JAWLINE", score: displayedJawline, isLocked: isLocked, rawPSL: nil),
            MetricCard(title: "CHEEKBONES", score: displayedCheekbones, isLocked: isLocked, rawPSL: nil),
            MetricCard(title: "EYE AREA", score: displayedEyeArea, isLocked: isLocked, rawPSL: nil),
            MetricCard(title: "HARMONY", score: displayedHarmony, isLocked: isLocked, rawPSL: nil)
        ]
    }
    
    private func getMetricsForSection(_ section: ResultsSection) -> [MetricCard] {
        // All results are free - no locking needed
        let isLocked = false
        
        switch section {
        case .overall:
            // This should not be called for overall section anymore
            // Use getOverallMetrics() and getKeyMetrics() instead
            return []
        case .eyes:
            return [
                MetricCard(title: "ORBITAL DEPTH", score: results.eyes.orbitalDepth ?? 50.0, isLocked: isLocked, rawPSL: nil),
                MetricCard(title: "CANTHAL TILT", score: results.eyes.canthalTilt ?? 50.0, isLocked: isLocked, rawPSL: nil),
                MetricCard(title: "EYEBROW DENSITY", score: results.eyes.eyebrowDensity ?? 50.0, isLocked: isLocked, rawPSL: nil),
                MetricCard(title: "EYELASH DENSITY", score: results.eyes.eyelashDensity ?? 50.0, isLocked: isLocked, rawPSL: nil),
                MetricCard(title: "EYELID EXPOSURE", score: results.eyes.eyelidExposure ?? 50.0, isLocked: isLocked, rawPSL: nil),
                MetricCard(title: "UNDER EYE HEALTH", score: results.eyes.underEyeHealth ?? 50.0, isLocked: isLocked, rawPSL: nil)
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
                MetricCard(title: "HYOID SKIN TIGHTNESS", score: results.lowerThird.hyoidSkinTightness ?? 50.0, isLocked: isLocked, rawPSL: nil),
                MetricCard(title: "JAW WIDTH", score: results.lowerThird.jawWidth ?? 50.0, isLocked: isLocked, rawPSL: nil)
            ]
        case .upperThird:
            return [
                MetricCard(title: "NORWOOD STAGE", score: results.upperThird.norwoodStage ?? 50.0, isLocked: isLocked, rawPSL: nil),
                MetricCard(title: "FOREHEAD PROJECTION", score: results.upperThird.foreheadProjection ?? 50.0, isLocked: isLocked, rawPSL: nil),
                MetricCard(title: "HAIRLINE RECESSION", score: results.upperThird.hairlineRecession ?? 50.0, isLocked: isLocked, rawPSL: nil),
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
    let gender: String?
    let isSmall: Bool
    let isMedium: Bool
    let isCompact: Bool
    let isIPhone: Bool
    @StateObject private var scaleManager = ScaleManager.shared
    
    init(metric: MetricCard, gender: String?, isSmall: Bool = false, isMedium: Bool = false, isCompact: Bool = false, isIPhone: Bool = false) {
        self.metric = metric
        self.gender = gender
        self.isSmall = isSmall
        self.isMedium = isMedium
        self.isCompact = isCompact
        self.isIPhone = isIPhone
    }
    
    // Check if this is a PSL or Potential card (overall section cards)
    private var isOverallCard: Bool {
        (metric.title.contains("Overall") || metric.title == "POTENTIAL") && !isSmall && !isMedium && !isCompact
    }
    
    // Determine card size based on isOverallCard, isMedium, isCompact, and isSmall
    private var cardSize: (padding: CGFloat, height: CGFloat, fontSize: CGFloat, scoreSize: CGFloat, labelSize: CGFloat, cornerRadius: CGFloat) {
        if isOverallCard {
            return (16, 180, 14, 36, 12, 16)
        } else if isCompact {
            // Reduce sizes for iPhone
            if isIPhone {
                return (8, 70, 10, 20, 9, 10)
            }
            return (10, 85, 11, 24, 10, 12)
        } else if isMedium {
            return (12, 140, 12, 28, 11, 12)
        } else if isSmall {
            // Reduce sizes for iPhone
            if isIPhone {
                return (8, 80, 9, 18, 8, 8)
            }
            return (10, 100, 10, 20, 9, 10)
        } else {
            return (12, 140, 11, 24, 10, 12)
        }
    }
    
    var body: some View {
        let size = cardSize
        let spacing: CGFloat = isIPhone ? (isSmall ? 4 : (isCompact ? 4 : 6)) : (isOverallCard ? 12 : (isSmall ? 6 : (isCompact ? 6 : (isMedium ? 10 : 8))))
        VStack(alignment: .leading, spacing: spacing) {
            HStack {
                Text(metric.title)
                    .font(.system(size: size.fontSize, weight: (isOverallCard || isMedium || isCompact) ? .semibold : .medium))
                    .foregroundColor(.white)
                    .lineLimit(2)
                    .minimumScaleFactor(0.8)
                    .fixedSize(horizontal: false, vertical: true)
                
                Spacer()
                
                if metric.isLocked {
                    Image(systemName: "lock.fill")
                        .font(.system(size: isSmall ? 8 : 10))
                        .foregroundColor(.white.opacity(0.6))
                }
            }
            
            // Score display (blurred if locked)
            // For PSL/POTENTIAL, use scale conversion; for other metrics, use raw score
            ZStack {
                if let rawPSL = metric.rawPSL {
                    // PSL/POTENTIAL: Use scale conversion
                    let display = scaleManager.getFullDisplay(rawPSL, gender: gender)
                    VStack(alignment: .leading, spacing: isIPhone ? (isSmall ? 1 : (isCompact ? 1 : 2)) : (isOverallCard ? 4 : (isSmall ? 2 : (isCompact ? 2 : (isMedium ? 3 : 2))))) {
                        Text(display.score)
                            .font(.system(size: size.scoreSize, weight: .bold))
                            .foregroundColor(.white)
                        if !isSmall {
                            Text(display.label)
                                .font(.system(size: size.labelSize, weight: .medium))
                                .foregroundColor(.white)
                                .lineLimit(1)
                        }
                    }
                    .blur(radius: metric.isLocked ? 8 : 0)
                } else {
                    // Other metrics: Show raw score (0-100)
                    Text(String(format: "%.0f", metric.score))
                        .font(.system(size: size.scoreSize, weight: .bold))
                        .foregroundColor(.white)
                        .blur(radius: metric.isLocked ? 8 : 0)
                }
            }
            .frame(height: isIPhone ? (isSmall ? 24 : (isCompact ? 32 : 50)) : (isSmall ? 30 : (isOverallCard ? 60 : (isCompact ? 40 : (isMedium ? 50 : 40)))))
            
            // Progress bar
            if !isSmall {
                GeometryReader { geometry in
                    ZStack(alignment: .leading) {
                        Rectangle()
                            .fill(Color.gray.opacity(0.3))
                            .frame(height: (isOverallCard || isMedium || isCompact) ? 4 : 4)
                        
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
                                height: (isOverallCard || isMedium || isCompact) ? 4 : 4
                            )
                    }
                }
                .frame(height: (isOverallCard || isMedium || isCompact) ? 4 : 4)
            }
        }
        .padding(size.padding)
        .background(
            RoundedRectangle(cornerRadius: size.cornerRadius)
                .fill((isOverallCard || isMedium || isCompact) ? Color.white.opacity(0.15) : Color.white.opacity(0.1))
                .overlay(
                    RoundedRectangle(cornerRadius: size.cornerRadius)
                        .stroke(Color.white.opacity(0.2), lineWidth: (isOverallCard || isMedium || isCompact) ? 1.5 : 1)
                )
        )
        .frame(height: size.height)
    }
}

// MARK: - Unlock Sheet (Deprecated - all results are free)
// This view is no longer used since all results are free
// Keeping for reference but not used in the app
struct UnlockSheet: View {
    @Binding var isPresented: Bool
    @StateObject private var subscriptionManager = SubscriptionManager.shared
    @State private var showUpgrade = false
    
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
                        showUpgrade = true
                    }) {
                        Text("Upgrade to Pro")
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
            .fullScreenCover(isPresented: $showUpgrade) {
                UpgradeView()
            }
        }
    }
}

// MARK: - ShareSheet
struct ShareSheet: UIViewControllerRepresentable {
    let activityItems: [Any]
    
    func makeUIViewController(context: Context) -> UIActivityViewController {
        var items = activityItems
        // Ensure we have valid items
        if items.isEmpty {
            print("⚠️ ShareSheet: No activity items provided")
        }
        let controller = UIActivityViewController(activityItems: items, applicationActivities: nil)
        
        // Configure for iPad
        if let popover = controller.popoverPresentationController {
            if let windowScene = UIApplication.shared.connectedScenes.first as? UIWindowScene,
               let window = windowScene.windows.first,
               let rootViewController = window.rootViewController {
                popover.sourceView = rootViewController.view
                popover.sourceRect = CGRect(x: UIScreen.main.bounds.width / 2, y: UIScreen.main.bounds.height / 2, width: 0, height: 0)
                popover.permittedArrowDirections = []
            }
        }
        
        return controller
    }
    
    func updateUIViewController(_ uiViewController: UIActivityViewController, context: Context) {}
}

