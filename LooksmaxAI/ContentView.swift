//
//  ContentView.swift
//  LooksmaxAI
//
//  Created by Admin on 2026. 01. 07..
//

import SwiftUI
import PhotosUI
import UIKit

struct ContentView: View {
    @AppStorage("lastAnalysisResults") private var lastAnalysisResultsData: Data?
    @State private var selectedTab = 0
    @State private var directResults: FaceAnalysisResults? = nil
    @State private var directResultsGender: String? = nil
    @State private var directResultsThumbnailPath: String? = nil
    @State private var directResultsThumbnailImage: UIImage? = nil
    @Environment(\.scenePhase) private var scenePhase
    
    init() {
        // Migrate thumbnail paths on app launch
        ResultHistoryManager.shared.migrateThumbnailPaths()
        
        // Check subscription status on app launch
        SubscriptionManager.shared.checkSubscriptionStatus()
        
        // Configure tab bar appearance to maintain consistent color
        // Use a dark gray color that's not pure black to avoid it turning black
        let appearance = UITabBarAppearance()
        appearance.configureWithOpaqueBackground()
        
        // Use a dark gray color (RGB: 28, 28, 30) which is the standard iOS dark background
        // This ensures the tab bar stays consistent and doesn't turn pure black
        appearance.backgroundColor = UIColor(red: 28/255.0, green: 28/255.0, blue: 30/255.0, alpha: 1.0)
        
        // Set the standard appearance for all tab bars with blur effect (glassmorphism)
        appearance.configureWithDefaultBackground()
        appearance.backgroundColor = UIColor(red: 0.11, green: 0.11, blue: 0.12, alpha: 0.95) // Dark gray with slight transparency
        
        // Add shadow for depth
        appearance.shadowColor = UIColor.black.withAlphaComponent(0.3)
        appearance.shadowImage = UIImage()
        
        // Configure selection colors
        appearance.selectionIndicatorTintColor = UIColor.systemBlue
        appearance.stackedLayoutAppearance.selected.iconColor = UIColor.systemBlue
        appearance.stackedLayoutAppearance.selected.titleTextAttributes = [.foregroundColor: UIColor.systemBlue]
        appearance.stackedLayoutAppearance.normal.iconColor = UIColor.gray
        appearance.stackedLayoutAppearance.normal.titleTextAttributes = [.foregroundColor: UIColor.gray]
        
        UITabBar.appearance().standardAppearance = appearance
        if #available(iOS 15.0, *) {
            UITabBar.appearance().scrollEdgeAppearance = appearance
        }
        
        // Use filled icons and add selection indicator
        UITabBar.appearance().tintColor = UIColor.systemBlue
        UITabBar.appearance().unselectedItemTintColor = UIColor.gray
    }
    
    var body: some View {
        TabView(selection: $selectedTab) {
            RateView(onAnalysisComplete: { results, gender in
                // Store results and switch tab
                print("🔄 onAnalysisComplete called with results, PSL: \(results.overall.psl ?? 0), gender: \(gender ?? "nil")")
                directResults = results
                
                // Get the latest result from history to get gender, thumbnail path and image
                // This ensures we get the gender that was actually saved (not nil from callback)
                let history = ResultHistoryManager.shared.loadHistory()
                if let latestItem = history.first {
                    // Use gender from history (it's guaranteed to be saved there)
                    directResultsGender = latestItem.gender ?? gender
                    directResultsThumbnailPath = latestItem.thumbnailPath
                    // Load thumbnail image immediately for instant display
                    if let path = latestItem.thumbnailPath {
                        directResultsThumbnailImage = ResultHistoryManager.shared.loadThumbnail(for: path)
                        print("📸 ContentView: Loaded thumbnail image: \(directResultsThumbnailImage != nil ? "success" : "failed"), path: \(path), gender: \(directResultsGender ?? "nil")")
                    }
                } else {
                    // Fallback to gender from callback if history not available yet
                    directResultsGender = gender
                }
                
                // Switch to Results tab after analysis
                DispatchQueue.main.async {
                    selectedTab = 1
                    print("🔄 Tab switched to Results (tag 1)")
                }
            })
                .tabItem {
                    Label("Rate", systemImage: "star.fill")
                }
                .tag(0)
            
            Group {
                if let results = directResults {
                    // Show ResultsView directly
                    ResultsView(results: results, onClose: {
                        print("🔄 Close button tapped, returning to list")
                        // Update state immediately (we're already on main thread from button tap)
                        directResults = nil
                        directResultsGender = nil
                        directResultsThumbnailPath = nil
                        directResultsThumbnailImage = nil
                        print("🔄 directResults set to nil, should show ResultsListView now")
                    }, gender: directResultsGender, thumbnailPath: directResultsThumbnailPath, thumbnailImage: directResultsThumbnailImage)
                } else {
                    // Show results list
                    ResultsListView()
                }
            }
            .animation(.default, value: directResults == nil)
            .tabItem {
                Label("Results", systemImage: "chart.bar.fill")
            }
            .tag(1)
            
            ProfileView()
                .tabItem {
                    Label("Maxxing", systemImage: "person.fill")
                }
                .tag(2)
            
            SettingsView()
                .tabItem {
                    Label("Settings", systemImage: "gearshape.fill")
                }
                .tag(3)
        }
        .onReceive(NotificationCenter.default.publisher(for: NSNotification.Name("SwitchToRateTab"))) { _ in
            selectedTab = 0
        }
        .onChange(of: scenePhase) { newPhase in
            // Check subscription status when app becomes active
            if newPhase == .active {
                SubscriptionManager.shared.checkSubscriptionStatus()
            }
        }
    }
}

enum PhotoStep {
    case front
    case side
}

struct RateView: View {
    let onAnalysisComplete: (FaceAnalysisResults, String?) -> Void
    @State private var showGenderSelection = false
    @State private var selectedGender: String? = nil
    @State private var currentPhotoStep: PhotoStep? = nil
    @State private var frontImage: UIImage? = nil
    @State private var sideImage: UIImage? = nil
    @State private var isAnalyzing = false
    @State private var analysisResults: FaceAnalysisResults? = nil
    @State private var errorMessage: String? = nil
    @State private var showError = false
    @State private var showUpgradePrompt = false
    @StateObject private var usageTracker = UsageTracker.shared
    @StateObject private var subscriptionManager = SubscriptionManager.shared
    @State private var isShowingUpgrade = false
    @AppStorage("lastAnalysisResults") private var lastAnalysisResultsData: Data?
    
    // Detect if device is iPhone (not iPad)
    private var isIPhone: Bool {
        UIDevice.current.userInterfaceIdiom == .phone
    }
    
    // Welcome/landing view when no gender/photo step selected
    private var welcomeView: some View {
        Group {
            if isIPhone {
                welcomeViewIPhone
            } else {
                welcomeViewIPad
            }
        }
    }
    
    private var welcomeViewIPhone: some View {
        VStack(spacing: isIPhone ? 16 : 30) {
            // Headline
            VStack(spacing: isIPhone ? 6 : 12) {
                Text("Discover Your True Rating")
                    .font(.system(size: isIPhone ? 26 : 36, weight: .bold, design: .rounded))
                    .foregroundColor(.white)
                
                HStack(spacing: 4) {
                    Image(systemName: "brain.head.profile")
                        .font(.system(size: isIPhone ? 10 : 12, weight: .medium))
                    Text("Powered by AI")
                        .font(.system(size: isIPhone ? 11 : 14, weight: .medium, design: .rounded))
                    Text("·")
                        .font(.system(size: isIPhone ? 11 : 14, weight: .medium))
                    Text("Analyzed with Precision")
                        .font(.system(size: isIPhone ? 11 : 14, weight: .medium, design: .rounded))
                }
                .foregroundColor(.gray.opacity(0.7))
            }
            .multilineTextAlignment(.center)
            .padding(.horizontal)
            .padding(.top, isIPhone ? 20 : 40)
            
            // Subheadline
            Text("AI-powered facial analysis for accurate and personalized insights.")
                .font(.system(size: isIPhone ? 14 : 19, weight: .regular))
                .foregroundColor(Color(red: 0.8, green: 0.8, blue: 0.8))
                .multilineTextAlignment(.center)
                .padding(.horizontal)
                .padding(.top, isIPhone ? 4 : 8)
            
            // Key Features with card-style layout
            VStack(spacing: isIPhone ? 10 : 16) {
                RateFeatureRow(icon: "chart.bar.fill", text: "50+ facial metrics analyzed", keyTerm: "50+ facial metrics")
                RateFeatureRow(icon: "star.fill", text: "PSL & Objective scale ratings", keyTerm: "PSL & Objective")
                RateFeatureRow(icon: "lightbulb.fill", text: "Personalized looksmaxxing advice", keyTerm: "Personalized")
            }
            .padding(.horizontal, isIPhone ? 20 : 40)
            .padding(.top, isIPhone ? 12 : 20)
            
            // Usage indicator for free users
            if !subscriptionManager.isPro {
                let remaining = usageTracker.getFaceRatingRemaining()
                if remaining > 0 {
                    Text("\(remaining) free rating\(remaining == 1 ? "" : "s") remaining")
                        .font(.system(size: isIPhone ? 11 : 14))
                        .foregroundColor(.gray)
                        .padding(.top, isIPhone ? 4 : 8)
                } else {
                    VStack(spacing: isIPhone ? 6 : 8) {
                        Text("You've used all free ratings")
                            .font(.system(size: isIPhone ? 14 : 16))
                            .foregroundColor(.white)
                        Text("Upgrade to Pro for unlimited ratings")
                            .font(.system(size: isIPhone ? 11 : 14))
                            .foregroundColor(.gray)
                        Button(action: {
                            isShowingUpgrade = true
                        }) {
                            Text("Upgrade to Pro")
                                .font(.system(size: isIPhone ? 14 : 16))
                                .foregroundColor(.white)
                                .padding(.horizontal, isIPhone ? 20 : 24)
                                .padding(.vertical, isIPhone ? 10 : 12)
                                .background(Color.red)
                                .cornerRadius(10)
                        }
                        .padding(.top, isIPhone ? 4 : 8)
                    }
                    .padding(isIPhone ? 12 : 16)
                    .background(
                        RoundedRectangle(cornerRadius: 12)
                            .fill(Color.white.opacity(0.1))
                    )
                    .padding(.horizontal, isIPhone ? 20 : 40)
                    .padding(.top, isIPhone ? 12 : 20)
                }
            }
            
            // Begin Button with gradient and icon
            Button(action: {
                if usageTracker.canUseFaceRating() {
                    showGenderSelection = true
                } else {
                    isShowingUpgrade = true
                }
            }) {
                HStack(spacing: 10) {
                    Image(systemName: "arrow.right.circle.fill")
                        .font(.system(size: isIPhone ? 16 : 20, weight: .semibold))
                    Text("Start My Analysis")
                        .font(.system(size: isIPhone ? 16 : 18, weight: .semibold))
                }
                .foregroundColor(.white)
                .frame(maxWidth: .infinity)
                .padding(.vertical, isIPhone ? 14 : 16)
                .background(
                    LinearGradient(
                        colors: [Color.blue, Color.cyan],
                        startPoint: .leading,
                        endPoint: .trailing
                    )
                )
                .cornerRadius(14)
                .shadow(color: Color.blue.opacity(0.4), radius: 12, x: 0, y: 6)
            }
            .padding(.horizontal, isIPhone ? 20 : 40)
            .padding(.top, isIPhone ? 16 : 30)
            .padding(.bottom, isIPhone ? 20 : 40)
            
            Spacer()
        }
    }
    
    private var welcomeViewIPad: some View {
        ScrollView {
            VStack(spacing: 30) {
                // Headline
                VStack(spacing: 12) {
                    Text("Discover Your True Rating")
                        .font(.system(size: 36, weight: .bold, design: .rounded))
                        .foregroundColor(.white)
                    
                    HStack(spacing: 4) {
                        Image(systemName: "brain.head.profile")
                            .font(.system(size: 12, weight: .medium))
                        Text("Powered by AI")
                            .font(.system(size: 14, weight: .medium, design: .rounded))
                        Text("·")
                            .font(.system(size: 14, weight: .medium))
                        Text("Analyzed with Precision")
                            .font(.system(size: 14, weight: .medium, design: .rounded))
                    }
                    .foregroundColor(.gray.opacity(0.7))
                }
                .multilineTextAlignment(.center)
                .padding(.horizontal)
                .padding(.top, 40)
                
                // Subheadline
                Text("AI-powered facial analysis for accurate and personalized insights.")
                    .font(.system(size: 19, weight: .regular))
                    .foregroundColor(Color(red: 0.8, green: 0.8, blue: 0.8))
                    .multilineTextAlignment(.center)
                    .padding(.horizontal)
                    .padding(.top, 8)
                
                // Key Features with card-style layout
                VStack(spacing: 16) {
                    RateFeatureRow(icon: "chart.bar.fill", text: "50+ facial metrics analyzed", keyTerm: "50+ facial metrics")
                    RateFeatureRow(icon: "star.fill", text: "PSL & Objective scale ratings", keyTerm: "PSL & Objective")
                    RateFeatureRow(icon: "lightbulb.fill", text: "Personalized looksmaxxing advice", keyTerm: "Personalized")
                }
                .padding(.horizontal, 40)
                .padding(.top, 20)
                
                // Usage indicator for free users
                if !subscriptionManager.isPro {
                    let remaining = usageTracker.getFaceRatingRemaining()
                    if remaining > 0 {
                        Text("\(remaining) free rating\(remaining == 1 ? "" : "s") remaining")
                            .font(.subheadline)
                            .foregroundColor(.gray)
                            .padding(.top, 8)
                    } else {
                        VStack(spacing: 8) {
                            Text("You've used all free ratings")
                                .font(.headline)
                                .foregroundColor(.white)
                            Text("Upgrade to Pro for unlimited ratings")
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
                            .padding(.top, 8)
                        }
                        .padding()
                        .background(
                            RoundedRectangle(cornerRadius: 12)
                                .fill(Color.white.opacity(0.1))
                        )
                        .padding(.horizontal, 40)
                        .padding(.top, 20)
                    }
                }
                
                // Begin Button with gradient and icon
                Button(action: {
                    if usageTracker.canUseFaceRating() {
                        showGenderSelection = true
                    } else {
                        isShowingUpgrade = true
                    }
                }) {
                    HStack(spacing: 10) {
                        Image(systemName: "arrow.right.circle.fill")
                            .font(.system(size: 20, weight: .semibold))
                        Text("Start My Analysis")
                            .font(.system(size: 18, weight: .semibold))
                    }
                    .foregroundColor(.white)
                    .frame(maxWidth: .infinity)
                    .padding(.vertical, 16)
                    .background(
                        LinearGradient(
                            colors: [Color.blue, Color.cyan],
                            startPoint: .leading,
                            endPoint: .trailing
                        )
                    )
                    .cornerRadius(14)
                    .shadow(color: Color.blue.opacity(0.4), radius: 12, x: 0, y: 6)
                }
                .padding(.horizontal, 40)
                .padding(.top, 30)
                .padding(.bottom, 40)
            }
        }
    }
    
    private var analyzingView: some View {
        VStack(spacing: 20) {
            ProgressView()
                .scaleEffect(1.5)
                .tint(.blue)
            Text("Analyzing your face...")
                .font(.headline)
                .foregroundColor(.gray)
            Text("This may take 30-60 seconds")
                .font(.caption)
                .foregroundColor(.gray.opacity(0.7))
        }
        .transition(.opacity)
    }
    
    private var genderSelectionView: some View {
        VStack(spacing: 30) {
            Text("Select Your Gender")
                .font(.title2)
                .fontWeight(.semibold)
                .foregroundColor(.white)
                .padding(.top)
            
            HStack(spacing: 30) {
                // Male button
                Button(action: {
                    selectedGender = "Male"
                }) {
                    VStack(spacing: 12) {
                        Image("man")
                            .resizable()
                            .aspectRatio(contentMode: .fit)
                            .frame(width: 120, height: 120)
                        
                        Text("Male")
                            .font(.headline)
                            .foregroundColor(.white)
                    }
                    .padding(20)
                    .background(Color.black)
                    .overlay(
                        RoundedRectangle(cornerRadius: 12)
                            .stroke(
                                selectedGender == "Male" ? Color.blue : Color.clear,
                                lineWidth: 3
                            )
                    )
                    .cornerRadius(12)
                }
                
                // Female button
                Button(action: {
                    selectedGender = "Female"
                }) {
                    VStack(spacing: 12) {
                        Image("woman")
                            .resizable()
                            .aspectRatio(contentMode: .fit)
                            .frame(width: 120, height: 120)
                        
                        Text("Female")
                            .font(.headline)
                            .foregroundColor(.white)
                    }
                    .padding(20)
                    .background(Color.black)
                    .overlay(
                        RoundedRectangle(cornerRadius: 12)
                            .stroke(
                                selectedGender == "Female" ? Color.blue : Color.clear,
                                lineWidth: 3
                            )
                    )
                    .cornerRadius(12)
                }
            }
            .padding(.horizontal)
            
            if selectedGender != nil {
                Button(action: {
                    currentPhotoStep = .front
                }) {
                    Text("Continue")
                        .font(.headline)
                        .foregroundColor(.white)
                        .frame(maxWidth: .infinity)
                        .padding()
                        .background(Color.blue)
                        .cornerRadius(10)
                }
                .padding(.horizontal)
                .padding(.top)
            }
        }
        .frame(maxWidth: .infinity, maxHeight: .infinity)
        .background(Color.black)
    }
    
    private var mainContentView: some View {
        Group {
            if isAnalyzing {
                analyzingView
            } else if !showGenderSelection && currentPhotoStep == nil {
                welcomeView
            } else if showGenderSelection && currentPhotoStep == nil {
                genderSelectionView
            } else if currentPhotoStep == .front {
                PhotoSelectionView(
                    gender: selectedGender ?? "Male",
                    title: "Take or Pick a front selfie",
                    selectedImage: $frontImage,
                    isSideProfile: false,
                    onContinue: {
                        currentPhotoStep = .side
                    }
                )
            } else if currentPhotoStep == .side {
                PhotoSelectionView(
                    gender: selectedGender ?? "Male",
                    title: "Take a side profile selfie",
                    selectedImage: $sideImage,
                    isSideProfile: true,
                    sideProfileButtonText: "Generate Results",
                    onContinue: {
                        if usageTracker.canUseFaceRating() {
                            analyzeFaces()
                        } else {
                            isShowingUpgrade = true
                        }
                    }
                )
            }
        }
    }
    
    var body: some View {
        NavigationStack {
            ZStack {
                Color.black.ignoresSafeArea()
                mainContentView
            }
            .navigationTitle("Mogged")
            .navigationBarTitleDisplayMode(.inline)
            .toolbar {
                // Show back button and start over when user has progressed past initial screen
                if showGenderSelection || currentPhotoStep != nil {
                    ToolbarItem(placement: .navigationBarLeading) {
                        Button(action: {
                            if let step = currentPhotoStep {
                                // Go back to previous step
                                if step == .side {
                                    currentPhotoStep = .front
                                } else if step == .front {
                                    currentPhotoStep = nil
                                    showGenderSelection = true
                                }
                            } else if showGenderSelection {
                                // Go back to initial screen
                                showGenderSelection = false
                            }
                        }) {
                            HStack(spacing: 4) {
                                Image(systemName: "chevron.left")
                                Text("Back")
                            }
                            .foregroundColor(.white)
                        }
                    }
                    
                    ToolbarItem(placement: .navigationBarTrailing) {
                        Button(action: {
                            resetRateView()
                        }) {
                            Text("Start Over")
                                .foregroundColor(.red)
                        }
                    }
                }
            }
        }
        .onAppear {
            loadLastResults()
        }
        .alert("Analysis Failed", isPresented: $showError) {
            Button("OK", role: .cancel) { }
            Button("Retry") {
                if let front = frontImage, let side = sideImage, let gender = selectedGender {
                    analyzeFaces()
                }
            }
        } message: {
            Text(errorMessage ?? "An unknown error occurred. Please try again.")
        }
        .fullScreenCover(isPresented: $isShowingUpgrade) {
            UpgradeView()
        }
    }
    
    private func analyzeFaces() {
        guard let front = frontImage, let side = sideImage, let gender = selectedGender else {
            return
        }
        
        isAnalyzing = true
        errorMessage = nil
        
        // Gender is now optional - backend will auto-detect if provided, otherwise it detects automatically
        FaceAnalysisService.shared.analyzeFaces(
            frontImage: front,
            sideImage: side,
            gender: gender  // Can be nil for auto-detection
        ) { result in
            print("📱 Completion handler called on main thread: \(Thread.isMainThread)")
            
            // Ensure we're on main thread for UI updates
            if Thread.isMainThread {
                self.handleAnalysisResult(result)
            } else {
                DispatchQueue.main.async {
                    self.handleAnalysisResult(result)
                }
            }
        }
    }
    
    private func handleAnalysisResult(_ result: Result<FaceAnalysisResults, Error>) {
        print("📱 Handling analysis result on main thread: \(Thread.isMainThread)")
        
        // Process the result
        switch result {
        case .success(let results):
            print("✅ Analysis successful, PSL: \(results.overall.psl ?? 0)")
            
            // Increment usage counter (only for free users)
            usageTracker.incrementFaceRating()
            
            // Save results first
            analysisResults = results
            saveResults(results)
            ResultHistoryManager.shared.saveResult(results, frontImage: frontImage, gender: selectedGender)
            print("💾 Results saved to history with thumbnail")
            
            // CRITICAL: Stop the loading indicator BEFORE navigation
            print("📱 Setting isAnalyzing = false")
            isAnalyzing = false
            
            // Reset RateView to initial state
            resetRateView()
            print("🔄 RateView reset to initial state")
            
            // Small delay to ensure UI updates before navigation
            DispatchQueue.main.asyncAfter(deadline: .now() + 0.1) {
                // Switch to Results tab and pass results
                print("🔄 Calling onAnalysisComplete with results to switch tabs...")
                self.onAnalysisComplete(results, selectedGender)
                print("🔄 onAnalysisComplete called with results")
            }
        case .failure(let error):
            print("❌ Analysis failed: \(error.localizedDescription)")
            // Stop loading on error too
            isAnalyzing = false
            errorMessage = error.localizedDescription
            showError = true
        }
    }
    
    private func saveResults(_ results: FaceAnalysisResults) {
        if let encoded = try? JSONEncoder().encode(results) {
            lastAnalysisResultsData = encoded
        }
    }
    
    private func loadLastResults() {
        if let data = lastAnalysisResultsData,
           let results = try? JSONDecoder().decode(FaceAnalysisResults.self, from: data) {
            analysisResults = results
        }
    }
    
    private func resetRateView() {
        // Reset all state variables to initial state
        showGenderSelection = false
        selectedGender = nil
        currentPhotoStep = nil
        frontImage = nil
        sideImage = nil
        analysisResults = nil
        errorMessage = nil
        showError = false
    }
}

struct PhotoSelectionView: View {
    let gender: String
    let title: String
    @Binding var selectedImage: UIImage?
    let isSideProfile: Bool
    var sideProfileButtonText: String = "Generate Photo" // Default for other views
    let onContinue: () -> Void
    @State private var showImagePicker = false
    @State private var showCamera = false
    @State private var showPhotoSourceOptions = false
    @State private var imagePickerSourceType: UIImagePickerController.SourceType = .photoLibrary
    @State private var isValidating = false
    @State private var validationError: String? = nil
    @State private var showValidationError = false
    @State private var showGuidance = false
    
    var body: some View {
        VStack(spacing: 30) {
            VStack(spacing: 10) {
                HStack {
                    Spacer()
                    Button(action: {
                        showGuidance = true
                    }) {
                        HStack(spacing: 5) {
                            Image(systemName: "info.circle")
                            Text("Photo Tips")
                        }
                        .font(.caption)
                        .foregroundColor(.blue)
                    }
                }
                .padding(.horizontal)
                
                Text(title)
                    .font(.title2)
                    .fontWeight(.semibold)
                    .multilineTextAlignment(.center)
                
                if isSideProfile {
                    Text(gender == "Female" ? "Turn your head 90° to show your side profile (hair can cover your ear)" : "Turn your head 90° to show your side profile with your ear visible")
                        .font(.caption)
                        .foregroundColor(.gray)
                        .multilineTextAlignment(.center)
                        .padding(.horizontal)
                } else {
                    Text("Face the camera directly like a passport photo with good lighting")
                        .font(.caption)
                        .foregroundColor(.gray)
                        .multilineTextAlignment(.center)
                        .padding(.horizontal)
                }
            }
            .padding(.top)
            
            // Display placeholder or selected image
            if let image = selectedImage {
                Image(uiImage: image)
                    .resizable()
                    .scaledToFit()
                    .frame(width: 300, height: 300)
                    .cornerRadius(10)
                    .shadow(radius: 5)
            } else {
                if isSideProfile {
                    Image(gender == "Male" ? "Man_side" : "Woman_side")
                        .resizable()
                        .scaledToFit()
                        .frame(width: 300, height: 300)
                        .cornerRadius(10)
                } else {
                    Image(gender == "Male" ? "Man_front" : "Woman_front")
                        .resizable()
                        .scaledToFit()
                        .frame(width: 300, height: 300)
                        .cornerRadius(10)
                }
            }
            
            if selectedImage != nil {
                // Show Continue and Retake buttons when image is selected
                VStack(spacing: 15) {
                    if isValidating {
                        HStack {
                            ProgressView()
                            Text("Validating image...")
                                .font(.subheadline)
                                .foregroundColor(.gray)
                        }
                        .padding()
                    } else {
                        Button(action: {
                            validateAndContinue()
                        }) {
                            Text(isSideProfile ? sideProfileButtonText : "Continue")
                                .font(.headline)
                                .foregroundColor(.white)
                                .frame(maxWidth: .infinity)
                                .padding()
                                .background(Color.blue)
                                .cornerRadius(10)
                        }
                    }
                    
                    Button(action: {
                        showPhotoSourceOptions = true
                    }) {
                        Text("Retake/Pick Another Photo")
                            .font(.headline)
                            .foregroundColor(.blue)
                            .frame(maxWidth: .infinity)
                            .padding()
                            .background(Color.clear)
                            .overlay(
                                RoundedRectangle(cornerRadius: 10)
                                    .stroke(Color.blue, lineWidth: 2)
                            )
                    }
                }
                .padding(.horizontal)
            } else {
                // Show Pick and Take buttons when no image is selected
                HStack(spacing: 20) {
                    Button(action: {
                        imagePickerSourceType = .photoLibrary
                        showImagePicker = true
                    }) {
                        Text("Pick a Photo")
                            .font(.headline)
                            .foregroundColor(.white)
                            .frame(maxWidth: .infinity)
                            .padding()
                            .background(Color.blue)
                            .cornerRadius(10)
                    }
                    
                    Button(action: {
                        imagePickerSourceType = .camera
                        showCamera = true
                    }) {
                        Text("Take a Photo")
                            .font(.headline)
                            .foregroundColor(.white)
                            .frame(maxWidth: .infinity)
                            .padding()
                            .background(Color.blue)
                            .cornerRadius(10)
                    }
                }
                .padding(.horizontal)
            }
        }
        .padding()
        .sheet(isPresented: $showImagePicker) {
            ImagePicker(selectedImage: $selectedImage, sourceType: imagePickerSourceType)
        }
        .sheet(isPresented: $showCamera) {
            ImagePicker(selectedImage: $selectedImage, sourceType: .camera)
        }
        .confirmationDialog("Choose Photo Source", isPresented: $showPhotoSourceOptions, titleVisibility: .visible) {
            Button("Pick from Library") {
                imagePickerSourceType = .photoLibrary
                showImagePicker = true
            }
            
            Button("Take Photo") {
                imagePickerSourceType = .camera
                showCamera = true
            }
            
            Button("Cancel", role: .cancel) { }
        }
        .alert("Image Validation Failed", isPresented: $showValidationError) {
            Button("OK", role: .cancel) { }
            Button("Retake") {
                showPhotoSourceOptions = true
            }
            Button("View Tips") {
                showGuidance = true
            }
        } message: {
            Text(validationError ?? "Please try again with a different photo.")
        }
        .sheet(isPresented: $showGuidance) {
            PhotoGuidanceView(isSideProfile: isSideProfile, gender: gender)
        }
    }
    
    private func validateAndContinue() {
        guard let image = selectedImage else { return }
        
        isValidating = true
        validationError = nil
        
        if isSideProfile {
            ImageValidationService.shared.validateSideImage(image, gender: gender) { result in
                DispatchQueue.main.async {
                    isValidating = false
                    
                    if result.isValid {
                        onContinue()
                    } else {
                        validationError = result.error?.errorDescription ?? "Image validation failed. Please try again."
                        showValidationError = true
                    }
                }
            }
        } else {
            ImageValidationService.shared.validateFrontImage(image, gender: gender) { result in
                DispatchQueue.main.async {
                    isValidating = false
                    
                    if result.isValid {
                        onContinue()
                    } else {
                        validationError = result.error?.errorDescription ?? "Image validation failed. Please try again."
                        showValidationError = true
                    }
                }
            }
        }
    }
}

struct ImagePicker: UIViewControllerRepresentable {
    @Binding var selectedImage: UIImage?
    var sourceType: UIImagePickerController.SourceType
    @Environment(\.dismiss) var dismiss
    
    func makeUIViewController(context: Context) -> UIImagePickerController {
        let picker = UIImagePickerController()
        picker.sourceType = sourceType
        picker.delegate = context.coordinator
        return picker
    }
    
    func updateUIViewController(_ uiViewController: UIImagePickerController, context: Context) {}
    
    func makeCoordinator() -> Coordinator {
        Coordinator(self)
    }
    
    class Coordinator: NSObject, UIImagePickerControllerDelegate, UINavigationControllerDelegate {
        let parent: ImagePicker
        
        init(_ parent: ImagePicker) {
            self.parent = parent
        }
        
        func imagePickerController(_ picker: UIImagePickerController, didFinishPickingMediaWithInfo info: [UIImagePickerController.InfoKey : Any]) {
            if let image = info[.originalImage] as? UIImage {
                parent.selectedImage = image
            }
            parent.dismiss()
        }
        
        func imagePickerControllerDidCancel(_ picker: UIImagePickerController) {
            parent.dismiss()
        }
    }
}

// ResultsView is now in ResultsView.swift

struct ProfileView: View {
    @State private var iconAnimations: [Bool] = [false, false, false, false]
    
    // Detect if device is iPhone (not iPad)
    private var isIPhone: Bool {
        UIDevice.current.userInterfaceIdiom == .phone
    }
    
    var body: some View {
        NavigationStack {
            ZStack {
                // Background gradient
                LinearGradient(
                    colors: [
                        Color.black,
                        Color(red: 0.05, green: 0.05, blue: 0.1),
                        Color.black
                    ],
                    startPoint: .topLeading,
                    endPoint: .bottomTrailing
                )
                .ignoresSafeArea()
                
                // Static view for iPhone, ScrollView for iPad
                if isIPhone {
                    VStack(spacing: isIPhone ? 12 : 24) {
                        // Enhanced Header
                        VStack(spacing: isIPhone ? 6 : 12) {
                            HStack(spacing: isIPhone ? 8 : 12) {
                                Image(systemName: "rocket.fill")
                                    .font(.system(size: isIPhone ? 24 : 32, weight: .bold))
                                    .foregroundStyle(
                                        LinearGradient(
                                            colors: [Color.cyan, Color.purple],
                                            startPoint: .topLeading,
                                            endPoint: .bottomTrailing
                                        )
                                    )
                                    .scaleEffect(iconAnimations[0] ? 1.1 : 1.0)
                                    .animation(.spring(response: 0.6, dampingFraction: 0.6).repeatForever(autoreverses: true), value: iconAnimations[0])
                                
                                Text("Start Maxxing")
                                    .font(.system(size: isIPhone ? 26 : 36, weight: .bold, design: .rounded))
                                    .foregroundColor(.white)
                            }
                            
                            Text("Explore personalized strategies to level up your social and physical game.")
                                .font(.system(size: isIPhone ? 12 : 15, weight: .medium))
                                .foregroundColor(.gray.opacity(0.9))
                                .multilineTextAlignment(.center)
                                .padding(.horizontal, isIPhone ? 16 : 20)
                        }
                        .padding(.top, isIPhone ? 12 : 20)
                        .padding(.bottom, isIPhone ? 6 : 10)
                        
                        // Tindermaxxing Card
                        MaxxingCard(
                            icon: "heart.fill",
                            iconColor: .pink,
                            iconBackground: LinearGradient(colors: [Color.pink.opacity(0.3), Color.red.opacity(0.2)], startPoint: .topLeading, endPoint: .bottomTrailing),
                            title: "Tindermaxxing",
                            description: "Upload your photo of yourself and create Neurotypical or \"NT\" pics for your dating profile.",
                            cardGradient: LinearGradient(colors: [Color.pink.opacity(0.15), Color.red.opacity(0.1)], startPoint: .topLeading, endPoint: .bottomTrailing),
                            destination: OnlineDatingMaxxingView(),
                            animationIndex: 0,
                            iconAnimations: $iconAnimations,
                            isIPhone: isIPhone
                        )
                        
                        // Rizz Coach Card
                        MaxxingCard(
                            icon: "message.fill",
                            iconColor: .cyan,
                            iconBackground: LinearGradient(colors: [Color.cyan.opacity(0.3), Color.blue.opacity(0.2)], startPoint: .topLeading, endPoint: .bottomTrailing),
                            title: "Rizz Coach",
                            description: "Struggling with conversations? Get advice and text responses to help.",
                            cardGradient: LinearGradient(colors: [Color.cyan.opacity(0.15), Color.blue.opacity(0.1)], startPoint: .topLeading, endPoint: .bottomTrailing),
                            destination: RizzMaxxingView(),
                            animationIndex: 1,
                            iconAnimations: $iconAnimations,
                            isIPhone: isIPhone
                        )
                        
                        // Looksmaxxing Card
                        MaxxingCard(
                            icon: "sparkles",
                            iconColor: .purple,
                            iconBackground: LinearGradient(colors: [Color.purple.opacity(0.3), Color.pink.opacity(0.2)], startPoint: .topLeading, endPoint: .bottomTrailing),
                            title: "Looksmaxxing",
                            description: "Understand your looks — and how to improve them. Get personalized looksmaxxing advice.",
                            cardGradient: LinearGradient(colors: [Color.purple.opacity(0.15), Color.pink.opacity(0.1)], startPoint: .topLeading, endPoint: .bottomTrailing),
                            destination: LooksmaxxingView(),
                            animationIndex: 2,
                            iconAnimations: $iconAnimations,
                            isIPhone: isIPhone
                        )
                        
                        // Sub 5 to Mogger Card
                        MaxxingCard(
                            icon: "crown.fill",
                            iconColor: .yellow,
                            iconBackground: LinearGradient(colors: [Color.yellow.opacity(0.3), Color.orange.opacity(0.2)], startPoint: .topLeading, endPoint: .bottomTrailing),
                            title: "Sub 5 to Mogger",
                            description: "See what you would look like as a mogger.",
                            cardGradient: LinearGradient(colors: [Color.yellow.opacity(0.15), Color.orange.opacity(0.1)], startPoint: .topLeading, endPoint: .bottomTrailing),
                            destination: Sub5ToChadView(),
                            animationIndex: 3,
                            iconAnimations: $iconAnimations,
                            isIPhone: isIPhone
                        )
                        
                        Spacer()
                    }
                    .padding(isIPhone ? 12 : 16)
                } else {
                    ScrollView {
                        VStack(spacing: 24) {
                            // Enhanced Header
                            VStack(spacing: 12) {
                                HStack(spacing: 12) {
                                    Image(systemName: "rocket.fill")
                                        .font(.system(size: 32, weight: .bold))
                                        .foregroundStyle(
                                            LinearGradient(
                                                colors: [Color.cyan, Color.purple],
                                                startPoint: .topLeading,
                                                endPoint: .bottomTrailing
                                            )
                                        )
                                        .scaleEffect(iconAnimations[0] ? 1.1 : 1.0)
                                        .animation(.spring(response: 0.6, dampingFraction: 0.6).repeatForever(autoreverses: true), value: iconAnimations[0])
                                    
                                    Text("Start Maxxing")
                                        .font(.system(size: 36, weight: .bold, design: .rounded))
                                        .foregroundColor(.white)
                                }
                                
                                Text("Explore personalized strategies to level up your social and physical game.")
                                    .font(.system(size: 15, weight: .medium))
                                    .foregroundColor(.gray.opacity(0.9))
                                    .multilineTextAlignment(.center)
                                    .padding(.horizontal, 20)
                            }
                            .padding(.top, 20)
                            .padding(.bottom, 10)
                            
                            // Tindermaxxing Card
                            MaxxingCard(
                                icon: "heart.fill",
                                iconColor: .pink,
                                iconBackground: LinearGradient(colors: [Color.pink.opacity(0.3), Color.red.opacity(0.2)], startPoint: .topLeading, endPoint: .bottomTrailing),
                                title: "Tindermaxxing",
                                description: "Upload your photo of yourself and create Neurotypical or \"NT\" pics for your dating profile.",
                                cardGradient: LinearGradient(colors: [Color.pink.opacity(0.15), Color.red.opacity(0.1)], startPoint: .topLeading, endPoint: .bottomTrailing),
                                destination: OnlineDatingMaxxingView(),
                                animationIndex: 0,
                                iconAnimations: $iconAnimations,
                                isIPhone: false
                            )
                            
                            // Rizz Coach Card
                            MaxxingCard(
                                icon: "message.fill",
                                iconColor: .cyan,
                                iconBackground: LinearGradient(colors: [Color.cyan.opacity(0.3), Color.blue.opacity(0.2)], startPoint: .topLeading, endPoint: .bottomTrailing),
                                title: "Rizz Coach",
                                description: "Struggling with conversations? Get advice and text responses to help.",
                                cardGradient: LinearGradient(colors: [Color.cyan.opacity(0.15), Color.blue.opacity(0.1)], startPoint: .topLeading, endPoint: .bottomTrailing),
                                destination: RizzMaxxingView(),
                                animationIndex: 1,
                                iconAnimations: $iconAnimations,
                                isIPhone: false
                            )
                            
                            // Looksmaxxing Card
                            MaxxingCard(
                                icon: "sparkles",
                                iconColor: .purple,
                                iconBackground: LinearGradient(colors: [Color.purple.opacity(0.3), Color.pink.opacity(0.2)], startPoint: .topLeading, endPoint: .bottomTrailing),
                                title: "Looksmaxxing",
                                description: "Understand your looks — and how to improve them. Get personalized looksmaxxing advice.",
                                cardGradient: LinearGradient(colors: [Color.purple.opacity(0.15), Color.pink.opacity(0.1)], startPoint: .topLeading, endPoint: .bottomTrailing),
                                destination: LooksmaxxingView(),
                                animationIndex: 2,
                                iconAnimations: $iconAnimations,
                                isIPhone: false
                            )
                            
                            // Sub 5 to Mogger Card
                            MaxxingCard(
                                icon: "crown.fill",
                                iconColor: .yellow,
                                iconBackground: LinearGradient(colors: [Color.yellow.opacity(0.3), Color.orange.opacity(0.2)], startPoint: .topLeading, endPoint: .bottomTrailing),
                                title: "Sub 5 to Mogger",
                                description: "See what you would look like as a mogger.",
                                cardGradient: LinearGradient(colors: [Color.yellow.opacity(0.15), Color.orange.opacity(0.1)], startPoint: .topLeading, endPoint: .bottomTrailing),
                                destination: Sub5ToChadView(),
                                animationIndex: 3,
                                iconAnimations: $iconAnimations,
                                isIPhone: false
                            )
                        }
                        .padding()
                    }
                }
            }
            .onAppear {
                // Trigger icon animations on appear
                DispatchQueue.main.asyncAfter(deadline: .now() + 0.1) {
                    for i in 0..<iconAnimations.count {
                        DispatchQueue.main.asyncAfter(deadline: .now() + Double(i) * 0.15) {
                            iconAnimations[i] = true
                        }
                    }
                }
            }
        }
    }
}

struct MaxxingCard<Destination: View>: View {
    let icon: String
    let iconColor: Color
    let iconBackground: LinearGradient
    let title: String
    let description: String
    let cardGradient: LinearGradient
    let destination: Destination
    let animationIndex: Int
    @Binding var iconAnimations: [Bool]
    let isIPhone: Bool
    @State private var isPressed = false
    
    init(icon: String, iconColor: Color, iconBackground: LinearGradient, title: String, description: String, cardGradient: LinearGradient, destination: Destination, animationIndex: Int, iconAnimations: Binding<[Bool]>, isIPhone: Bool = false) {
        self.icon = icon
        self.iconColor = iconColor
        self.iconBackground = iconBackground
        self.title = title
        self.description = description
        self.cardGradient = cardGradient
        self.destination = destination
        self.animationIndex = animationIndex
        self._iconAnimations = iconAnimations
        self.isIPhone = isIPhone
    }
    
    var body: some View {
        NavigationLink(destination: destination) {
            VStack(alignment: .leading, spacing: isIPhone ? 10 : 16) {
                HStack(spacing: isIPhone ? 12 : 16) {
                    // Icon with badge background
                    ZStack {
                        Circle()
                            .fill(iconBackground)
                            .frame(width: isIPhone ? 44 : 56, height: isIPhone ? 44 : 56)
                            .shadow(color: iconColor.opacity(0.3), radius: 8, x: 0, y: 4)
                        
                        Image(systemName: icon)
                            .font(.system(size: isIPhone ? 18 : 24, weight: .semibold))
                            .foregroundColor(iconColor)
                            .scaleEffect(iconAnimations[animationIndex] ? 1.1 : 1.0)
                            .animation(.spring(response: 0.6, dampingFraction: 0.6).repeatForever(autoreverses: true), value: iconAnimations[animationIndex])
                    }
                    
                    // Title
                    Text(title)
                        .font(.system(size: isIPhone ? 16 : 20, weight: .bold))
                        .foregroundColor(.white)
                    
                    Spacer()
                    
                    // Chevron
                    Image(systemName: "chevron.right")
                        .font(.system(size: isIPhone ? 12 : 14, weight: .semibold))
                        .foregroundColor(.gray.opacity(0.6))
                        .offset(x: isPressed ? 4 : 0)
                }
                
                // Description
                Text(description)
                    .font(.system(size: isIPhone ? 11 : 14, weight: .regular))
                    .foregroundColor(.gray.opacity(0.8))
                    .multilineTextAlignment(.leading)
                    .fixedSize(horizontal: false, vertical: true)
            }
            .padding(isIPhone ? 14 : 20)
            .frame(maxWidth: .infinity, alignment: .leading)
            .background(
                RoundedRectangle(cornerRadius: isIPhone ? 16 : 20)
                    .fill(
                        // Glassmorphism effect
                        LinearGradient(
                            colors: [
                                Color.white.opacity(0.12),
                                Color.white.opacity(0.08)
                            ],
                            startPoint: .topLeading,
                            endPoint: .bottomTrailing
                        )
                    )
                    .background(
                        RoundedRectangle(cornerRadius: isIPhone ? 16 : 20)
                            .fill(cardGradient.opacity(0.3))
                    )
                    .overlay(
                        RoundedRectangle(cornerRadius: isIPhone ? 16 : 20)
                            .stroke(
                                LinearGradient(
                                    colors: [
                                        Color.white.opacity(0.25),
                                        Color.white.opacity(0.1)
                                    ],
                                    startPoint: .topLeading,
                                    endPoint: .bottomTrailing
                                ),
                                lineWidth: 1
                            )
                    )
            )
            .shadow(color: Color.white.opacity(0.05), radius: 6, x: 0, y: 2)
            .scaleEffect(isPressed ? 0.98 : 1.0)
            .animation(.spring(response: 0.3, dampingFraction: 0.6), value: isPressed)
        }
        .buttonStyle(PlainButtonStyle())
        .simultaneousGesture(
            DragGesture(minimumDistance: 0)
                .onChanged { _ in
                    isPressed = true
                }
                .onEnded { _ in
                    isPressed = false
                }
        )
    }
}

struct SettingsView: View {
    @StateObject private var scaleManager = ScaleManager.shared
    @StateObject private var subscriptionManager = SubscriptionManager.shared
    @StateObject private var usageTracker = UsageTracker.shared
    @State private var isShowingUpgrade = false
    @State private var isShowingCreditPurchase = false
    @State private var isRestoring = false
    @State private var showRestoreSuccess = false
    @State private var showRestoreError = false
    @State private var restoreError: String?
    
    init() {
        // Configure segmented control appearance for better visibility
        UISegmentedControl.appearance().setTitleTextAttributes([
            .foregroundColor: UIColor.white,
            .font: UIFont.systemFont(ofSize: 13, weight: .medium)
        ], for: .normal)
        
        UISegmentedControl.appearance().setTitleTextAttributes([
            .foregroundColor: UIColor.black,
            .font: UIFont.systemFont(ofSize: 13, weight: .semibold)
        ], for: .selected)
        
        // Make unselected segments more visible
        UISegmentedControl.appearance().backgroundColor = UIColor(white: 0.2, alpha: 1.0)
        UISegmentedControl.appearance().selectedSegmentTintColor = UIColor.white
    }
    
    var body: some View {
        NavigationStack {
            ZStack {
                Color.black.ignoresSafeArea()
                
                List {
                    Section(header: Text("Display Scale")
                        .foregroundColor(.white)
                        .font(.headline)) {
                        Picker("Scale", selection: $scaleManager.selectedScale) {
                            ForEach(DisplayScale.allCases, id: \.self) { scale in
                                Text(scale.displayName).tag(scale)
                            }
                        }
                        .pickerStyle(.segmented)
                        
                        Text("Choose how PSL scores are displayed:")
                            .font(.subheadline)
                            .fontWeight(.semibold)
                            .foregroundColor(.white)
                            .padding(.top, 12)
                        
                        VStack(alignment: .leading, spacing: 10) {
                            Text("• PSL Scale (0-8): Traditional looksmax scale")
                                .font(.body)
                                .fontWeight(.medium)
                                .foregroundColor(.white.opacity(0.9))
                            Text("• 1-10 Objective Scale: More intuitive rating")
                                .font(.body)
                                .fontWeight(.medium)
                                .foregroundColor(.white.opacity(0.9))
                        }
                        .padding(.top, 12)
                    }
                    .listRowBackground(Color.white.opacity(0.05))
                    
                    Section(header: Text("Scale Reference")
                        .foregroundColor(.white)
                        .font(.headline)) {
                        ScaleReferenceSection(selectedScale: scaleManager.selectedScale)
                    }
                    .listRowBackground(Color.white.opacity(0.05))
                    
                    Section(header: Text("Credits")
                        .foregroundColor(.white)
                        .font(.headline)) {
                        Button {
                            isShowingCreditPurchase = true
                        } label: {
                            HStack {
                                Image(systemName: "sparkles")
                                    .foregroundColor(.cyan)
                                    .font(.system(size: 18))
                                VStack(alignment: .leading, spacing: 4) {
                                    Text("Purchase Credits")
                                        .foregroundColor(.white)
                                        .font(.body)
                                    Text("\(usageTracker.getImageGenerationCreditsRemaining()) credits available")
                                        .foregroundColor(.gray)
                                        .font(.caption)
                                }
                                Spacer()
                                Image(systemName: "chevron.right")
                                    .font(.caption)
                                    .foregroundColor(.gray)
                            }
                        }
                        
                        Text("Credits are used for photo generation features. Pro subscribers receive monthly credits, and all users can purchase additional credits.")
                            .font(.caption)
                            .foregroundColor(.gray.opacity(0.8))
                            .padding(.top, 4)
                    }
                    .listRowBackground(Color.white.opacity(0.05))
                    
                    Section(header: Text("Pro")
                        .foregroundColor(.white)
                        .font(.headline)) {
                        if subscriptionManager.isPro {
                            // Show Restore Purchases for Pro users
                            Button {
                                handleRestorePurchases()
                            } label: {
                                HStack {
                                    if isRestoring {
                                        ProgressView()
                                            .tint(.white)
                                            .scaleEffect(0.8)
                                    } else {
                                        Image(systemName: "arrow.clockwise")
                                            .foregroundColor(.blue)
                                    }
                                    Text(isRestoring ? "Restoring..." : "Restore Purchases")
                                        .foregroundColor(.white)
                                    Spacer()
                                    if !isRestoring {
                                        Image(systemName: "chevron.right")
                                            .font(.caption)
                                            .foregroundColor(.gray)
                                    }
                                }
                            }
                            .disabled(isRestoring)
                        } else {
                            // Show Upgrade to Pro for free users
                            Button {
                                isShowingUpgrade = true
                            } label: {
                                HStack {
                                    Image(systemName: "star.fill")
                                        .foregroundColor(.yellow)
                                    Text("Upgrade to Pro")
                                        .foregroundColor(.white)
                                    Spacer()
                                    Image(systemName: "chevron.right")
                                        .font(.caption)
                                        .foregroundColor(.gray)
                                }
                            }
                        }
                    }
                    .listRowBackground(Color.white.opacity(0.05))
                    
                    Section(header: Text("Legal")
                        .foregroundColor(.white)
                        .font(.headline)) {
                        Link(destination: URL(string: "https://www.faceratingapp.com/privacy")!) {
                            HStack {
                                Image(systemName: "lock.shield.fill")
                                    .foregroundColor(.blue)
                                Text("Privacy Policy")
                                    .foregroundColor(.white)
                                Spacer()
                                Image(systemName: "chevron.right")
                                    .font(.caption)
                                    .foregroundColor(.gray)
                            }
                        }
                        
                        Link(destination: URL(string: "https://www.faceratingapp.com/terms")!) {
                            HStack {
                                Image(systemName: "doc.text.fill")
                                    .foregroundColor(.blue)
                                Text("Terms of Use")
                                    .foregroundColor(.white)
                                Spacer()
                                Image(systemName: "chevron.right")
                                    .font(.caption)
                                    .foregroundColor(.gray)
                            }
                        }
                    }
                    .listRowBackground(Color.white.opacity(0.05))
                }
                .scrollContentBackground(.hidden)
                .listStyle(.insetGrouped)
            }
        }
        .toolbar(.visible, for: .tabBar)
        .fullScreenCover(isPresented: $isShowingUpgrade) {
            UpgradeView()
        }
        .fullScreenCover(isPresented: $isShowingCreditPurchase) {
            CreditPurchaseView()
        }
        .alert("Restore Successful", isPresented: $showRestoreSuccess) {
            Button("OK", role: .cancel) { }
        } message: {
            Text("Your purchases have been restored successfully.")
        }
        .alert("Restore Failed", isPresented: $showRestoreError) {
            Button("OK", role: .cancel) { }
        } message: {
            Text(restoreError ?? "Unable to restore purchases. Please try again.")
        }
    }
    
    private func handleRestorePurchases() {
        isRestoring = true
        restoreError = nil
        
        Task {
            await subscriptionManager.restorePurchases()
            
            await MainActor.run {
                isRestoring = false
                if subscriptionManager.isPro {
                    showRestoreSuccess = true
                } else {
                    restoreError = "No active subscriptions found to restore."
                    showRestoreError = true
                }
            }
        }
    }
}

// MARK: - Scale Category View
// MARK: - Scale Reference Section
struct ScaleReferenceSection: View {
    let selectedScale: DisplayScale
    @State private var showDetails = false
    
    var body: some View {
        VStack(spacing: 0) {
            Button(action: {
                withAnimation {
                    showDetails.toggle()
                }
            }) {
                HStack {
                    VStack(alignment: .leading, spacing: 4) {
                        Text(selectedScale == .psl ? "PSL Scale (0-8)" : "1-10 Objective Scale")
                            .font(.headline)
                            .foregroundColor(.white)
                        Text(showDetails ? "Tap to hide details" : "Tap to view scale reference")
                            .font(.caption)
                            .foregroundColor(.gray)
                    }
                    Spacer()
                    Image(systemName: showDetails ? "chevron.up" : "chevron.down")
                        .foregroundColor(.gray)
                        .font(.caption)
                }
                .padding()
            }
            
            if showDetails {
                if selectedScale == .psl {
                    pslScaleContent
                } else {
                    objectiveScaleContent
                }
            }
        }
        .background(
            RoundedRectangle(cornerRadius: 12)
                .fill(Color.white.opacity(0.05))
        )
    }
    
    private var pslScaleContent: some View {
        VStack(alignment: .leading, spacing: 12) {
            ScaleCategoryView(title: "Subhuman", ranges: [
                "Below 0.5: Low Subhuman",
                "0.5 - 0.7: Subhuman",
                "0.8 - 1.3: High Subhuman"
            ])
            
            ScaleCategoryView(title: "Sub 5", ranges: [
                "1.4 - 1.8: LTN-/LTB-",
                "1.9 - 2.3: LTN/LTB",
                "2.4 - 2.7: LTN+/LTB+",
                "2.8 - 3.2: MTN-/MTB-",
                "3.3 - 4.5: MTN/MTB",
                "4.6 - 5.0: MTN+/MTB+"
            ])
            
            ScaleCategoryView(title: "HTN/HTB", ranges: [
                "5.1 - 5.3: HTN-/HTB-",
                "5.4 - 5.6: HTN/HTB",
                "5.7 - 5.9: HTN+/HTB+"
            ])
            
            ScaleCategoryView(title: "Chadlite/Stacylite", ranges: [
                "6.0 - 6.3: Low Chadlite/Stacylite",
                "6.4 - 6.5: Chadlite/Stacylite",
                "6.6 - 6.8: High Chadlite/Stacylite"
            ])
            
            ScaleCategoryView(title: "Chad/Stacy", ranges: [
                "6.9 - 7.0: Low Chad/Stacy",
                "7.1 - 7.2: Chad/Stacy",
                "7.3 - 7.4: High Chad/Stacy"
            ])
            
            ScaleCategoryView(title: "Adam/Eve", ranges: [
                "7.5 - 7.7: Adamlite/Evelite",
                "7.8 - 8.0: True Adam/Eve"
            ])
        }
        .padding(.horizontal)
        .padding(.bottom)
    }
    
    private var objectiveScaleContent: some View {
        VStack(alignment: .leading, spacing: 12) {
            ScaleCategoryView(title: "Subhuman", ranges: [
                "Below 0.5: Low Subhuman",
                "0.5 - 1.0: Subhuman",
                "1.1 - 1.4: High Subhuman"
            ])
            
            ScaleCategoryView(title: "Normie/Becky/Sub5", ranges: [
                "1.5 - 1.9: LTN-/LTB-",
                "2.0 - 2.5: LTN/LTB",
                "2.6 - 3.4: LTN+/LTB+",
                "3.5 - 3.9: MTN-/MTB-",
                "4.0 - 4.5: MTN/MTB",
                "4.6 - 5.1: MTN+/MTB+"
            ])
            
            ScaleCategoryView(title: "High-tier Normie", ranges: [
                "5.2 - 5.7: HTN-/HTB-",
                "5.8 - 6.0: HTN/HTB",
                "6.1 - 6.5: HTN+/HTB+"
            ])
            
            ScaleCategoryView(title: "Chadlite/Stacylite", ranges: [
                "6.6 - 6.9: Low Chadlite/Stacylite",
                "7.0 - 7.4: Chadlite/Stacylite",
                "7.5 - 7.7: High Chadlite/Stacylite"
            ])
            
            ScaleCategoryView(title: "Chad/Stacy", ranges: [
                "7.8 - 8.0: Low Chad/Stacy",
                "8.1 - 8.3: Chad/Stacy",
                "8.4 - 8.5: High Chad/True Stacy"
            ])
            
            ScaleCategoryView(title: "Terrachad/Terrastacy", ranges: [
                "8.6 - 9.4: Terrachad/Terrastacy"
            ])
            
            ScaleCategoryView(title: "Adam/Eve", ranges: [
                "9.5 - 9.6: Adamlite/Evelite",
                "9.7 - 9.9: True Adam/Eve"
            ])
        }
        .padding(.horizontal)
        .padding(.bottom)
    }
}

// MARK: - Scale Category View
struct ScaleCategoryView: View {
    let title: String
    let ranges: [String]
    
    var body: some View {
        VStack(alignment: .leading, spacing: 6) {
            Text(title)
                .font(.subheadline)
                .fontWeight(.semibold)
                .foregroundColor(.cyan)
            
            ForEach(ranges, id: \.self) { range in
                Text(range)
                    .font(.caption)
                    .foregroundColor(.gray)
                    .padding(.leading, 8)
            }
        }
        .padding(.vertical, 4)
    }
}

// MARK: - Rate Feature Row Component
struct RateFeatureRow: View {
    let icon: String
    let text: String
    let keyTerm: String
    
    // Detect if device is iPhone (not iPad)
    private var isIPhone: Bool {
        UIDevice.current.userInterfaceIdiom == .phone
    }
    
    var body: some View {
        HStack(spacing: isIPhone ? 12 : 16) {
            // SF Symbol icon with gradient background
            ZStack {
                Circle()
                    .fill(
                        LinearGradient(
                            colors: [Color.cyan.opacity(0.3), Color.purple.opacity(0.2)],
                            startPoint: .topLeading,
                            endPoint: .bottomTrailing
                        )
                    )
                    .frame(width: isIPhone ? 36 : 44, height: isIPhone ? 36 : 44)
                
                Image(systemName: icon)
                    .font(.system(size: isIPhone ? 14 : 18, weight: .semibold))
                    .foregroundStyle(
                        LinearGradient(
                            colors: [Color.cyan, Color.purple],
                            startPoint: .topLeading,
                            endPoint: .bottomTrailing
                        )
                    )
            }
            
            // Split text to bold the key term
            if let range = text.range(of: keyTerm) {
                let before = String(text[..<range.lowerBound])
                let key = String(text[range])
                let after = String(text[range.upperBound...])
                
                (Text(before) +
                 Text(key).fontWeight(.bold).foregroundColor(.cyan) +
                 Text(after))
                    .font(.system(size: isIPhone ? 12 : 16, weight: .medium))
                    .foregroundColor(.white)
                    .lineLimit(nil)
                    .fixedSize(horizontal: false, vertical: true)
            } else {
                Text(text)
                    .font(.system(size: isIPhone ? 12 : 16, weight: .medium))
                    .foregroundColor(.white)
                    .lineLimit(nil)
                    .fixedSize(horizontal: false, vertical: true)
            }
            
            Spacer()
        }
        .padding(.horizontal, isIPhone ? 14 : 18)
        .padding(.vertical, isIPhone ? 12 : 16)
        .background(
            RoundedRectangle(cornerRadius: isIPhone ? 12 : 16)
                .fill(
                    LinearGradient(
                        colors: [Color.white.opacity(0.1), Color.white.opacity(0.05)],
                        startPoint: .topLeading,
                        endPoint: .bottomTrailing
                    )
                )
                .overlay(
                    RoundedRectangle(cornerRadius: isIPhone ? 12 : 16)
                        .stroke(
                            LinearGradient(
                                colors: [Color.white.opacity(0.2), Color.white.opacity(0.1)],
                                startPoint: .topLeading,
                                endPoint: .bottomTrailing
                            ),
                            lineWidth: 1
                        )
                )
        )
        .shadow(color: .black.opacity(0.3), radius: 6, x: 0, y: 3)
    }
}

#Preview {
    ContentView()
}
