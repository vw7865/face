//
//  Sub5ToChadView.swift
//  LooksmaxAI
//
//  Created by Admin on 2026. 01. 15..
//

import SwiftUI

struct Sub5ToChadView: View {
    @Environment(\.dismiss) var dismiss
    @State private var showGenderSelection = false
    @State private var selectedGender: String? = nil
    @State private var frontImage: UIImage? = nil
    @State private var sideImage: UIImage? = nil
    @State private var isGenerating = false
    @State private var generatedChadImage: UIImage? = nil
    @State private var errorMessage: String? = nil
    @State private var showError = false
    @State private var showSaveSuccessAlert = false
    @StateObject private var usageTracker = UsageTracker.shared
    @StateObject private var subscriptionManager = SubscriptionManager.shared
    @State private var isShowingUpgrade = false
    @State private var isShowingCreditPurchase = false
    @State private var showOutOfCreditsActionSheet = false
    
    // Photo step enum
    private enum PhotoStep {
        case front
        case side
    }
    @State private var currentPhotoStep: PhotoStep? = nil
    
    var body: some View {
        NavigationStack {
            ZStack {
                Color.black.ignoresSafeArea()
                
                if isGenerating {
                    VStack(spacing: 20) {
                        ProgressView()
                            .scaleEffect(1.5)
                            .tint(.blue)
                        Text("Generating your Mogger version...")
                            .font(.headline)
                            .foregroundColor(.gray)
                        Text("This may take 30-60 seconds")
                            .font(.caption)
                            .foregroundColor(.gray.opacity(0.7))
                    }
                    .transition(.opacity)
                } else if generatedChadImage != nil {
                    // Show generated chad image
                    resultsView
                } else if !showGenderSelection {
                    // Show gender selection first
                    genderSelectionView
                } else if currentPhotoStep == nil || currentPhotoStep == .front {
                    // Show front photo selection
                    frontPhotoSelectionView
                } else if currentPhotoStep == .side {
                    // Show side profile selection
                    sidePhotoSelectionView
                }
            }
            .navigationTitle("Sub 5 to Mogger")
            .navigationBarTitleDisplayMode(.inline)
            .navigationBarBackButtonHidden(true)
            .toolbarVisibility(.hidden, for: .tabBar)
            .toolbar {
                // Credit balance display (visible to all users)
                ToolbarItem(placement: .navigationBarTrailing) {
                    CreditBalanceView(showCreditPurchase: $isShowingCreditPurchase)
                }
                
                // Only show back button when NOT showing results
                if generatedChadImage == nil {
                    ToolbarItem(placement: .navigationBarLeading) {
                        Button(action: {
                            if currentPhotoStep == .side {
                                // Go back to front photo selection
                                currentPhotoStep = .front
                                sideImage = nil
                            } else if currentPhotoStep == .front {
                                // Go back to gender selection
                                currentPhotoStep = nil
                                frontImage = nil
                            } else if showGenderSelection {
                                // Go back to gender selection
                                showGenderSelection = false
                            } else {
                                // Close view
                                dismiss()
                            }
                        }) {
                            HStack(spacing: 4) {
                                Image(systemName: "chevron.left")
                                Text("Back")
                            }
                            .foregroundColor(.white)
                        }
                    }
                }
                
                ToolbarItem(placement: .navigationBarTrailing) {
                    Button(action: {
                        resetView()
                    }) {
                        Text("Start Over")
                            .foregroundColor(.red)
                    }
                }
            }
            .onDisappear {
                NotificationCenter.default.post(name: NSNotification.Name("ShowTabBar"), object: nil)
            }
            .alert("Generation Failed", isPresented: $showError) {
                Button("OK", role: .cancel) { }
                Button("Retry") {
                    if let frontImg = frontImage, let sideImg = sideImage, let gender = selectedGender {
                        generateChadVersion(frontImage: frontImg, sideImage: sideImg, gender: gender)
                    }
                }
            } message: {
                Text(errorMessage ?? "An unknown error occurred. Please try again.")
            }
            .fullScreenCover(isPresented: $isShowingUpgrade) {
                UpgradeView()
            }
            .fullScreenCover(isPresented: $isShowingCreditPurchase) {
                CreditPurchaseView()
            }
            .confirmationDialog("Out of Credits", isPresented: $showOutOfCreditsActionSheet, titleVisibility: .visible) {
                Button("Purchase Credits") {
                    isShowingCreditPurchase = true
                }
                
                if !subscriptionManager.isPro {
                    Button("Upgrade to Pro") {
                        isShowingUpgrade = true
                    }
                }
                
                Button("Cancel", role: .cancel) { }
            } message: {
                let credits = usageTracker.getImageGenerationCreditsRemaining()
                if subscriptionManager.isPro {
                    Text("You have \(credits) credits remaining. Purchase more credits to continue generating images.")
                } else {
                    Text("You have \(credits) credits remaining. Pro subscribers get 30 credits/month, or you can purchase credits individually.")
                }
            }
        }
    }
    
    // MARK: - Upgrade Prompt View
    private var upgradePromptView: some View {
        VStack(spacing: 30) {
            Image(systemName: "crown.fill")
                .font(.system(size: 60))
                .foregroundColor(.yellow)
            
            Text("Pro Feature")
                .font(.title)
                .fontWeight(.bold)
                .foregroundColor(.white)
            
            Text("Sub 5 to Mogger is a Pro-only feature. Upgrade to Pro to see what you'd look like as a mogger!")
                .font(.body)
                .foregroundColor(.gray)
                .multilineTextAlignment(.center)
                .padding(.horizontal, 40)
            
            Button(action: {
                isShowingUpgrade = true
            }) {
                Text("Upgrade to Pro")
                    .font(.headline)
                    .foregroundColor(.white)
                    .padding(.horizontal, 32)
                    .padding(.vertical, 16)
                    .background(Color.red)
                    .cornerRadius(12)
            }
        }
        .frame(maxWidth: .infinity, maxHeight: .infinity)
        .background(Color.black)
    }
    
    // MARK: - Gender Selection View
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
                    showGenderSelection = true
                    currentPhotoStep = .front
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
                    showGenderSelection = true
                    currentPhotoStep = .front
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
        }
        .frame(maxWidth: .infinity, maxHeight: .infinity)
        .background(Color.black)
    }
    
    // MARK: - Front Photo Selection View
    private var frontPhotoSelectionView: some View {
        PhotoSelectionView(
            gender: selectedGender ?? "Male",
            title: "Take or Pick a front selfie",
            selectedImage: $frontImage,
            isSideProfile: false,
            onContinue: {
                // Move to side profile selection after front photo is selected
                if frontImage != nil {
                    currentPhotoStep = .side
                }
            }
        )
    }
    
    // MARK: - Side Photo Selection View
    private var sidePhotoSelectionView: some View {
        PhotoSelectionView(
            gender: selectedGender ?? "Male",
            title: "Take a side profile selfie",
            selectedImage: $sideImage,
            isSideProfile: true,
            onContinue: {
                // Generate chad version after both photos are selected (check credits)
                if let frontImg = frontImage, let sideImg = sideImage, let gender = selectedGender {
                    generateChadVersion(frontImage: frontImg, sideImage: sideImg, gender: gender)
                }
            }
        )
    }
    
    // MARK: - Results View
    private var resultsView: some View {
        ScrollView {
            VStack(spacing: 20) {
                if let chadImage = generatedChadImage {
                    Image(uiImage: chadImage)
                        .resizable()
                        .scaledToFit()
                        .frame(maxWidth: .infinity)
                        .cornerRadius(12)
                        .padding()
                    
                    Button(action: {
                        UIImageWriteToSavedPhotosAlbum(chadImage, nil, nil, nil)
                        showSaveSuccessAlert = true
                    }) {
                        HStack {
                            Image(systemName: "square.and.arrow.down")
                            Text("Save to Photos")
                        }
                        .font(.headline)
                        .foregroundColor(.white)
                        .frame(maxWidth: .infinity)
                        .padding()
                        .background(Color.blue)
                        .cornerRadius(10)
                    }
                    .padding(.horizontal)
                    .alert("Saved to Photos", isPresented: $showSaveSuccessAlert) {
                        Button("OK", role: .cancel) { }
                    } message: {
                        Text("Your Mogger image has been saved to your photo library.")
                    }
                }
            }
            .padding()
        }
    }
    
    // MARK: - Helper Functions
    private func generateChadVersion(frontImage: UIImage, sideImage: UIImage, gender: String) {
        // Check if user has credits
        guard usageTracker.canUseImageGeneration() else {
            showOutOfCreditsActionSheet = true
            return
        }
        
        // Deduct credit before generation
        guard usageTracker.useImageGenerationCredit() else {
            errorMessage = "Failed to deduct credit. Please try again."
            showError = true
            return
        }
        
        isGenerating = true
        errorMessage = nil
        
        ChadGenerationService.shared.generateChadVersion(
            frontImage: frontImage,
            sideImage: sideImage,
            gender: gender
        ) { result in
            DispatchQueue.main.async {
                isGenerating = false
                
                switch result {
                case .success(let chadImage):
                    generatedChadImage = chadImage
                case .failure(let error):
                    // Refund credit on failure
                    usageTracker.addImageGenerationCredits(1)
                    errorMessage = error.localizedDescription
                    showError = true
                }
            }
        }
    }
    
    private func resetView() {
        showGenderSelection = false
        selectedGender = nil
        currentPhotoStep = nil
        frontImage = nil
        sideImage = nil
        generatedChadImage = nil
        errorMessage = nil
        showError = false
    }
}

// MARK: - Credit Balance View
struct CreditBalanceView: View {
    @StateObject private var usageTracker = UsageTracker.shared
    @StateObject private var subscriptionManager = SubscriptionManager.shared
    @Binding var showCreditPurchase: Bool
    
    var body: some View {
        // Credits visible to all users (free users see "0 Credits")
        Button(action: {
            showCreditPurchase = true
        }) {
            HStack(spacing: 6) {
                Image(systemName: "sparkles")
                    .font(.system(size: 12))
                    .foregroundColor(.cyan)
                Text("\(usageTracker.getImageGenerationCreditsRemaining()) Credits")
                    .font(.system(size: 14, weight: .semibold))
                    .foregroundColor(.white)
            }
            .padding(.horizontal, 10)
            .padding(.vertical, 6)
            .background(
                Capsule()
                    .fill(Color.cyan.opacity(0.2))
                    .overlay(
                        Capsule()
                            .stroke(Color.cyan.opacity(0.4), lineWidth: 1)
                    )
            )
        }
        .buttonStyle(PlainButtonStyle())
    }
}

