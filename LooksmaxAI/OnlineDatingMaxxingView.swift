//
//  OnlineDatingMaxxingView.swift
//  LooksmaxAI
//
//  Created by Admin on 2026. 01. 14..
//

import SwiftUI
import UIKit

struct OnlineDatingMaxxingView: View {
    @Environment(\.dismiss) var dismiss
    @FocusState private var focusedField: Field?
    @State private var userPhoto: UIImage? = nil
    @State private var referenceImage: UIImage? = nil
    @State private var prompt: String = ""
    @State private var showUserPhotoPicker = false
    @State private var showReferenceImagePicker = false
    @State private var showImageSourceOptions = false
    @State private var imagePickerSourceType: UIImagePickerController.SourceType = .photoLibrary
    @State private var isSelectingUserPhoto = true  // Track which picker we're showing
    @State private var isGenerating = false
    @State private var showResults = false
    @State private var generatedImage: UIImage? = nil
    @State private var errorMessage: String? = nil
    @State private var isEditingMode = false
    @State private var editPrompt: String = ""
    @State private var originalUserPhoto: UIImage? = nil  // Store original for re-swapping
    @State private var originalReferenceImage: UIImage? = nil  // Store original reference
    @State private var originalPrompt: String = ""  // Store original prompt
    @State private var wasFaceSwap: Bool = false  // Track if this was a face swap (2 images)
    @State private var selectedMode: DatingMode? = nil  // Track which mode user selected
    @State private var swapType: SwapType = .faceAndBody  // Track swap type for full body swap
    @State private var singleImageSwapType: SwapType = .faceAndBody  // Swap type for single image mode
    @State private var sceneDescription: String = ""  // Scene description for single image mode
    @State private var matchClothingToScene: Bool = true  // Toggle for matching clothing to scene (single image)
    @State private var clothingSource: ClothingSource = .myImage  // Clothing source for full body swap
    @State private var additionalComments: String = ""  // Additional comments for full body swap
    @State private var showSaveSuccessAlert = false  // Show alert when image is saved
    @StateObject private var usageTracker = UsageTracker.shared
    @StateObject private var subscriptionManager = SubscriptionManager.shared
    @State private var isShowingUpgrade = false
    @State private var isShowingCreditPurchase = false
    @State private var showOutOfCreditsActionSheet = false
    
    enum Field {
        case sceneDescription
        case additionalComments
        case editPrompt
    }
    
    private func dismissKeyboard() {
        UIApplication.shared.sendAction(#selector(UIResponder.resignFirstResponder), to: nil, from: nil, for: nil)
    }
    
    enum ClothingSource: String, CaseIterable {
        case myImage = "My image"
        case referenceImage = "The reference image"
    }
    
    enum DatingMode {
        case singleImage  // Generate image using your face
        case fullBodySwap  // Full body swap with reference
    }
    
    enum SwapType: String, CaseIterable {
        case faceOnly = "Face Only"
        case faceAndBody = "Face + Body"
        
        var prompt: String {
            switch self {
            case .faceOnly:
                return "Replace only the face of the person in the reference image with the face/identity from my uploaded selfie. Keep the original body, head size, pose, clothing, background, lighting, and environment exactly the same. Preserve the identity and proportions of the selfie face: same face shape, jawline, nose size, eye spacing, and mouth shape. Only adjust angle and expression to match the reference. Do not stretch, shrink, smooth, slim, or beautify the face. Photorealistic, natural, high detail, no artifacts, no distortion, no deformed or melted features."
            case .faceAndBody:
                return "Replace the face and visible body of the person in the reference image with the identity and body build from my uploaded selfie. Keep the original pose, clothing, background, lighting, and environment exactly the same. Match the body proportions and overall build from my selfie naturally and accurately — no exaggeration or reshaping beyond what is needed to align to the pose. Preserve my selfie identity: same face shape, jawline, nose size, eye spacing, mouth shape, and natural body features. Only adjust angle and expression to match the reference pose. Do not invent muscles, slim the body, or stylize or beautify the face or physique. Photorealistic, natural skin texture, high detail, clean edges, no distortion or artifacts."
            }
        }
    }
    
    var body: some View {
        NavigationStack {
            ZStack {
                Color(red: 0.1, green: 0.1, blue: 0.12).ignoresSafeArea() // Dark gray instead of pure black
                
                if selectedMode == nil {
                    modeSelectionView
                } else if !showResults {
                    inputView
                } else {
                    resultsView
                }
            }
            .navigationTitle("")
            .navigationBarTitleDisplayMode(.inline)
            .navigationBarBackButtonHidden(true)
            .toolbarVisibility(.hidden, for: .tabBar)
            .toolbar {
                // Credit balance display (visible to all users)
                ToolbarItem(placement: .navigationBarTrailing) {
                    CreditBalanceView(showCreditPurchase: $isShowingCreditPurchase)
                }
                
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
            .sheet(isPresented: $showUserPhotoPicker) {
                ImagePicker(selectedImage: $userPhoto, sourceType: imagePickerSourceType)
            }
            .sheet(isPresented: $showReferenceImagePicker) {
                ImagePicker(selectedImage: $referenceImage, sourceType: imagePickerSourceType)
            }
            .confirmationDialog("Select Photo Source", isPresented: $showImageSourceOptions, titleVisibility: .visible) {
                Button("Photo Library") {
                    imagePickerSourceType = .photoLibrary
                    if isSelectingUserPhoto {
                        showUserPhotoPicker = true
                    } else {
                        showReferenceImagePicker = true
                    }
                }
                Button("Camera") {
                    imagePickerSourceType = .camera
                    if isSelectingUserPhoto {
                        showUserPhotoPicker = true
                    } else {
                        showReferenceImagePicker = true
                    }
                }
                Button("Cancel", role: .cancel) { }
            }
            .alert("Image Saved", isPresented: $showSaveSuccessAlert) {
                Button("OK", role: .cancel) {}
            } message: {
                Text("The image has been saved to your Photos library.")
            }
        }
    }
    
    private var upgradePromptView: some View {
        VStack(spacing: 30) {
            Image(systemName: "crown.fill")
                .font(.system(size: 60))
                .foregroundColor(.yellow)
            
            Text("Pro Feature")
                .font(.title)
                .fontWeight(.bold)
                .foregroundColor(.white)
            
            Text("Tindermaxxing is a Pro-only feature. Upgrade to Pro to create stunning dating profile photos with AI!")
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
        .background(Color(red: 0.1, green: 0.1, blue: 0.12))
    }
    
    private var modeSelectionView: some View {
        VStack(spacing: 24) {
            // Heading
            Text("Create Dating Photos")
                .font(.system(size: 32, weight: .semibold, design: .rounded))
                .foregroundColor(.white)
                .padding(.horizontal, 24)
                .padding(.vertical, 16)
                .background(
                    RoundedRectangle(cornerRadius: 20)
                        .fill(Color.white.opacity(0.1))
                        .overlay(
                            RoundedRectangle(cornerRadius: 20)
                                .stroke(
                                    LinearGradient(
                                        colors: [Color.cyan.opacity(0.6), Color.purple.opacity(0.6)],
                                        startPoint: .leading,
                                        endPoint: .trailing
                                    ),
                                    lineWidth: 2
                                )
                        )
                )
                .shadow(color: .black.opacity(0.3), radius: 8, x: 0, y: 4)
                .padding(.top, 20)
                .padding(.horizontal)
            
            VStack(spacing: 24) { // Increased spacing between options
                Button(action: {
                    selectedMode = .singleImage
                    singleImageSwapType = .faceAndBody  // Default to Face + Body
                    sceneDescription = ""  // Reset scene description
                    matchClothingToScene = true  // Default to matching clothing
                }) {
                    VStack(spacing: 12) {
                        // Before -> After image preview
                        HStack(spacing: 16) {
                            Image("before")
                                .resizable()
                                .aspectRatio(contentMode: .fit)
                                .frame(width: 80, height: 80)
                                .clipShape(RoundedRectangle(cornerRadius: 12))
                                .overlay(
                                    RoundedRectangle(cornerRadius: 12)
                                        .stroke(Color.white.opacity(0.2), lineWidth: 1)
                                )
                                .shadow(color: .black.opacity(0.3), radius: 4, x: 0, y: 2)
                            
                            Image(systemName: "arrow.right")
                                .font(.system(size: 24, weight: .semibold))
                                .foregroundColor(.cyan)
                            
                            Image("After")
                                .resizable()
                                .aspectRatio(contentMode: .fit)
                                .frame(width: 80, height: 80)
                                .clipShape(RoundedRectangle(cornerRadius: 12))
                                .overlay(
                                    RoundedRectangle(cornerRadius: 12)
                                        .stroke(Color.white.opacity(0.2), lineWidth: 1)
                                )
                                .shadow(color: .black.opacity(0.3), radius: 4, x: 0, y: 2)
                        }
                        .padding(.vertical, 8)
                        
                        Text("Generate with Your Face")
                            .font(.system(size: 18, weight: .bold))
                            .foregroundColor(.white)
                        
                        Text("Upload your photo and describe the scene you want")
                            .font(.system(size: 13, weight: .medium))
                            .foregroundColor(.gray.opacity(0.9))
                            .multilineTextAlignment(.center)
                    }
                    .frame(maxWidth: .infinity)
                    .frame(height: 180)
                    .background(
                        RoundedRectangle(cornerRadius: 20)
                            .fill(Color.white.opacity(0.1))
                            .overlay(
                                RoundedRectangle(cornerRadius: 20)
                                    .stroke(
                                        LinearGradient(
                                            colors: [Color(red: 0.42, green: 0.39, blue: 1.0).opacity(0.8), Color.cyan.opacity(0.6)],
                                            startPoint: .leading,
                                            endPoint: .trailing
                                        ),
                                        lineWidth: 2
                                    )
                            )
                            .shadow(color: Color(red: 0.42, green: 0.39, blue: 1.0).opacity(0.3), radius: 8, x: 0, y: 4)
                    )
                }
                
                Button(action: {
                    selectedMode = .fullBodySwap
                    swapType = .faceAndBody  // Default to Face + Body
                    // Set prompt based on swap type
                    prompt = swapType.prompt
                }) {
                    VStack(spacing: 12) {
                        // Before + Second -> After2 image preview
                        HStack(spacing: 12) {
                            Image("before")
                                .resizable()
                                .aspectRatio(contentMode: .fit)
                                .frame(width: 60, height: 60)
                                .clipShape(RoundedRectangle(cornerRadius: 12))
                                .overlay(
                                    RoundedRectangle(cornerRadius: 12)
                                        .stroke(Color.white.opacity(0.2), lineWidth: 1)
                                )
                                .shadow(color: .black.opacity(0.3), radius: 4, x: 0, y: 2)
                            
                            Text("+")
                                .font(.system(size: 20, weight: .bold))
                                .foregroundColor(.white)
                            
                            Image("Second")
                                .resizable()
                                .aspectRatio(contentMode: .fit)
                                .frame(width: 60, height: 60)
                                .clipShape(RoundedRectangle(cornerRadius: 12))
                                .overlay(
                                    RoundedRectangle(cornerRadius: 12)
                                        .stroke(Color.white.opacity(0.2), lineWidth: 1)
                                )
                                .shadow(color: .black.opacity(0.3), radius: 4, x: 0, y: 2)
                            
                            Image(systemName: "arrow.right")
                                .font(.system(size: 20, weight: .semibold))
                                .foregroundColor(.purple)
                            
                            Image("After2")
                                .resizable()
                                .aspectRatio(contentMode: .fit)
                                .frame(width: 60, height: 60)
                                .clipShape(RoundedRectangle(cornerRadius: 12))
                                .overlay(
                                    RoundedRectangle(cornerRadius: 12)
                                        .stroke(Color.white.opacity(0.2), lineWidth: 1)
                                )
                                .shadow(color: .black.opacity(0.3), radius: 4, x: 0, y: 2)
                        }
                        .padding(.vertical, 8)
                        
                        Text("Full Body Swap")
                            .font(.system(size: 18, weight: .bold))
                            .foregroundColor(.white)
                        
                        Text("Upload your photo and a reference image to replace the person")
                            .font(.system(size: 13, weight: .medium))
                            .foregroundColor(.gray.opacity(0.9))
                            .multilineTextAlignment(.center)
                    }
                    .frame(maxWidth: .infinity)
                    .frame(height: 180)
                    .background(
                        RoundedRectangle(cornerRadius: 20)
                            .fill(Color.white.opacity(0.1))
                            .overlay(
                                RoundedRectangle(cornerRadius: 20)
                                    .stroke(
                                        LinearGradient(
                                            colors: [Color.purple.opacity(0.8), Color(red: 0.42, green: 0.39, blue: 1.0).opacity(0.6)],
                                            startPoint: .leading,
                                            endPoint: .trailing
                                        ),
                                        lineWidth: 2
                                    )
                            )
                            .shadow(color: Color.purple.opacity(0.3), radius: 8, x: 0, y: 4)
                    )
                }
            }
            .padding(.horizontal)
            .padding(.top, 40)
        }
    }
    
    private var inputView: some View {
        ScrollView {
            VStack(spacing: 24) {
                userPhotoSection
                
                if selectedMode == .fullBodySwap {
                    swapTypeSection
                    referenceImageSection
                    if swapType == .faceAndBody {
                        clothingSourceSection
                    }
                    additionalCommentsSection
                } else if selectedMode == .singleImage {
                    singleImageSwapTypeSection
                    sceneDescriptionSection
                    clothingToggleSection
                }
                
                generateButton
                
                if let error = errorMessage {
                    errorView(error)
                        .padding(.horizontal)
                }
            }
            .padding(.bottom, 40)
        }
        .onTapGesture {
            dismissKeyboard()
        }
    }
    
    
    private var userPhotoSection: some View {
        VStack(alignment: .leading, spacing: 12) {
            Text("Your Photo *")
                .font(.headline)
                .foregroundColor(.white)
            
            Text("Upload a photo of yourself (whole body, waist up, or face selfie)")
                .font(.caption)
                .foregroundColor(.gray)
            
            if let photo = userPhoto {
                HStack {
                    Image(uiImage: photo)
                        .resizable()
                        .scaledToFill()
                        .frame(width: 100, height: 100)
                        .cornerRadius(12)
                        .clipped()
                    
                    Button("Change Photo") {
                        isSelectingUserPhoto = true
                        showImageSourceOptions = true
                    }
                    .foregroundColor(.cyan)
                    
                    Spacer()
                }
            } else {
                Button(action: {
                    isSelectingUserPhoto = true
                    showImageSourceOptions = true
                }) {
                    VStack(spacing: 12) {
                        Image(systemName: "photo.badge.plus")
                            .font(.system(size: 40))
                            .foregroundColor(.gray)
                        Text("Tap to Upload Your Photo")
                            .font(.subheadline)
                            .foregroundColor(.gray)
                    }
                    .frame(maxWidth: .infinity)
                    .frame(height: 120)
                    .background(
                        RoundedRectangle(cornerRadius: 12)
                            .stroke(style: StrokeStyle(lineWidth: 2, dash: [5]))
                            .foregroundColor(.gray.opacity(0.5))
                    )
                }
            }
        }
        .padding()
        .background(sectionBackground)
    }
    
    private var swapTypeSection: some View {
        VStack(alignment: .leading, spacing: 12) {
            Text("Swap Type")
                .font(.headline)
                .foregroundColor(.white)
            
            Text("Choose what to replace in the reference image")
                .font(.caption)
                .foregroundColor(.gray)
            
            Picker("Swap Type", selection: $swapType) {
                ForEach(SwapType.allCases, id: \.self) { type in
                    Text(type.rawValue).tag(type)
                }
            }
            .pickerStyle(.segmented)
            .onChange(of: swapType) { oldValue, newValue in
                // Update prompt when swap type changes
                prompt = newValue.prompt
            }
        }
        .padding()
        .background(sectionBackground)
    }
    
    private var referenceImageSection: some View {
        VStack(alignment: .leading, spacing: 12) {
            Text("Reference Image *")
                .font(.headline)
                .foregroundColor(.white)
            
            Text("Upload a reference image to replace the person with yourself")
                .font(.caption)
                .foregroundColor(.gray)
            
            if let refImage = referenceImage {
                HStack {
                    Image(uiImage: refImage)
                        .resizable()
                        .scaledToFill()
                        .frame(width: 100, height: 100)
                        .cornerRadius(12)
                        .clipped()
                    
                    Button("Change") {
                        isSelectingUserPhoto = false
                        showImageSourceOptions = true
                    }
                    .foregroundColor(.cyan)
                    
                    Button("Remove") {
                        referenceImage = nil
                    }
                    .foregroundColor(.red)
                    
                    Spacer()
                }
            } else {
                Button(action: {
                    isSelectingUserPhoto = false
                    showImageSourceOptions = true
                }) {
                    VStack(spacing: 12) {
                        Image(systemName: "photo")
                            .font(.system(size: 30))
                            .foregroundColor(.gray)
                        Text("Tap to Upload Reference Image")
                            .font(.subheadline)
                            .foregroundColor(.gray)
                    }
                    .frame(maxWidth: .infinity)
                    .frame(height: 100)
                    .background(
                        RoundedRectangle(cornerRadius: 12)
                            .stroke(style: StrokeStyle(lineWidth: 2, dash: [5]))
                            .foregroundColor(.gray.opacity(0.5))
                    )
                }
            }
        }
        .padding()
        .background(sectionBackground)
    }
    
    // MARK: - Single Image Mode Sections
    private var singleImageSwapTypeSection: some View {
        VStack(alignment: .leading, spacing: 12) {
            Text("Swap Type")
                .font(.headline)
                .foregroundColor(.white)
            
            Text("Choose what to include from your photo")
                .font(.caption)
                .foregroundColor(.gray)
            
            Picker("Swap Type", selection: $singleImageSwapType) {
                ForEach(SwapType.allCases, id: \.self) { type in
                    Text(type.rawValue).tag(type)
                }
            }
            .pickerStyle(.segmented)
        }
        .padding()
        .background(sectionBackground)
    }
    
    private var sceneDescriptionSection: some View {
        VStack(alignment: .leading, spacing: 12) {
            Text("Describe Scene *")
                .font(.headline)
                .foregroundColor(.white)
            
            Text("Describe the scene you want (e.g., 'coffee shop', 'beach at sunset', 'professional office')")
                .font(.caption)
                .foregroundColor(.gray)
            
            TextField("e.g., 'coffee shop', 'beach at sunset', 'professional office'", text: $sceneDescription, axis: .vertical)
                .textFieldStyle(.plain)
                .padding(12)
                .background(Color.white.opacity(0.1))
                .cornerRadius(12)
                .foregroundColor(.white)
                .lineLimit(2...4)
                .focused($focusedField, equals: .sceneDescription)
                .submitLabel(.done)
                .onSubmit {
                    dismissKeyboard()
                }
        }
        .padding()
        .background(sectionBackground)
    }
    
    private var clothingToggleSection: some View {
        VStack(alignment: .leading, spacing: 12) {
            Toggle(isOn: $matchClothingToScene) {
                VStack(alignment: .leading, spacing: 4) {
                    Text("Match clothing to scene")
                        .font(.headline)
                        .foregroundColor(.white)
                    Text("Automatically change clothing to match the scene (e.g., swimwear for pool)")
                        .font(.caption)
                        .foregroundColor(.gray)
                }
            }
            .tint(.cyan)
        }
        .padding()
        .background(sectionBackground)
    }
    
    // MARK: - Full Body Swap Mode Sections
    private var clothingSourceSection: some View {
        VStack(alignment: .leading, spacing: 12) {
            Text("Use clothing from:")
                .font(.headline)
                .foregroundColor(.white)
            
            Text("Choose which clothing to use in the final image")
                .font(.caption)
                .foregroundColor(.gray)
            
            Picker("Clothing Source", selection: $clothingSource) {
                ForEach(ClothingSource.allCases, id: \.self) { source in
                    Text(source.rawValue).tag(source)
                }
            }
            .pickerStyle(.segmented)
        }
        .padding()
        .background(sectionBackground)
    }
    
    private var additionalCommentsSection: some View {
        VStack(alignment: .leading, spacing: 12) {
            Text("Additional Instructions")
                .font(.headline)
                .foregroundColor(.white)
            
            Text("Optional: Add any specific instructions (e.g., 'Use the person on the left', 'Keep the dog in the background')")
                .font(.caption)
                .foregroundColor(.gray)
            
            TextField("Optional: Add any specific instructions...", text: $additionalComments, axis: .vertical)
                .textFieldStyle(.plain)
                .padding(12)
                .background(Color.white.opacity(0.1))
                .cornerRadius(12)
                .foregroundColor(.white)
                .lineLimit(2...4)
                .focused($focusedField, equals: .additionalComments)
                .submitLabel(.done)
                .onSubmit {
                    dismissKeyboard()
                }
        }
        .padding()
        .background(sectionBackground)
    }
    
    private var generateButton: some View {
        VStack(spacing: 12) {
            // Show generate button for all users (credits check happens in generatePhoto)
            Button(action: {
                generatePhoto()
            }) {
                HStack {
                    if isGenerating {
                        ProgressView()
                            .progressViewStyle(CircularProgressViewStyle(tint: .white))
                    } else {
                        Image(systemName: "sparkles")
                            .font(.title3)
                    }
                    Text(isGenerating ? "Generating..." : "Generate Photo")
                        .font(.headline)
                        .foregroundColor(.white)
                }
                .frame(maxWidth: .infinity)
                .frame(height: 50)
                .background(
                    RoundedRectangle(cornerRadius: 12)
                        .fill(canGenerate ? Color.cyan : Color.gray.opacity(0.3))
                )
            }
            .disabled(!canGenerate || isGenerating)
            .padding(.horizontal)
        }
    }
    
    private var canGenerate: Bool {
        // For single image mode: need user photo and non-empty scene description
        if selectedMode == .singleImage {
            return userPhoto != nil && !sceneDescription.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty
        }
        // For full body swap: need both user photo AND reference image
        else if selectedMode == .fullBodySwap {
            return userPhoto != nil && referenceImage != nil
        }
        // Default: no mode selected, can't generate
        return false
    }
    
    private var resultsView: some View {
        ScrollView {
            VStack(spacing: 24) {
                Text("Generated Photo")
                    .font(.title2)
                    .fontWeight(.bold)
                    .foregroundColor(.white)
                    .padding(.top, 20)
                
                if let image = generatedImage {
                    Image(uiImage: image)
                        .resizable()
                        .scaledToFit()
                        .cornerRadius(16)
                        .padding()
                    
                    if isEditingMode {
                        // Edit mode: show prompt input
                        VStack(spacing: 16) {
                            Text("Edit Image")
                                .font(.headline)
                                .foregroundColor(.white)
                            
                            if wasFaceSwap {
                                // Show option to improve face swap or regular edit
                                VStack(spacing: 12) {
                                    Text("Face swap not good enough?")
                                        .font(.subheadline)
                                        .foregroundColor(.gray)
                                    
                                    Button(action: {
                                        // Re-do the face swap with improved prompt
                                        improveFaceSwap()
                                    }) {
                                        HStack {
                                            Image(systemName: "face.smiling")
                                            Text("Improve Face Swap")
                                        }
                                        .font(.subheadline)
                                        .foregroundColor(.white)
                                        .frame(maxWidth: .infinity)
                                        .frame(height: 44)
                                        .background(
                                            RoundedRectangle(cornerRadius: 12)
                                                .fill(Color.orange.opacity(0.8))
                                        )
                                    }
                                    
                                    Text("OR")
                                        .font(.caption)
                                        .foregroundColor(.gray)
                                        .padding(.vertical, 4)
                                }
                            }
                            
                            TextField("Describe what you want to change...", text: $editPrompt, axis: .vertical)
                                .textFieldStyle(.plain)
                                .padding()
                                .background(
                                    RoundedRectangle(cornerRadius: 12)
                                        .fill(Color.white.opacity(0.1))
                                )
                                .foregroundColor(.white)
                                .lineLimit(3...6)
                                .focused($focusedField, equals: .editPrompt)
                                .submitLabel(.done)
                                .onSubmit {
                                    dismissKeyboard()
                                }
                            
                            if wasFaceSwap {
                                Text("Examples: 'make face match better', 'fix face alignment', 'improve skin blending', 'adjust lighting'")
                                    .font(.caption)
                                    .foregroundColor(.gray)
                                    .padding(.horizontal)
                            }
                            
                            HStack(spacing: 12) {
                                Button(action: {
                                    isEditingMode = false
                                    editPrompt = ""
                                }) {
                                    Text("Cancel")
                                        .font(.headline)
                                        .foregroundColor(.white)
                                        .frame(maxWidth: .infinity)
                                        .frame(height: 50)
                                        .background(
                                            RoundedRectangle(cornerRadius: 12)
                                                .fill(Color.gray.opacity(0.3))
                                        )
                                }
                                
                                Button(action: {
                                    editGeneratedImage(image)
                                }) {
                                    HStack {
                                        if isGenerating {
                                            ProgressView()
                                                .progressViewStyle(CircularProgressViewStyle(tint: .white))
                                        } else {
                                            Image(systemName: "wand.and.stars")
                                            Text("Apply Edit")
                                        }
                                    }
                                    .font(.headline)
                                    .foregroundColor(.white)
                                    .frame(maxWidth: .infinity)
                                    .frame(height: 50)
                                    .background(
                                        RoundedRectangle(cornerRadius: 12)
                                            .fill(Color.purple)
                                    )
                                }
                                .disabled(isGenerating || editPrompt.trimmingCharacters(in: .whitespaces).isEmpty)
                            }
                        }
                        .padding(.horizontal)
                    } else {
                        // Normal mode: show action buttons
                        VStack(spacing: 12) {
                            Button(action: {
                                isEditingMode = true
                            }) {
                                HStack {
                                    Image(systemName: "pencil")
                                    Text("Edit Image")
                                }
                                .font(.headline)
                                .foregroundColor(.white)
                                .frame(maxWidth: .infinity)
                                .frame(height: 50)
                                .background(
                                    RoundedRectangle(cornerRadius: 12)
                                        .fill(Color.purple)
                                )
                            }
                            .padding(.horizontal)
                            
                            Button(action: {
                                saveImage(image)
                            }) {
                                HStack {
                                    Image(systemName: "square.and.arrow.down")
                                    Text("Save to Photos")
                                }
                                .font(.headline)
                                .foregroundColor(.white)
                                .frame(maxWidth: .infinity)
                                .frame(height: 50)
                                .background(
                                    RoundedRectangle(cornerRadius: 12)
                                        .fill(Color.cyan)
                                )
                            }
                            .padding(.horizontal)
                        }
                    }
                }
                
                if let error = errorMessage {
                    errorView(error)
                        .padding(.horizontal)
                }
                
                Button(action: {
                    resetView()
                }) {
                    Text("Generate Another")
                        .font(.headline)
                        .foregroundColor(.white)
                        .frame(maxWidth: .infinity)
                        .frame(height: 50)
                        .background(
                            RoundedRectangle(cornerRadius: 12)
                                .fill(Color.gray.opacity(0.3))
                        )
                }
                .padding(.horizontal)
            }
        }
    }
    
    private func errorView(_ message: String) -> some View {
        Text(message)
            .font(.subheadline)
            .foregroundColor(.red)
            .padding()
            .frame(maxWidth: .infinity, alignment: .leading)
            .background(
                RoundedRectangle(cornerRadius: 12)
                    .fill(Color.red.opacity(0.2))
            )
            .padding(.horizontal)
    }
    
    private var sectionBackground: some View {
        RoundedRectangle(cornerRadius: 16)
            .fill(Color.white.opacity(0.1))
            .overlay(
                RoundedRectangle(cornerRadius: 16)
                    .stroke(Color.white.opacity(0.2), lineWidth: 1)
            )
    }
    
    private func generatePhoto() {
        guard let userPhoto = userPhoto else { return }
        
        // Check if user has credits
        guard usageTracker.canUseImageGeneration() else {
            showOutOfCreditsActionSheet = true
            return
        }
        
        // Validate based on mode
        if selectedMode == .fullBodySwap && referenceImage == nil {
            errorMessage = "Please upload a reference image for full body swap"
            return
        }
        
        // Validate scene description for single image mode
        if selectedMode == .singleImage {
            let trimmedScene = sceneDescription.trimmingCharacters(in: .whitespacesAndNewlines)
            if trimmedScene.isEmpty {
                errorMessage = "Please describe the scene you want (e.g., 'coffee shop', 'beach at sunset')"
                return
            }
        }
        
        // Deduct credit before generation
        guard usageTracker.useImageGenerationCredit() else {
            errorMessage = "Failed to deduct credit. Please try again."
            return
        }
        
        isGenerating = true
        errorMessage = nil
        
        // Store originals for potential re-swapping/editing
        originalUserPhoto = userPhoto
        originalReferenceImage = referenceImage
        
        // Build prompt based on mode
        var finalPrompt = ""
        var swapTypeToUse: SwapType
        var clothingSourceToUse: ClothingSource?
        var matchClothing: Bool = false
        var comments: String = ""
        
        if selectedMode == .singleImage {
            swapTypeToUse = singleImageSwapType
            matchClothing = matchClothingToScene
            comments = ""  // Single image doesn't use comments field
            
            // Build prompt based on swap type
            if singleImageSwapType == .faceOnly {
                finalPrompt = "Take my face from the uploaded photo and place me in \(sceneDescription). Photorealistic, natural, high detail, no artifacts. Make sure there are proper body proportions."
            } else {
                finalPrompt = "Take my face and body from the uploaded photo and place me in \(sceneDescription). Photorealistic, natural, high detail, no artifacts. Make sure there are proper body proportions."
            }
        } else {
            // Full Body Swap
            swapTypeToUse = swapType
            clothingSourceToUse = swapType == .faceAndBody ? clothingSource : nil
            comments = additionalComments.trimmingCharacters(in: .whitespacesAndNewlines)
            
            // Use the base prompt from swap type
            finalPrompt = swapType.prompt
            
            // Add clothing instruction if Face + Body
            if swapType == .faceAndBody {
                if clothingSource == .referenceImage {
                    finalPrompt += " Keep the clothing from the reference image."
                } else {
                    finalPrompt += " Keep the clothing from my uploaded photo."
                }
            }
            
            // Add comments if provided
            if !comments.isEmpty {
                finalPrompt += " \(comments)"
            }
        }
        
        originalPrompt = finalPrompt
        wasFaceSwap = (selectedMode == .fullBodySwap)
        
        // Debug: Print the prompt being sent
        print("📝 Sending prompt to backend: \(finalPrompt)")
        print("📝 Prompt length: \(finalPrompt.count) characters")
        if selectedMode == .singleImage {
            print("📝 Match clothing to scene: \(matchClothing)")
        }
        if selectedMode == .fullBodySwap && swapType == .faceAndBody {
            print("📝 Clothing source: \(clothingSource.rawValue)")
        }
        
        DatingPhotoService.generatePhoto(
            userPhoto: userPhoto,
            referenceImage: referenceImage,
            prompt: finalPrompt,
            swapType: swapTypeToUse.rawValue,
            matchClothingToScene: matchClothing,
            clothingSource: clothingSourceToUse?.rawValue,
            comments: comments.isEmpty ? nil : comments
        ) { result in
            DispatchQueue.main.async {
                isGenerating = false
                
                switch result {
                case .success(let image):
                    generatedImage = image
                    showResults = true
                case .failure(let error):
                    // Refund credit on failure
                    usageTracker.addImageGenerationCredits(1)
                    errorMessage = error.localizedDescription
                }
            }
        }
    }
    
    private func resetView() {
        showResults = false
        generatedImage = nil
        errorMessage = nil
        isEditingMode = false
        editPrompt = ""
        originalUserPhoto = nil
        originalReferenceImage = nil
        originalPrompt = ""
        wasFaceSwap = false
        // Reset fields based on mode
        if selectedMode == .singleImage {
            sceneDescription = ""
            matchClothingToScene = true
            singleImageSwapType = .faceAndBody
        } else {
            additionalComments = ""
            clothingSource = .myImage
            swapType = .faceAndBody
        }
    }
    
    private func resetToModeSelection() {
        showResults = false
        generatedImage = nil
        errorMessage = nil
        isEditingMode = false
        editPrompt = ""
        userPhoto = nil
        referenceImage = nil
        sceneDescription = ""
        additionalComments = ""
        matchClothingToScene = true
        singleImageSwapType = .faceAndBody
        swapType = .faceAndBody
        clothingSource = .myImage
        originalUserPhoto = nil
        originalReferenceImage = nil
        originalPrompt = ""
        wasFaceSwap = false
        selectedMode = nil
    }
    
    private func editGeneratedImage(_ image: UIImage) {
        guard !editPrompt.trimmingCharacters(in: .whitespaces).isEmpty else { return }
        
        isGenerating = true
        errorMessage = nil
        
        // Use the generated image as the new "user photo" for editing
        // No reference image needed - just edit the generated image directly
        DatingPhotoService.generatePhoto(
            userPhoto: image,
            referenceImage: nil,  // No reference when editing
            prompt: editPrompt
        ) { result in
            DispatchQueue.main.async {
                isGenerating = false
                
                switch result {
                case .success(let editedImage):
                    generatedImage = editedImage
                    isEditingMode = false
                    editPrompt = ""
                case .failure(let error):
                    errorMessage = error.localizedDescription
                }
            }
        }
    }
    
    private func improveFaceSwap() {
        guard let originalUser = originalUserPhoto,
              let originalRef = originalReferenceImage else {
            errorMessage = "Original images not available. Please generate again."
            return
        }
        
        isGenerating = true
        errorMessage = nil
        isEditingMode = false
        
        // Re-do the face swap with an improved prompt that emphasizes better face matching
        let improvedPrompt = "\(originalPrompt). Improve the face and body swap - make the face from my photo match perfectly with the reference image. Ensure natural skin blending, proper face alignment, realistic lighting match, and seamless integration. High quality, photorealistic, no artifacts."
        
        DatingPhotoService.generatePhoto(
            userPhoto: originalUser,
            referenceImage: originalRef,
            prompt: improvedPrompt
        ) { result in
            DispatchQueue.main.async {
                isGenerating = false
                
                switch result {
                case .success(let improvedImage):
                    generatedImage = improvedImage
                    // Keep the originals for potential further improvements
                case .failure(let error):
                    errorMessage = error.localizedDescription
                }
            }
        }
    }
    
    private func saveImage(_ image: UIImage) {
        UIImageWriteToSavedPhotosAlbum(image, nil, nil, nil)
        // Show success alert
        showSaveSuccessAlert = true
    }
}

