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
    
    var body: some View {
        TabView(selection: $selectedTab) {
            RateView(onAnalysisComplete: { results in
                // Store results and switch tab
                print("🔄 onAnalysisComplete called with results, PSL: \(results.overall.psl ?? 0)")
                directResults = results
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
                        print("🔄 directResults set to nil, should show ResultsListView now")
                    })
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
    }
}

enum PhotoStep {
    case front
    case side
}

struct RateView: View {
    let onAnalysisComplete: (FaceAnalysisResults) -> Void
    @State private var showGenderSelection = false
    @State private var selectedGender: String? = nil
    @State private var currentPhotoStep: PhotoStep? = nil
    @State private var frontImage: UIImage? = nil
    @State private var sideImage: UIImage? = nil
    @State private var isAnalyzing = false
    @State private var analysisResults: FaceAnalysisResults? = nil
    @State private var errorMessage: String? = nil
    @State private var showError = false
    @AppStorage("lastAnalysisResults") private var lastAnalysisResultsData: Data?
    
    var body: some View {
        NavigationStack {
            ZStack {
                if isAnalyzing {
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
                        Text("(Server is starting up)")
                            .font(.caption2)
                            .foregroundColor(.gray.opacity(0.5))
                    }
                    .transition(.opacity)
                } else if !showGenderSelection && currentPhotoStep == nil {
                    VStack {
                        Text("Get Your Face Rating")
                            .font(.largeTitle)
                            .fontWeight(.bold)
                            .padding()
                        
                        Button(action: {
                            showGenderSelection = true
                        }) {
                            Text("Begin")
                                .font(.headline)
                                .foregroundColor(.white)
                                .frame(maxWidth: .infinity)
                                .padding()
                                .background(Color.blue)
                                .cornerRadius(10)
                        }
                        .padding(.horizontal)
                    }
                } else if showGenderSelection && currentPhotoStep == nil {
                    VStack(spacing: 20) {
                        Text("Select Your Gender")
                            .font(.title2)
                            .fontWeight(.semibold)
                            .padding(.top)
                        
                        HStack(spacing: 20) {
                            Button(action: {
                                selectedGender = "Male"
                            }) {
                                Text("Male")
                                    .font(.headline)
                                    .foregroundColor(.white)
                                    .frame(maxWidth: .infinity)
                                    .padding()
                                    .background(selectedGender == "Male" ? Color.blue : Color.gray)
                                    .cornerRadius(10)
                            }
                            
                            Button(action: {
                                selectedGender = "Female"
                            }) {
                                Text("Female")
                                    .font(.headline)
                                    .foregroundColor(.white)
                                    .frame(maxWidth: .infinity)
                                    .padding()
                                    .background(selectedGender == "Female" ? Color.blue : Color.gray)
                                    .cornerRadius(10)
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
                } else if currentPhotoStep == .front {
                    PhotoSelectionView(
                        gender: selectedGender ?? "Male",
                        title: "Take or Pick a front selfie",
                        selectedImage: $frontImage,
                        onContinue: {
                            currentPhotoStep = .side
                        },
                        isSideProfile: false
                    )
                } else if currentPhotoStep == .side {
                    PhotoSelectionView(
                        gender: selectedGender ?? "Male",
                        title: "Take a side profile selfie",
                        selectedImage: $sideImage,
                        onContinue: {
                            analyzeFaces()
                        },
                        isSideProfile: true
                    )
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
            
            // Save results first
            analysisResults = results
            saveResults(results)
            ResultHistoryManager.shared.saveResult(results, frontImage: frontImage)
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
                self.onAnalysisComplete(results)
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
    let onContinue: () -> Void
    let isSideProfile: Bool
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
                Image(systemName: gender == "Male" ? "person.fill" : "person.fill")
                    .resizable()
                    .scaledToFit()
                    .frame(width: 200, height: 200)
                    .foregroundColor(.gray)
                    .overlay(
                        RoundedRectangle(cornerRadius: 10)
                            .stroke(Color.gray, lineWidth: 2)
                    )
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
                            Text("Continue")
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
    var body: some View {
        NavigationStack {
            ZStack {
                Color.black.ignoresSafeArea()
                
                ScrollView {
                    VStack(spacing: 20) {
                        // Online Dating Maxxing Button
                        Button(action: {
                            // Placeholder: Online Dating Maxxing
                            print("Online Dating Maxxing tapped")
                        }) {
                            VStack(alignment: .leading, spacing: 12) {
                                HStack {
                                    Image(systemName: "heart.fill")
                                        .font(.title2)
                                        .foregroundColor(.pink)
                                    Text("Online Dating Maxxing")
                                        .font(.title2)
                                        .fontWeight(.bold)
                                        .foregroundColor(.white)
                                    Spacer()
                                }
                                
                                Text("Upload your photo of yourself and create Neurotypical or \"NT\" pics for your dating profile.")
                                    .font(.body)
                                    .foregroundColor(.gray)
                                    .multilineTextAlignment(.leading)
                            }
                            .padding()
                            .frame(maxWidth: .infinity, alignment: .leading)
                            .background(
                                RoundedRectangle(cornerRadius: 16)
                                    .fill(Color.white.opacity(0.1))
                                    .overlay(
                                        RoundedRectangle(cornerRadius: 16)
                                            .stroke(Color.white.opacity(0.2), lineWidth: 1)
                                    )
                            )
                        }
                        .buttonStyle(PlainButtonStyle())
                        
                        // RizzMaxxing Button
                        NavigationLink(destination: RizzMaxxingView()) {
                            VStack(alignment: .leading, spacing: 12) {
                                HStack {
                                    Image(systemName: "message.fill")
                                        .font(.title2)
                                        .foregroundColor(.cyan)
                                    Text("RizzMaxxing")
                                        .font(.title2)
                                        .fontWeight(.bold)
                                        .foregroundColor(.white)
                                    Spacer()
                                    Image(systemName: "chevron.right")
                                        .foregroundColor(.gray)
                                }
                                
                                Text("Having problems talking to girls? Is she leaving you on opened on Snap? Get help now.")
                                    .font(.body)
                                    .foregroundColor(.gray)
                                    .multilineTextAlignment(.leading)
                            }
                            .padding()
                            .frame(maxWidth: .infinity, alignment: .leading)
                            .background(
                                RoundedRectangle(cornerRadius: 16)
                                    .fill(Color.white.opacity(0.1))
                                    .overlay(
                                        RoundedRectangle(cornerRadius: 16)
                                            .stroke(Color.white.opacity(0.2), lineWidth: 1)
                                    )
                            )
                        }
                        .buttonStyle(PlainButtonStyle())
                        
                        // Looksmaxxing Button
                        Button(action: {
                            // Placeholder: Looksmaxxing
                            print("Looksmaxxing tapped")
                        }) {
                            VStack(alignment: .leading, spacing: 12) {
                                HStack {
                                    Image(systemName: "sparkles")
                                        .font(.title2)
                                        .foregroundColor(.purple)
                                    Text("Looksmaxxing")
                                        .font(.title2)
                                        .fontWeight(.bold)
                                        .foregroundColor(.white)
                                    Spacer()
                                }
                                
                                Text("Truth about your looks and how to improve and ascend (if needed).")
                                    .font(.body)
                                    .foregroundColor(.gray)
                                    .multilineTextAlignment(.leading)
                            }
                            .padding()
                            .frame(maxWidth: .infinity, alignment: .leading)
                            .background(
                                RoundedRectangle(cornerRadius: 16)
                                    .fill(Color.white.opacity(0.1))
                                    .overlay(
                                        RoundedRectangle(cornerRadius: 16)
                                            .stroke(Color.white.opacity(0.2), lineWidth: 1)
                                    )
                            )
                        }
                        .buttonStyle(PlainButtonStyle())
                    }
                    .padding()
                }
            }
        }
    }
}

struct SettingsView: View {
    @StateObject private var scaleManager = ScaleManager.shared
    
    var body: some View {
        NavigationView {
            List {
                Section(header: Text("Display Scale")) {
                    Picker("Scale", selection: $scaleManager.selectedScale) {
                        ForEach(DisplayScale.allCases, id: \.self) { scale in
                            Text(scale.displayName).tag(scale)
                        }
                    }
                    .pickerStyle(.segmented)
                    
                    Text("Choose how PSL scores are displayed:")
                        .font(.caption)
                        .foregroundColor(.gray)
                        .padding(.top, 4)
                    
                    VStack(alignment: .leading, spacing: 8) {
                        Text("• PSL Scale (0-8): Traditional looksmax scale")
                            .font(.caption)
                            .foregroundColor(.gray)
                        Text("• 1-10 Objective Scale: More intuitive rating")
                            .font(.caption)
                            .foregroundColor(.gray)
                    }
                    .padding(.top, 8)
                }
                
                Section(header: Text("About")) {
                    Text("PSL Scale is the traditional looksmax rating system (0-8 range).")
                        .font(.caption)
                        .foregroundColor(.gray)
                    Text("1-10 Objective Scale provides a more intuitive rating (0-10 range).")
                        .font(.caption)
                        .foregroundColor(.gray)
                }
                
                Section(header: Text("Pro")) {
                    Button(action: {
                        // Placeholder: Upgrade to Pro
                        print("Upgrade to Pro tapped")
                    }) {
                        HStack {
                            Image(systemName: "star.fill")
                                .foregroundColor(.yellow)
                            Text("Upgrade to Pro")
                            Spacer()
                            Image(systemName: "chevron.right")
                                .font(.caption)
                                .foregroundColor(.gray)
                        }
                    }
                }
                
                Section(header: Text("Legal")) {
                    Button(action: {
                        // Placeholder: Privacy Policy
                        print("Privacy Policy tapped")
                    }) {
                        HStack {
                            Image(systemName: "lock.shield.fill")
                                .foregroundColor(.blue)
                            Text("Privacy Policy")
                            Spacer()
                            Image(systemName: "chevron.right")
                                .font(.caption)
                                .foregroundColor(.gray)
                        }
                    }
                    
                    Button(action: {
                        // Placeholder: Terms of Use
                        print("Terms of Use tapped")
                    }) {
                        HStack {
                            Image(systemName: "doc.text.fill")
                                .foregroundColor(.blue)
                            Text("Terms of Use")
                            Spacer()
                            Image(systemName: "chevron.right")
                                .font(.caption)
                                .foregroundColor(.gray)
                        }
                    }
                }
            }
        }
    }
}

#Preview {
    ContentView()
}
