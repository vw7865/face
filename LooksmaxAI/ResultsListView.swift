//
//  ResultsListView.swift
//  LooksmaxAI
//
//  Created by Admin on 2026. 01. 11..
//

import SwiftUI

// Wrapper to make FaceAnalysisResults identifiable for sheet presentation
struct SelectedAnalysis: Identifiable {
    let id = UUID()
    let result: FaceAnalysisResults
    let thumbnailPath: String?
    let gender: String?
    let thumbnailImage: UIImage? // Pass image directly for instant display
}

struct ResultsListView: View {
    @State private var selectedAnalysis: SelectedAnalysis? = nil
    @State private var refreshTrigger = UUID()
    @State private var isEditMode = false
    @State private var showDeleteAllAlert = false
    @State private var history: [ResultHistoryItem] = []
    @StateObject private var usageTracker = UsageTracker.shared
    @StateObject private var subscriptionManager = SubscriptionManager.shared
    @State private var isShowingUpgrade = false
    @AppStorage("lastAnalysisResults") private var lastAnalysisResultsData: Data?
    
    private var displayedHistory: [ResultHistoryItem] {
        let allHistory = history
        if subscriptionManager.isPro {
            return allHistory
        } else {
            // Free users: limit to last 3 results
            return Array(allHistory.prefix(usageTracker.getMaxHistoryResults()))
        }
    }
    
    private var latestResult: FaceAnalysisResults? {
        history.first?.result
    }
    
    var body: some View {
        NavigationStack {
            ZStack {
                Color.black.ignoresSafeArea()
                
                if history.isEmpty {
                    // Enhanced empty state
                    VStack(spacing: 24) {
                        Image(systemName: "face.smiling")
                            .imageScale(.large)
                            .foregroundStyle(
                                LinearGradient(
                                    colors: [Color.cyan, Color.purple],
                                    startPoint: .topLeading,
                                    endPoint: .bottomTrailing
                                )
                            )
                            .font(.system(size: 80))
                        
                        Text("🧐 No Ratings Yet")
                            .font(.system(size: 28, weight: .bold, design: .rounded))
                            .foregroundColor(.white)
                        
                        Text("Start by uploading a photo to get your first score!")
                            .font(.system(size: 16, weight: .medium))
                            .foregroundColor(.gray)
                            .multilineTextAlignment(.center)
                            .padding(.horizontal, 40)
                        
                        Button(action: {
                            // Post notification to switch to Rate tab
                            NotificationCenter.default.post(name: NSNotification.Name("SwitchToRateTab"), object: nil)
                        }) {
                            HStack(spacing: 8) {
                                Image(systemName: "camera.fill")
                                Text("Get My First Rating")
                                    .font(.system(size: 16, weight: .semibold))
                            }
                            .foregroundColor(.white)
                            .padding(.horizontal, 24)
                            .padding(.vertical, 12)
                            .background(
                                LinearGradient(
                                    colors: [Color.blue, Color.cyan],
                                    startPoint: .leading,
                                    endPoint: .trailing
                                )
                            )
                            .cornerRadius(12)
                            .shadow(color: Color.blue.opacity(0.4), radius: 8, x: 0, y: 4)
                        }
                        .padding(.top, 8)
                    }
                    .padding()
                } else {
                    // Results list
                    VStack(spacing: 0) {
                        if isEditMode {
                            // Clear All button when in edit mode
                            Button(action: {
                                showDeleteAllAlert = true
                            }) {
                                HStack {
                                    Image(systemName: "trash.fill")
                                    Text("Delete All Results")
                                }
                                .font(.headline)
                                .foregroundColor(.red)
                                .frame(maxWidth: .infinity)
                                .padding()
                                .background(
                                    RoundedRectangle(cornerRadius: 12)
                                        .fill(Color.red.opacity(0.2))
                                )
                            }
                            .padding()
                        }
                        
                        ScrollView {
                            LazyVStack(spacing: 16) {
                                // Show upgrade prompt if free user has more than 3 results
                                if !subscriptionManager.isPro && history.count > usageTracker.getMaxHistoryResults() {
                                    VStack(spacing: 12) {
                                        Text("Viewing last \(usageTracker.getMaxHistoryResults()) results")
                                            .font(.headline)
                                            .foregroundColor(.white)
                                        Text("Upgrade to Pro to view all \(history.count) results")
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
                                
                                ForEach(displayedHistory, id: \.id) { item in
                                    ResultRowView(
                                        item: item,
                                        isEditMode: isEditMode,
                                        onTap: {
                                            if !isEditMode {
                                                // Load thumbnail image immediately for instant display
                                                var thumbnailImage: UIImage? = nil
                                                if let path = item.thumbnailPath {
                                                    thumbnailImage = ResultHistoryManager.shared.loadThumbnail(for: path)
                                                }
                                                
                                                // Set result with gender, thumbnail path, and image included
                                                selectedAnalysis = SelectedAnalysis(
                                                    result: item.result,
                                                    thumbnailPath: item.thumbnailPath,
                                                    gender: item.gender,
                                                    thumbnailImage: thumbnailImage
                                                )
                                            }
                                        },
                                        onDelete: {
                                            withAnimation(.spring(response: 0.3)) {
                                                deleteResult(item.id)
                                            }
                                        }
                                    )
                                    .opacity(isEditMode ? 0.95 : 1.0)
                                    .scaleEffect(isEditMode ? 0.98 : 1.0)
                                }
                            }
                            .padding()
                        }
                    }
                }
            }
            .navigationTitle("Face Ratings")
            .navigationBarTitleDisplayMode(.large)
            .toolbar {
                ToolbarItem(placement: .navigationBarTrailing) {
                    if !history.isEmpty {
                        HStack {
                            if isEditMode {
                                Button("Done") {
                                    withAnimation(.spring(response: 0.3)) {
                                        isEditMode = false
                                    }
                                }
                                .foregroundColor(.white)
                            } else {
                                Button(action: {
                                    withAnimation(.spring(response: 0.3)) {
                                        isEditMode = true
                                    }
                                }) {
                                    Image(systemName: "trash")
                                        .foregroundColor(.white)
                                }
                            }
                        }
                    }
                }
            }
            .fullScreenCover(item: $selectedAnalysis) { selected in
                ResultsView(
                    results: selected.result,
                    onClose: {
                        print("📋 ResultsView onClose called from fullScreenCover")
                        selectedAnalysis = nil
                        refreshTrigger = UUID() // Trigger refresh when closing
                    },
                    gender: selected.gender,
                    thumbnailPath: selected.thumbnailPath,
                    thumbnailImage: selected.thumbnailImage
                )
            }
            .onAppear {
                loadHistory()
            }
            .onChange(of: lastAnalysisResultsData) { _ in
                // Refresh when new results are saved
                print("📋 New results data detected, refreshing list")
                loadHistory()
            }
            .onChange(of: refreshTrigger) { _ in
                // Refresh when trigger changes
                loadHistory()
            }
            .alert("Delete All Results", isPresented: $showDeleteAllAlert) {
                Button("Cancel", role: .cancel) {}
                Button("Delete All", role: .destructive) {
                    deleteAllResults()
                }
            } message: {
                Text("Are you sure you want to delete all \(history.count) results? This action cannot be undone.")
            }
            .fullScreenCover(isPresented: $isShowingUpgrade) {
                UpgradeView()
            }
        }
    }
    
    private func loadHistory() {
        history = ResultHistoryManager.shared.loadHistory()
        print("📋 Loaded history - Count: \(history.count)")
    }
    
    private func deleteResult(_ id: String) {
        ResultHistoryManager.shared.deleteResult(id: id)
        // Immediately update the local state
        loadHistory()
    }
    
    private func deleteAllResults() {
        ResultHistoryManager.shared.clearHistory()
        isEditMode = false
        // Immediately update the local state
        loadHistory()
    }
}

struct ResultRowView: View {
    let item: ResultHistoryItem
    let isEditMode: Bool
    let onTap: () -> Void
    let onDelete: () -> Void
    @StateObject private var scaleManager = ScaleManager.shared
    @State private var thumbnailImage: UIImage? = nil
    
    private var psl: Double {
        item.result.overall.psl ?? 50.0
    }
    
    private var display: (score: String, label: String) {
        scaleManager.getFullDisplay(psl, gender: item.gender)
    }
    
    private var scoreColor: Color {
        getScoreColor(psl: psl)
    }
    
    private func getScoreColor(psl: Double) -> Color {
        let pslScale = scaleManager.convertToPSLScale(psl)
        switch pslScale {
        case ..<1.4:
            return .red // Subhuman
        case 1.4..<3.3:
            return .orange // LTN
        case 3.3..<5.7:
            return .yellow // MTN
        case 5.7..<6.9:
            return .green // HTN/Chadlite
        case 6.9..<7.8:
            return .cyan // Chad/Stacy
        default:
            return .purple // Adam/Eve
        }
    }
    
    private func formatDate(_ date: Date) -> String {
        let calendar = Calendar.current
        let now = Date()
        
        if calendar.isDateInToday(date) {
            return "Today"
        } else if calendar.isDateInYesterday(date) {
            return "Yesterday"
        } else if let days = calendar.dateComponents([.day], from: date, to: now).day, days < 7 {
            return "\(days) days ago"
        } else {
            let formatter = DateFormatter()
            formatter.dateFormat = "MMM d, yyyy"
            return formatter.string(from: date)
        }
    }
    
    var body: some View {
        HStack(spacing: 16) {
            // Thumbnail Image - circular/squircle with tighter crop
            Group {
                if let thumbnail = thumbnailImage {
                    Image(uiImage: thumbnail)
                        .resizable()
                        .aspectRatio(contentMode: .fill)
                } else {
                    // Placeholder while loading or if no thumbnail
                    Image(systemName: "person.fill")
                        .foregroundColor(.gray.opacity(0.5))
                        .font(.system(size: 24))
                }
            }
            .frame(width: 70, height: 70)
            .clipShape(RoundedRectangle(cornerRadius: 16)) // Squircle shape
            .overlay(
                RoundedRectangle(cornerRadius: 16)
                    .stroke(
                        LinearGradient(
                            colors: [Color.white.opacity(0.3), Color.white.opacity(0.1)],
                            startPoint: .topLeading,
                            endPoint: .bottomTrailing
                        ),
                        lineWidth: 2
                    )
            )
            .shadow(color: .black.opacity(0.3), radius: 4, x: 0, y: 2)
            
            // Score and Label Section
            VStack(alignment: .leading, spacing: 6) {
                // Score with color coding
                Text(display.score)
                    .font(.system(size: 22, weight: .bold, design: .rounded))
                    .foregroundColor(scoreColor)
                
                // Label as colored badge - allow wrapping
                Text(display.label)
                    .font(.system(size: 12, weight: .semibold))
                    .foregroundColor(.white)
                    .padding(.horizontal, 10)
                    .padding(.vertical, 4)
                    .background(
                        Capsule()
                            .fill(scoreColor.opacity(0.3))
                            .overlay(
                                Capsule()
                                    .stroke(scoreColor.opacity(0.6), lineWidth: 1)
                            )
                    )
                    .fixedSize(horizontal: true, vertical: false)
            }
            .frame(minWidth: 90, maxWidth: 120, alignment: .leading)
            
            // Date
            VStack(alignment: .leading, spacing: 4) {
                Text(formatDate(item.timestamp))
                    .font(.system(size: 15, weight: .semibold))
                    .foregroundColor(.white)
                
                Text(item.timestamp, style: .time)
                    .font(.system(size: 12, weight: .regular))
                    .foregroundColor(.gray.opacity(0.8))
            }
            
            Spacer()
            
            // Delete button or Chevron
            if isEditMode {
                Button(action: onDelete) {
                    Image(systemName: "trash.fill")
                        .foregroundColor(.red)
                        .font(.system(size: 20))
                        .padding(8)
                        .background(
                            Circle()
                                .fill(Color.red.opacity(0.2))
                        )
                }
                .buttonStyle(PlainButtonStyle())
            } else {
                Image(systemName: "chevron.right")
                    .foregroundColor(.gray.opacity(0.6))
                    .font(.system(size: 14, weight: .semibold))
            }
        }
        .padding(16)
        .background(
            RoundedRectangle(cornerRadius: 18)
                .fill(
                    LinearGradient(
                        colors: [Color.white.opacity(0.12), Color.white.opacity(0.08)],
                        startPoint: .topLeading,
                        endPoint: .bottomTrailing
                    )
                )
                .overlay(
                    RoundedRectangle(cornerRadius: 18)
                        .stroke(
                            LinearGradient(
                                colors: [Color.white.opacity(0.25), Color.white.opacity(0.1)],
                                startPoint: .topLeading,
                                endPoint: .bottomTrailing
                            ),
                            lineWidth: 1
                        )
                )
        )
        .shadow(color: Color.white.opacity(0.08), radius: 6, x: 0, y: 2)
        .contentShape(Rectangle())
        .onTapGesture {
            if !isEditMode {
                onTap()
            }
        }
        .onAppear {
            loadThumbnail()
        }
    }
    
    private func loadThumbnail() {
        guard let path = item.thumbnailPath else {
            return
        }
        
        // Load thumbnail on background thread
        DispatchQueue.global(qos: .userInitiated).async {
            if let image = ResultHistoryManager.shared.loadThumbnail(for: path) {
                DispatchQueue.main.async {
                    thumbnailImage = image
                }
            }
        }
    }
}

