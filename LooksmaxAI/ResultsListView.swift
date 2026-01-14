//
//  ResultsListView.swift
//  LooksmaxAI
//
//  Created by Admin on 2026. 01. 11..
//

import SwiftUI

struct ResultsListView: View {
    @State private var selectedResult: FaceAnalysisResults? = nil
    @State private var showResultsView = false
    @State private var refreshTrigger = UUID()
    @AppStorage("lastAnalysisResults") private var lastAnalysisResultsData: Data?
    
    private var history: [ResultHistoryItem] {
        ResultHistoryManager.shared.loadHistory()
    }
    
    private var latestResult: FaceAnalysisResults? {
        history.first?.result
    }
    
    var body: some View {
        NavigationStack {
            ZStack {
                Color.black.ignoresSafeArea()
                
                if history.isEmpty {
                    // Empty state
                    VStack(spacing: 20) {
                        Image(systemName: "chart.bar.fill")
                            .imageScale(.large)
                            .foregroundStyle(.tint)
                            .font(.system(size: 60))
                        Text("No Results Yet")
                            .font(.title)
                            .foregroundColor(.white)
                        Text("Complete a face rating to see your results here")
                            .font(.subheadline)
                            .foregroundColor(.gray)
                            .multilineTextAlignment(.center)
                            .padding(.horizontal)
                    }
                    .padding()
                } else {
                    // Results list
                    ScrollView {
                        LazyVStack(spacing: 16) {
                            ForEach(history, id: \.id) { item in
                                ResultRowView(item: item) {
                                    selectedResult = item.result
                                    showResultsView = true
                                }
                            }
                        }
                        .padding()
                    }
                }
            }
            .navigationTitle("Face Ratings")
            .navigationBarTitleDisplayMode(.large)
            .sheet(isPresented: $showResultsView) {
                if let result = selectedResult {
                    ResultsView(results: result, onClose: {
                        print("📋 ResultsView onClose called from sheet")
                        showResultsView = false
                        refreshTrigger = UUID() // Trigger refresh when closing
                    })
                }
            }
            .onAppear {
                print("📋 ResultsListView onAppear - History count: \(history.count)")
                // Force refresh when view appears
                refreshTrigger = UUID()
            }
            .onChange(of: lastAnalysisResultsData) { _ in
                // Refresh when new results are saved
                print("📋 New results data detected, refreshing list")
                refreshTrigger = UUID()
            }
        }
    }
}

struct ResultRowView: View {
    let item: ResultHistoryItem
    let onTap: () -> Void
    @StateObject private var scaleManager = ScaleManager.shared
    @State private var thumbnailImage: UIImage? = nil
    
    var body: some View {
        Button(action: onTap) {
            HStack(spacing: 16) {
                // Thumbnail Image
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
                .frame(width: 60, height: 60)
                .clipShape(RoundedRectangle(cornerRadius: 8))
                .overlay(
                    RoundedRectangle(cornerRadius: 8)
                        .stroke(Color.white.opacity(0.2), lineWidth: 1)
                )
                .background(
                    RoundedRectangle(cornerRadius: 8)
                        .fill(Color.white.opacity(0.05))
                )
                
                // PSL Score Badge
                VStack(spacing: 4) {
                    let psl = item.result.overall.psl ?? 50.0
                    let display = scaleManager.getFullDisplay(psl)
                    
                    Text(display.score)
                        .font(.system(size: 24, weight: .bold))
                        .foregroundColor(.white)
                    
                    Text(display.label)
                        .font(.system(size: 10, weight: .medium))
                        .foregroundColor(.cyan.opacity(0.8))
                        .lineLimit(1)
                }
                .frame(width: 80)
                
                // Date and time
                VStack(alignment: .leading, spacing: 4) {
                    Text(item.timestamp, style: .date)
                        .font(.headline)
                        .foregroundColor(.white)
                    
                    Text(item.timestamp, style: .time)
                        .font(.subheadline)
                        .foregroundColor(.gray)
                }
                
                Spacer()
                
                // Chevron
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
        .buttonStyle(PlainButtonStyle())
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

