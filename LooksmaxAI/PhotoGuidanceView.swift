//
//  PhotoGuidanceView.swift
//  LooksmaxAI
//
//  Created by Admin on 2026. 01. 07..
//

import SwiftUI

struct PhotoGuidanceView: View {
    let isSideProfile: Bool
    let gender: String
    @Environment(\.dismiss) var dismiss
    
    var body: some View {
        NavigationView {
            ScrollView {
                VStack(alignment: .leading, spacing: 25) {
                    // Header
                    VStack(spacing: 10) {
                        Image(systemName: isSideProfile ? "person.crop.circle.badge.checkmark" : "camera.fill")
                            .font(.system(size: 50))
                            .foregroundColor(.blue)
                        
                        Text(isSideProfile ? "Side Profile Photo Tips" : "Front Photo Tips")
                            .font(.title)
                            .fontWeight(.bold)
                    }
                    .frame(maxWidth: .infinity)
                    .padding(.top)
                    
                    // Lighting Section
                    GuidanceSection(
                        icon: "lightbulb.fill",
                        title: "Lighting",
                        tips: [
                            "Use natural, even lighting (near a window works great)",
                            "Avoid harsh shadows on your face",
                            "Don't use flash directly - it creates harsh shadows",
                            "Face the light source, not away from it",
                            "Avoid backlighting (light behind you)"
                        ]
                    )
                    
                    // Positioning Section
                    GuidanceSection(
                        icon: "viewfinder",
                        title: isSideProfile ? "Side Profile Position" : "Front Position",
                        tips: isSideProfile ? [
                            "Turn your head 90 degrees to the side",
                            "Keep your shoulders straight",
                            "Look straight ahead (not at camera)",
                            gender == "Female" ? "Hair can cover your ear - that's okay!" : "Make sure your ear is visible",
                            "Keep your chin level, not tilted up or down"
                        ] : [
                            "Face the camera directly (like a passport photo)",
                            "Keep your head straight, not tilted",
                            "Look directly at the camera lens",
                            "Keep your shoulders square to the camera",
                            "Center your face in the frame"
                        ]
                    )
                    
                    // Distance & Framing
                    GuidanceSection(
                        icon: "crop",
                        title: "Distance & Framing",
                        tips: [
                            "Stand 2-3 feet from the camera",
                            "Your face should fill about 30-40% of the frame",
                            "Include your head and shoulders",
                            "Make sure your entire face is visible",
                            "Avoid cropping off parts of your face"
                        ]
                    )
                    
                    // Background
                    GuidanceSection(
                        icon: "photo.fill",
                        title: "Background",
                        tips: [
                            "Use a plain, uncluttered background",
                            "Avoid busy patterns or distracting elements",
                            "A solid color wall works best",
                            "Make sure nothing is behind your head"
                        ]
                    )
                    
                    // Expression
                    GuidanceSection(
                        icon: "face.smiling",
                        title: "Expression",
                        tips: [
                            "Use a neutral, relaxed expression",
                            "Keep your mouth closed",
                            "Don't smile or frown",
                            "Keep your eyes open and looking forward",
                            "Relax your facial muscles"
                        ]
                    )
                    
                    // Quality Tips
                    GuidanceSection(
                        icon: "checkmark.circle.fill",
                        title: "Quality Checklist",
                        tips: [
                            "Use the highest quality camera available",
                            "Make sure the photo is in focus",
                            "Avoid blurry or pixelated images",
                            "Remove glasses if possible (or use non-reflective lenses)",
                            "Tie back long hair if it covers your face",
                            "Remove hats or head coverings"
                        ]
                    )
                    
                    // Why it matters
                    VStack(alignment: .leading, spacing: 10) {
                        HStack {
                            Image(systemName: "info.circle.fill")
                                .foregroundColor(.blue)
                            Text("Why This Matters")
                                .font(.headline)
                        }
                        
                        Text("Good lighting and proper positioning help our AI accurately analyze your facial features. Poor photos can result in inaccurate scores or failed analysis.")
                            .font(.subheadline)
                            .foregroundColor(.gray)
                    }
                    .padding()
                    .background(Color.blue.opacity(0.1))
                    .cornerRadius(10)
                    
                    Spacer(minLength: 20)
                }
                .padding()
            }
            .navigationTitle("Photo Tips")
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

struct GuidanceSection: View {
    let icon: String
    let title: String
    let tips: [String]
    
    var body: some View {
        VStack(alignment: .leading, spacing: 15) {
            HStack {
                Image(systemName: icon)
                    .font(.title3)
                    .foregroundColor(.blue)
                Text(title)
                    .font(.headline)
            }
            
            VStack(alignment: .leading, spacing: 10) {
                ForEach(tips, id: \.self) { tip in
                    HStack(alignment: .top, spacing: 10) {
                        Image(systemName: "checkmark.circle.fill")
                            .font(.caption)
                            .foregroundColor(.green)
                            .padding(.top, 2)
                        Text(tip)
                            .font(.subheadline)
                            .fixedSize(horizontal: false, vertical: true)
                    }
                }
            }
        }
        .padding()
        .background(Color.gray.opacity(0.1))
        .cornerRadius(12)
    }
}

#Preview {
    PhotoGuidanceView(isSideProfile: false, gender: "Male")
}






