//
//  CreditPurchaseView.swift
//  LooksmaxAI
//
//  Created by Admin on 2026. 01. 24..
//

import SwiftUI
import StoreKit

struct CreditPurchaseView: View {
    @Environment(\.dismiss) var dismiss
    @StateObject private var usageTracker = UsageTracker.shared
    @StateObject private var subscriptionManager = SubscriptionManager.shared
    @StateObject private var creditPurchaseManager = CreditPurchaseManager.shared
    @State private var isPurchasing = false
    @State private var purchaseError: String? = nil
    @State private var showPurchaseError = false
    @State private var selectedPackage: CreditPackage? = nil
    
    enum CreditPackage: Int, CaseIterable {
        case five = 5
        case ten = 10
        case twentyFive = 25
        case fifty = 50
        
        var productID: String {
            return "com.facemaxxing.LooksmaxAI.Credits.\(rawValue)"
        }
        
        var pricePerCredit: String {
            switch self {
            case .five: return "$0.60"
            case .ten: return "$0.50"
            case .twentyFive: return "$0.40"
            case .fifty: return "$0.34"
            }
        }
        
        var isPopular: Bool {
            return self == .fifty
        }
    }
    
    var body: some View {
        NavigationStack {
            ZStack {
                Color.black.ignoresSafeArea()
                
                VStack(spacing: 0) {
                    // Header section
                    headerSection
                        .padding(.top, 20)
                    
                    // Current credits display
                    currentCreditsSection
                        .padding(.top, 16)
                    
                    // Section subtitle
                    Text("Available Plans")
                        .font(.subheadline)
                        .foregroundColor(.gray)
                        .padding(.top, 8)
                    
                    // Credit info
                    Text("1 credit = 1 photo generation")
                        .font(.caption)
                        .foregroundColor(.gray.opacity(0.8))
                        .padding(.top, 4)
                    
                    // Credit packages - vertical list
                    creditPackagesSection
                        .padding(.top, 24)
                    
                    Spacer()
                    
                    // Continue button
                    continueButton
                        .padding(.horizontal)
                        .padding(.bottom, 30)
                }
            }
            .navigationTitle("")
            .navigationBarTitleDisplayMode(.inline)
            .toolbar {
                ToolbarItem(placement: .navigationBarTrailing) {
                    Button(action: {
                        dismiss()
                    }) {
                        Image(systemName: "xmark.circle.fill")
                            .font(.system(size: 24))
                            .foregroundColor(.white.opacity(0.8))
                    }
                }
            }
            .alert("Purchase Error", isPresented: $showPurchaseError) {
                Button("OK", role: .cancel) { }
            } message: {
                Text(purchaseError ?? "An error occurred during purchase")
            }
        }
        .onAppear {
            // Load credit products when view appears
            creditPurchaseManager.loadCreditProducts()
        }
    }
    
    private var headerSection: some View {
        VStack(spacing: 12) {
            Text("Purchase Credits")
                .font(.title2)
                .fontWeight(.bold)
                .foregroundColor(.white)
        }
    }
    
    private var currentCreditsSection: some View {
        VStack(spacing: 6) {
            Text("Current Credits")
                .font(.subheadline)
                .foregroundColor(.gray)
            
            HStack(spacing: 6) {
                Image(systemName: "sparkles")
                    .font(.system(size: 20))
                    .foregroundColor(.cyan)
                Text("\(usageTracker.getImageGenerationCreditsRemaining())")
                    .font(.system(size: 28, weight: .bold))
                    .foregroundColor(.white)
            }
            .padding(.vertical, 12)
            .padding(.horizontal, 20)
            .background(
                RoundedRectangle(cornerRadius: 12)
                    .fill(Color.cyan.opacity(0.1))
                    .overlay(
                        RoundedRectangle(cornerRadius: 12)
                            .stroke(Color.cyan.opacity(0.3), lineWidth: 1.5)
                    )
            )
        }
    }
    
    private var creditPackagesSection: some View {
        VStack(spacing: 12) {
            if creditPurchaseManager.isLoadingProducts {
                ProgressView()
                    .padding()
            } else {
                ForEach(CreditPackage.allCases, id: \.self) { package in
                    if let product = creditPurchaseManager.getProduct(for: package.rawValue) {
                        CreditPackageRow(
                            package: package,
                            product: product,
                            isSelected: selectedPackage == package,
                            onSelect: {
                                selectedPackage = package
                            }
                        )
                    } else {
                        // Product not loaded - show disabled state
                        CreditPackageRow(
                            package: package,
                            product: nil,
                            isSelected: false,
                            isDisabled: true,
                            onSelect: {
                                // Disabled - cannot select
                            }
                        )
                    }
                }
            }
        }
        .padding(.horizontal)
    }
    
    private var continueButton: some View {
        Button(action: {
            guard let package = selectedPackage else { return }
            purchaseSelectedCredits(package)
        }) {
            HStack {
                if isPurchasing {
                    ProgressView()
                        .progressViewStyle(CircularProgressViewStyle(tint: .white))
                }
                Text(isPurchasing ? "Processing..." : "Continue")
                    .font(.headline)
                    .foregroundColor(.white)
            }
            .frame(maxWidth: .infinity)
            .padding()
            .background(
                RoundedRectangle(cornerRadius: 12)
                    .fill(selectedPackage != nil && !isPurchasing ? Color.cyan : Color.gray.opacity(0.3))
            )
        }
        .disabled(selectedPackage == nil || isPurchasing)
    }
    
    private func purchaseSelectedCredits(_ package: CreditPackage) {
        // Ensure products are loaded before allowing purchase
        guard !creditPurchaseManager.isLoadingProducts else {
            purchaseError = "Products are still loading. Please wait."
            showPurchaseError = true
            return
        }
        
        guard let product = creditPurchaseManager.getProduct(for: package.rawValue) else {
            purchaseError = "Product not available. Please try again."
            showPurchaseError = true
            return
        }
        
        isPurchasing = true
        purchaseError = nil
        
        Task {
            do {
                let success = try await creditPurchaseManager.purchaseCredits(product)
                
                await MainActor.run {
                    isPurchasing = false
                    
                    if success {
                        // Purchase successful - credits already added by CreditPurchaseManager
                        // Dismiss after a brief delay to show success
                        DispatchQueue.main.asyncAfter(deadline: .now() + 0.5) {
                            dismiss()
                        }
                    } else {
                        // User cancelled - no error needed
                    }
                }
            } catch {
                await MainActor.run {
                    isPurchasing = false
                    purchaseError = error.localizedDescription
                    showPurchaseError = true
                }
            }
        }
    }
}

// MARK: - Credit Package Row
struct CreditPackageRow: View {
    let package: CreditPurchaseView.CreditPackage
    let product: Product?
    let isSelected: Bool
    let isDisabled: Bool
    let onSelect: () -> Void
    
    init(package: CreditPurchaseView.CreditPackage, product: Product?, isSelected: Bool, isDisabled: Bool = false, onSelect: @escaping () -> Void) {
        self.package = package
        self.product = product
        self.isSelected = isSelected
        self.isDisabled = isDisabled
        self.onSelect = onSelect
    }
    
    private var displayPrice: String {
        if let product = product {
            return product.displayPrice
        }
        return "Loading..."
    }
    
    var body: some View {
        Button(action: {
            if !isDisabled {
                onSelect()
            }
        }) {
            HStack {
                VStack(alignment: .leading, spacing: 4) {
                    HStack(spacing: 4) {
                        Text("\(package.rawValue) Credits")
                            .font(.system(size: 17, weight: .semibold))
                            .foregroundColor(isDisabled ? .gray : .white)
                        
                        if package.isPopular && !isDisabled {
                            Text("BEST VALUE")
                                .font(.system(size: 9, weight: .bold))
                                .foregroundColor(.white)
                                .padding(.horizontal, 6)
                                .padding(.vertical, 2)
                                .background(
                                    Capsule()
                                        .fill(Color.cyan)
                                )
                        }
                    }
                    
                    if product != nil {
                        Text("\(displayPrice) • \(package.pricePerCredit) per credit")
                            .font(.system(size: 14))
                            .foregroundColor(.gray)
                    } else {
                        Text(displayPrice)
                            .font(.system(size: 14))
                            .foregroundColor(.gray)
                    }
                }
                
                Spacer()
                
                // Selection indicator
                ZStack {
                    Circle()
                        .fill(isSelected ? Color.cyan : Color.clear)
                        .frame(width: 24, height: 24)
                        .overlay(
                            Circle()
                                .stroke(isSelected ? Color.cyan : Color.gray.opacity(0.5), lineWidth: 2)
                        )
                    
                    if isSelected {
                        Image(systemName: "checkmark")
                            .font(.system(size: 12, weight: .bold))
                            .foregroundColor(.white)
                    }
                }
            }
            .padding()
            .background(
                RoundedRectangle(cornerRadius: 12)
                    .fill(isDisabled ? Color.white.opacity(0.02) : Color.white.opacity(0.05))
                    .overlay(
                        RoundedRectangle(cornerRadius: 12)
                            .stroke(isSelected ? Color.cyan.opacity(0.5) : Color.white.opacity(0.1), lineWidth: isSelected ? 2 : 1)
                    )
            )
            .opacity(isDisabled ? 0.5 : 1.0)
        }
        .buttonStyle(PlainButtonStyle())
        .disabled(isDisabled)
    }
}

// MARK: - Compact Credit Card (removed - no longer needed)
