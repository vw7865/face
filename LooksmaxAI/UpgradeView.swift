//
//  UpgradeView.swift
//  LooksmaxAI
//
//  Created by Admin on 2026. 01. 14..
//

import SwiftUI
import StoreKit
import UIKit

struct UpgradeView: View {
    @Environment(\.dismiss) var dismiss
    @StateObject private var subscriptionManager = SubscriptionManager.shared
    @State private var selectedPlan: PlanType = .yearly
    @State private var isPurchasing = false
    @State private var purchaseError: String?
    @State private var showError = false
    @State private var showSuccess = false
    @State private var isRestoring = false
    @State private var showRestoreSuccess = false
    @State private var showRestoreError = false
    @State private var restoreError: String?
    
    enum PlanType {
        case weekly
        case yearly
    }
    
    private var weeklyProduct: Product? {
        subscriptionManager.getWeeklyProduct()
    }
    
    private var yearlyProduct: Product? {
        subscriptionManager.getYearlyProduct()
    }
    
    private func formatPrice(_ product: Product?) -> String {
        guard let product = product else {
            return "Loading..."
        }
        return product.displayPrice
    }
    
    private func getPeriod(for product: Product?) -> String {
        guard let product = product else {
            return ""
        }
        if product.id.contains("Weekly") {
            return "per week"
        } else if product.id.contains("Yearly") {
            return "per year"
        }
        return ""
    }
    
    private func calculateSavings() -> String? {
        guard let weekly = weeklyProduct,
              let yearly = yearlyProduct else {
            return nil
        }
        
        let weeklyPrice = Double(truncating: weekly.price as NSDecimalNumber)
        let yearlyPrice = Double(truncating: yearly.price as NSDecimalNumber)
        let weeklyYearlyTotal = weeklyPrice * 52.0 // 52 weeks in a year
        
        if yearlyPrice < weeklyYearlyTotal {
            let savings = weeklyYearlyTotal - yearlyPrice
            let percentage = Int((savings / weeklyYearlyTotal) * 100.0)
            return "Save \(percentage)%"
        }
        
        return nil
    }
    
    var body: some View {
        NavigationStack {
            ZStack {
                Color.black.ignoresSafeArea()
                
                VStack(spacing: 16) {
                    // Header
                    headerSection
                    
                    // Features List
                    featuresSection
                    
                    // Pricing Cards
                    if subscriptionManager.isLoadingProducts {
                        ProgressView()
                            .tint(.white)
                            .padding()
                    } else {
                        pricingSection
                    }
                    
                    // Upgrade Button
                    upgradeButton
                    
                    // Restore Purchases Button
                    restoreButton
                    
                    // Terms
                    termsSection
                    
                    Spacer()
                }
                .padding(.horizontal, 20)
                .padding(.vertical, 20)
                .onAppear {
                    if subscriptionManager.products.isEmpty {
                        subscriptionManager.loadProducts()
                    }
                }
            }
            .navigationTitle("Upgrade to Pro")
            .navigationBarTitleDisplayMode(.inline)
            .navigationBarBackButtonHidden(true)
            .toolbar {
                ToolbarItem(placement: .navigationBarLeading) {
                    Button(action: {
                        dismiss()
                    }) {
                        Image(systemName: "xmark.circle.fill")
                            .font(.system(size: 24))
                            .foregroundColor(.white.opacity(0.8))
                    }
                }
            }
        }
    }
    
    private var headerSection: some View {
        VStack(spacing: 8) {
            Text("Unlock Pro Features")
                .font(.title2)
                .fontWeight(.bold)
                .foregroundColor(.white)
            
            Text("Get access to all looksmaxxing tools")
                .font(.caption)
                .foregroundColor(.gray)
                .multilineTextAlignment(.center)
        }
    }
    
    private var featuresSection: some View {
        VStack(alignment: .leading, spacing: 8) {
            Text("Pro Features")
                .font(.headline)
                .fontWeight(.bold)
                .foregroundColor(.white)
            
            FeatureRow(icon: "infinity", title: "Unlimited Face Ratings", description: "Get unlimited PSL and Objective scale ratings")
            FeatureRow(icon: "sparkles", title: "Unlimited Looksmaxxing Advice", description: "Get personalized looksmaxxing recommendations anytime")
            FeatureRow(icon: "message.fill", title: "Unlimited Rizz Coach", description: "Get unlimited dating conversation advice")
            FeatureRow(icon: "photo.fill", title: "Tindermaxxing", description: "Create stunning dating profile photos with AI")
            FeatureRow(icon: "crown.fill", title: "Sub 5 to Mogger", description: "See what you look like as a mogger")
            FeatureRow(icon: "chart.bar.fill", title: "Full History Access", description: "View unlimited results and track your progress")
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
    }
    
    private var pricingSection: some View {
        VStack(spacing: 12) {
            Text("Choose Your Plan")
                .font(.headline)
                .fontWeight(.bold)
                .foregroundColor(.white)
            
            // Weekly Plan
            PlanCard(
                title: "Weekly",
                price: formatPrice(weeklyProduct),
                period: getPeriod(for: weeklyProduct),
                isSelected: selectedPlan == .weekly,
                savings: nil
            ) {
                selectedPlan = .weekly
            }
            
            // Yearly Plan
            PlanCard(
                title: "Yearly",
                price: formatPrice(yearlyProduct),
                period: getPeriod(for: yearlyProduct),
                isSelected: selectedPlan == .yearly,
                savings: calculateSavings()
            ) {
                selectedPlan = .yearly
            }
        }
    }
    
    private var upgradeButton: some View {
        Button(action: {
            handlePurchase()
        }) {
            HStack {
                if isPurchasing {
                    ProgressView()
                        .tint(.white)
                } else {
                    Image(systemName: "star.fill")
                        .font(.body)
                }
                Text(isPurchasing ? "Processing..." : "Upgrade to Pro")
                    .font(.body)
                    .fontWeight(.semibold)
                    .foregroundColor(.white)
            }
            .frame(maxWidth: .infinity)
            .frame(height: 50)
            .background(
                RoundedRectangle(cornerRadius: 12)
                    .fill(isPurchasing ? Color.gray : Color.red)
            )
        }
        .disabled(isPurchasing || subscriptionManager.isLoadingProducts)
        .alert("Purchase Error", isPresented: $showError) {
            Button("OK", role: .cancel) { }
        } message: {
            Text(purchaseError ?? "An error occurred during purchase. Please try again.")
        }
        .alert("Success!", isPresented: $showSuccess) {
            Button("OK") {
                dismiss()
            }
        } message: {
            Text("Your Pro subscription is now active! Enjoy unlimited access to all features.")
        }
        .alert("Restore Successful", isPresented: $showRestoreSuccess) {
            Button("OK") {
                dismiss()
            }
        } message: {
            Text("Your purchases have been restored successfully.")
        }
        .alert("Restore Failed", isPresented: $showRestoreError) {
            Button("OK", role: .cancel) { }
        } message: {
            Text(restoreError ?? "Unable to restore purchases. Please try again.")
        }
    }
    
    private var restoreButton: some View {
        Button(action: {
            handleRestore()
        }) {
            Text(isRestoring ? "Restoring..." : "Restore Purchases")
                .font(.subheadline)
                .foregroundColor(.gray)
        }
        .disabled(isRestoring)
    }
    
    private func handleRestore() {
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
    
    private func handlePurchase() {
        guard let product = selectedPlan == .weekly ? weeklyProduct : yearlyProduct else {
            purchaseError = "Product not available. Please try again."
            showError = true
            return
        }
        
        isPurchasing = true
        purchaseError = nil
        
        Task {
            do {
                let transaction = try await subscriptionManager.purchase(product)
                if transaction != nil {
                    await MainActor.run {
                        isPurchasing = false
                        showSuccess = true
                    }
                } else {
                    await MainActor.run {
                        isPurchasing = false
                        // User cancelled or pending - don't show error
                    }
                }
            } catch let error as StoreError {
                await MainActor.run {
                    isPurchasing = false
                    purchaseError = "Purchase verification failed. Please try again."
                    showError = true
                }
            } catch {
                await MainActor.run {
                    isPurchasing = false
                    // Provide user-friendly error messages
                    let errorMessage: String
                    let nsError = error as NSError
                    if nsError.domain == "SKErrorDomain" {
                        switch nsError.code {
                        case 0: // SKErrorUnknown
                            errorMessage = "An unknown error occurred. Please try again."
                        case 1: // SKErrorClientInvalid
                            errorMessage = "Invalid client. Please try again."
                        case 2: // SKErrorPaymentCancelled
                            return // Don't show error for user cancellation
                        case 3: // SKErrorPaymentInvalid
                            errorMessage = "Invalid payment. Please check your payment method."
                        case 4: // SKErrorPaymentNotAllowed
                            errorMessage = "Payment not allowed. Please check your device settings."
                        case 5: // SKErrorStoreProductNotAvailable
                            errorMessage = "Product not available. Please try again later."
                        default:
                            errorMessage = "Purchase failed. Please try again."
                        }
                    } else {
                        errorMessage = error.localizedDescription.isEmpty ? "Purchase failed. Please try again." : error.localizedDescription
                    }
                    purchaseError = errorMessage
                    showError = true
                }
            }
        }
    }
    
    private var termsSection: some View {
        VStack(spacing: 6) {
            Text("Payment will be charged to your Apple ID account. Subscription automatically renews unless cancelled.")
                .font(.system(size: 10))
                .foregroundColor(.gray)
                .multilineTextAlignment(.center)
                .lineLimit(3)
            
            HStack(spacing: 12) {
                Link("Terms", destination: URL(string: "https://www.faceratingapp.com/terms")!)
                    .font(.system(size: 10))
                    .foregroundColor(.red)
                
                Text("•")
                    .font(.system(size: 10))
                    .foregroundColor(.gray)
                
                Link("Privacy", destination: URL(string: "https://www.faceratingapp.com/privacy")!)
                    .font(.system(size: 10))
                    .foregroundColor(.red)
            }
        }
    }
}

struct FeatureRow: View {
    let icon: String
    let title: String
    let description: String
    
    var body: some View {
        HStack(alignment: .top, spacing: 8) {
            Image(systemName: icon)
                .font(.system(size: 14))
                .foregroundColor(.red)
                .frame(width: 20)
            
            VStack(alignment: .leading, spacing: 2) {
                Text(title)
                    .font(.system(size: 13))
                    .fontWeight(.semibold)
                    .foregroundColor(.white)
                
                Text(description)
                    .font(.system(size: 11))
                    .foregroundColor(.gray)
                    .fixedSize(horizontal: false, vertical: true)
                    .lineLimit(nil)
            }
            
            Spacer()
        }
    }
}

struct PlanCard: View {
    let title: String
    let price: String
    let period: String
    let isSelected: Bool
    let savings: String?
    let action: () -> Void
    
    var body: some View {
        Button(action: action) {
            HStack {
                VStack(alignment: .leading, spacing: 4) {
                    HStack(spacing: 6) {
                        Text(title)
                            .font(.system(size: 16))
                            .fontWeight(.bold)
                            .foregroundColor(.white)
                        
                        if let savings = savings {
                            Text(savings)
                                .font(.system(size: 9))
                                .fontWeight(.semibold)
                                .foregroundColor(.white)
                                .padding(.horizontal, 6)
                                .padding(.vertical, 2)
                                .background(
                                    Capsule()
                                        .fill(Color.green)
                                )
                        }
                    }
                    
                    HStack(alignment: .firstTextBaseline, spacing: 2) {
                        Text(price)
                            .font(.system(size: 20))
                            .fontWeight(.bold)
                            .foregroundColor(.white)
                        
                        Text(period)
                            .font(.system(size: 11))
                            .foregroundColor(.gray)
                    }
                }
                
                Spacer()
                
                // Selection indicator
                ZStack {
                    Circle()
                        .fill(isSelected ? Color.red : Color.clear)
                        .frame(width: 20, height: 20)
                        .overlay(
                            Circle()
                                .stroke(isSelected ? Color.red : Color.gray, lineWidth: 2)
                        )
                    
                    if isSelected {
                        Image(systemName: "checkmark")
                            .font(.system(size: 10, weight: .bold))
                            .foregroundColor(.white)
                    }
                }
            }
            .padding(12)
            .background(
                RoundedRectangle(cornerRadius: 12)
                    .fill(isSelected ? Color.red.opacity(0.2) : Color.white.opacity(0.1))
                    .overlay(
                        RoundedRectangle(cornerRadius: 12)
                            .stroke(isSelected ? Color.red : Color.white.opacity(0.2), lineWidth: isSelected ? 2 : 1)
                    )
            )
        }
        .buttonStyle(PlainButtonStyle())
    }
}

// Extension to find UITabBarController and UINavigationController
extension UIViewController {
    func findTabBarController() -> UITabBarController? {
        if let tabBarController = self as? UITabBarController {
            return tabBarController
        }
        if let navController = self as? UINavigationController {
            return navController.viewControllers.first?.findTabBarController()
        }
        if let presented = self.presentedViewController {
            return presented.findTabBarController()
        }
        return self.parent?.findTabBarController() ?? self.presentingViewController?.findTabBarController()
    }
    
    func findNavigationController() -> UINavigationController? {
        if let navController = self as? UINavigationController {
            return navController
        }
        if let navController = self.navigationController {
            return navController
        }
        return self.parent?.findNavigationController()
    }
}


#Preview {
    UpgradeView()
}

