//
//  SubscriptionManager.swift
//  LooksmaxAI
//
//  Created by Admin on 2026. 01. 07..
//

import Foundation
import StoreKit

class SubscriptionManager: ObservableObject {
    static let shared = SubscriptionManager()
    
    @Published var isPro: Bool = false
    @Published var isLoading: Bool = false
    @Published var products: [Product] = []
    @Published var isLoadingProducts: Bool = false
    @Published var currentSubscription: Product.SubscriptionInfo.Status?
    @Published var subscriptionExpirationDate: Date?
    @Published var activeProductID: String?
    
    // Product IDs from App Store Connect
    private let weeklyProductID = "com.facemaxxing.LooksmaxAI.Weekly"
    private let yearlyProductID = "com.facemaxxing.LooksmaxAI.Yearly"
    
    private var updateListenerTask: Task<Void, Error>?
    
    private init() {
        // Start listening for subscription updates
        updateListenerTask = listenForTransactions()
        loadProducts()
        checkSubscriptionStatus()
    }
    
    deinit {
        updateListenerTask?.cancel()
    }
    
    // MARK: - StoreKit Transaction Listener
    private func listenForTransactions() -> Task<Void, Error> {
        return Task.detached {
            for await result in Transaction.updates {
                do {
                    let transaction = try self.checkVerified(result)
                    // Only finish and react to subscription transactions; credit consumables
                    // are handled and finished by CreditPurchaseManager.
                    let isSubscription = transaction.productID == self.weeklyProductID || transaction.productID == self.yearlyProductID
                    if isSubscription {
                        await transaction.finish()
                        await MainActor.run {
                            self.checkSubscriptionStatus()
                        }
                    }
                    // Credit products: do not finish here; CreditPurchaseManager owns them.
                } catch {
                    print("❌ Transaction verification failed: \(error)")
                }
            }
        }
    }
    
    private func checkVerified<T>(_ result: VerificationResult<T>) throws -> T {
        switch result {
        case .unverified:
            throw StoreError.failedVerification
        case .verified(let safe):
            return safe
        }
    }
    
    func loadProducts() {
        guard !isLoadingProducts else { return } // Prevent multiple simultaneous loads
        isLoadingProducts = true
        Task {
            do {
                let productIDs = [weeklyProductID, yearlyProductID]
                let storeProducts = try await Product.products(for: productIDs)
                
                await MainActor.run {
                    self.products = storeProducts
                    self.isLoadingProducts = false
                    print("✅ Loaded \(storeProducts.count) products from App Store")
                }
            } catch {
                await MainActor.run {
                    self.isLoadingProducts = false
                    print("❌ Error loading products: \(error.localizedDescription)")
                }
            }
        }
    }
    
    func getProduct(for productID: String) -> Product? {
        return products.first { $0.id == productID }
    }
    
    func getWeeklyProduct() -> Product? {
        return getProduct(for: weeklyProductID)
    }
    
    func getYearlyProduct() -> Product? {
        return getProduct(for: yearlyProductID)
    }
    
    // MARK: - Subscription Status
    func checkSubscriptionStatus() {
        Task {
            await updateSubscriptionStatus()
        }
    }
    
    @MainActor
    private func updateSubscriptionStatus() async {
        // Start with false - user is NOT Pro by default
        var isCurrentlySubscribed = false
        
        print("🔍 [Subscription Check] Starting subscription status check...")
        print("🔍 [Subscription Check] isPro initial value: \(isPro)")
        print("🔍 [Subscription Check] Checking for product IDs: \(weeklyProductID), \(yearlyProductID)")
        
        var transactionCount = 0
        var foundTransactions: [String] = []
        
        // Check current entitlements from StoreKit
        for await result in Transaction.currentEntitlements {
            transactionCount += 1
            do {
                let transaction = try checkVerified(result)
                foundTransactions.append(transaction.productID)
                
                print("🔍 [Subscription Check] Found transaction #\(transactionCount):")
                print("   - Product ID: \(transaction.productID)")
                print("   - Transaction ID: \(transaction.id)")
                
                // Check if transaction is for our subscription products
                if transaction.productID == weeklyProductID || transaction.productID == yearlyProductID {
                    print("   ✅ Matches our subscription products!")
                    
                    // Check if subscription is still active
                    if let expirationDate = transaction.expirationDate {
                        print("   - Expiration Date: \(expirationDate)")
                        print("   - Current Date: \(Date())")
                        print("   - Is Expired: \(expirationDate <= Date())")
                        
                        if expirationDate > Date() {
                            print("   ✅ Subscription is ACTIVE (not expired)")
                            isCurrentlySubscribed = true
                            await MainActor.run {
                                self.subscriptionExpirationDate = expirationDate
                                self.activeProductID = transaction.productID
                            }
                            break
                        } else {
                            print("   ⚠️ Subscription is EXPIRED")
                        }
                    } else {
                        // Non-expiring subscription (shouldn't happen with auto-renewable, but handle it)
                        print("   ⚠️ No expiration date (unusual for auto-renewable)")
                        isCurrentlySubscribed = true
                        await MainActor.run {
                            self.subscriptionExpirationDate = nil
                            self.activeProductID = transaction.productID
                        }
                        break
                    }
                } else {
                    print("   ❌ Not a subscription product (ignoring)")
                }
            } catch {
                print("❌ [Subscription Check] Error checking entitlement: \(error)")
            }
        }
        
        print("🔍 [Subscription Check] Total transactions found: \(transactionCount)")
        print("🔍 [Subscription Check] Found product IDs: \(foundTransactions)")
        print("🔍 [Subscription Check] isCurrentlySubscribed: \(isCurrentlySubscribed)")
        
        if !isCurrentlySubscribed {
            await MainActor.run {
                self.subscriptionExpirationDate = nil
                self.activeProductID = nil
            }
            print("🔍 [Subscription Check] No active subscription found - clearing subscription data")
        }
        
        // CRITICAL: Only set isPro to true if we verified an active subscription
        // Default is false - users must pay to access Pro features
        print("🔍 [Subscription Check] Setting isPro = \(isCurrentlySubscribed)")
        isPro = isCurrentlySubscribed
        print("📱 Subscription status: \(isPro ? "Pro" : "Free")")
        if let expiration = subscriptionExpirationDate {
            print("📅 Subscription expires: \(expiration)")
        } else {
            print("📅 No subscription expiration date")
        }
        
        // Reset/add credits when subscription status changes (if Pro)
        if isPro {
            print("🔍 [Subscription Check] User is Pro - checking credits reset/add")
            UsageTracker.shared.checkAndResetCreditsIfNeeded()
        } else {
            print("🔍 [Subscription Check] User is Free - keeping purchased credits")
            // Free users keep their purchased credits, just remove reset date
            UsageTracker.shared.checkAndResetCreditsIfNeeded()
        }
        
        // Notify other components of subscription status change
        NotificationCenter.default.post(name: NSNotification.Name("SubscriptionStatusChanged"), object: nil)
    }
    
    // MARK: - Purchase
    func purchase(_ product: Product) async throws -> Transaction? {
        let result = try await product.purchase()
        
        switch result {
        case .success(let verification):
            let transaction = try checkVerified(verification)
            await transaction.finish()
            await updateSubscriptionStatus()
            return transaction
        case .userCancelled:
            return nil
        case .pending:
            // Transaction is pending (e.g., waiting for parental approval)
            // StoreKit will deliver it via Transaction.updates when ready
            print("⏳ Purchase pending - will be delivered via Transaction.updates")
            return nil
        @unknown default:
            return nil
        }
    }
    
    // MARK: - Restore Purchases
    func restorePurchases() async {
        try? await AppStore.sync()
        await updateSubscriptionStatus()
    }
}

// MARK: - Store Error
enum StoreError: Error {
    case failedVerification
}

