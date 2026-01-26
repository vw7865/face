//
//  CreditPurchaseManager.swift
//  LooksmaxAI
//
//  Created by Admin on 2026. 01. 25..
//

import Foundation
import StoreKit

class CreditPurchaseManager: ObservableObject {
    static let shared = CreditPurchaseManager()
    
    // Product IDs from App Store Connect (consumable products)
    private let creditProductIDs = [
        "com.facemaxxing.LooksmaxAI.Credits.5",
        "com.facemaxxing.LooksmaxAI.Credits.10",
        "com.facemaxxing.LooksmaxAI.Credits.25",
        "com.facemaxxing.LooksmaxAI.Credits.50"
    ]
    
    @Published var creditProducts: [Product] = []
    @Published var isLoadingProducts: Bool = false
    @Published var purchaseError: String? = nil
    
    private var updateListenerTask: Task<Void, Error>?
    
    // Track processed transaction IDs to prevent double-processing
    private let processedTransactionsKey = "ProcessedCreditTransactions"
    private var processedTransactionIDs: Set<UInt64> {
        get {
            let array = UserDefaults.standard.array(forKey: processedTransactionsKey) as? [UInt64] ?? []
            return Set(array)
        }
        set {
            UserDefaults.standard.set(Array(newValue), forKey: processedTransactionsKey)
        }
    }
    
    private init() {
        // Start listening for transaction updates
        updateListenerTask = listenForTransactions()
        loadCreditProducts()
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
                    
                    // Check if this is a credit purchase transaction
                    if self.creditProductIDs.contains(transaction.productID) {
                        // Process the credit purchase (will check for duplicates internally)
                        await self.processCreditPurchase(transaction)
                    }
                    
                    // Always finish consumable transactions
                    await transaction.finish()
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
    
    // MARK: - Load Products
    func loadCreditProducts() {
        guard !isLoadingProducts else { return }
        isLoadingProducts = true
        
        Task {
            do {
                let products = try await Product.products(for: creditProductIDs)
                
                await MainActor.run {
                    self.creditProducts = products.sorted { product1, product2 in
                        // Sort by credit amount (extract number from product ID)
                        let amount1 = self.getCreditAmount(from: product1.id)
                        let amount2 = self.getCreditAmount(from: product2.id)
                        return amount1 < amount2
                    }
                    self.isLoadingProducts = false
                    print("✅ Loaded \(products.count) credit products from App Store")
                }
            } catch {
                await MainActor.run {
                    self.isLoadingProducts = false
                    print("❌ Error loading credit products: \(error.localizedDescription)")
                }
            }
        }
    }
    
    private func getCreditAmount(from productID: String) -> Int {
        // Extract credit amount from product ID (e.g., "com.facemaxxing.LooksmaxAI.Credits.50" -> 50)
        if let lastComponent = productID.components(separatedBy: ".").last,
           let amount = Int(lastComponent) {
            return amount
        }
        return 0
    }
    
    // MARK: - Purchase Credits
    func purchaseCredits(_ product: Product) async throws -> Bool {
        let result = try await product.purchase()
        
        switch result {
        case .success(let verification):
            let transaction = try checkVerified(verification)
            
            // Process the credit purchase
            await processCreditPurchase(transaction)
            
            // Finish the transaction (required for consumables)
            await transaction.finish()
            
            return true
            
        case .userCancelled:
            return false
            
        case .pending:
            // Transaction is pending (e.g., waiting for parental approval)
            // StoreKit will deliver it via Transaction.updates when ready
            print("⏳ Credit purchase pending - will be delivered via Transaction.updates")
            return false
            
        @unknown default:
            return false
        }
    }
    
    // MARK: - Process Credit Purchase
    @MainActor
    private func processCreditPurchase(_ transaction: Transaction) async {
        let transactionID = transaction.id
        var processedIDs = processedTransactionIDs
        
        // Check if this transaction has already been processed
        guard !processedIDs.contains(transactionID) else {
            print("⚠️ Transaction \(transactionID) already processed, skipping")
            return
        }
        
        let productID = transaction.productID
        let creditAmount = getCreditAmount(from: productID)
        
        guard creditAmount > 0 else {
            print("⚠️ Invalid credit amount for product: \(productID)")
            return
        }
        
        // Mark transaction as processed BEFORE adding credits (prevents double-processing)
        processedIDs.insert(transactionID)
        processedTransactionIDs = processedIDs
        
        // Add credits to user's account
        UsageTracker.shared.addImageGenerationCredits(creditAmount)
        
        print("✅ Credit purchase successful: Added \(creditAmount) credits (Transaction ID: \(transactionID))")
    }
    
    // MARK: - Get Product for Credit Amount
    func getProduct(for creditAmount: Int) -> Product? {
        let productID = "com.facemaxxing.LooksmaxAI.Credits.\(creditAmount)"
        return creditProducts.first { $0.id == productID }
    }
    
    // MARK: - Restore Purchases
    func restorePurchases() async {
        // For consumables, we check all transactions and process any unprocessed ones
        // Note: Consumables are typically not restorable, but we can check for any pending transactions
        try? await AppStore.sync()
    }
}
