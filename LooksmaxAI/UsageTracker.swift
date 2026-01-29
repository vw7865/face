//
//  UsageTracker.swift
//  LooksmaxAI
//
//  Created by Admin on 2026. 01. 18..
//

import Foundation

class UsageTracker: ObservableObject {
    static let shared = UsageTracker()
    
    // Free tier limits
    private let maxFreeFaceRatings = 1  // Reduced to 1 to prevent abuse
    private let maxFreeRizzCoach = 1
    private let maxFreeLooksmaxxing = 1
    private let maxFreeHistoryResults = 3
    
    // Pro subscription credits (monthly reset)
    private let weeklyMonthlyCredits = 30  // Credits for weekly subscribers (per month)
    private let yearlyMonthlyCredits = 30  // Credits for yearly subscribers (per month) - same as weekly
    private let defaultMonthlyCredits = 30  // Fallback/default credits - same as subscription credits
    
    // UserDefaults keys
    private let faceRatingsCountKey = "faceRatingsCount"
    private let rizzCoachCountKey = "rizzCoachCount"
    private let looksmaxxingCountKey = "looksmaxxingCount"
    private let imageGenerationCreditsKey = "imageGenerationCredits"
    private let lastCreditResetDateKey = "lastCreditResetDate"
    
    @Published var faceRatingsCount: Int = 0
    @Published var rizzCoachCount: Int = 0
    @Published var looksmaxxingCount: Int = 0
    @Published var imageGenerationCredits: Int = 0
    
    private init() {
        loadUsage()
        // Observe subscription changes
        NotificationCenter.default.addObserver(
            forName: NSNotification.Name("SubscriptionStatusChanged"),
            object: nil,
            queue: .main
        ) { [weak self] _ in
            // Refresh usage display when subscription changes
            self?.objectWillChange.send()
            // Check if credits need to be reset when subscription status changes
            self?.checkAndResetCreditsIfNeeded()
        }
    }
    
    private func loadUsage() {
        faceRatingsCount = UserDefaults.standard.integer(forKey: faceRatingsCountKey)
        rizzCoachCount = UserDefaults.standard.integer(forKey: rizzCoachCountKey)
        looksmaxxingCount = UserDefaults.standard.integer(forKey: looksmaxxingCountKey)
        
        print("🔍 [UsageTracker] Loading usage data...")
        print("🔍 [UsageTracker] SubscriptionManager.shared.isPro = \(SubscriptionManager.shared.isPro)")
        
        // Load credits for all users (free users can purchase credits)
        let storedCredits = UserDefaults.standard.integer(forKey: imageGenerationCreditsKey)
        print("🔍 [UsageTracker] Loading credits from UserDefaults: \(storedCredits)")
        imageGenerationCredits = storedCredits
        
        // Only check monthly reset for Pro users (they get monthly credits)
        if SubscriptionManager.shared.isPro {
            checkAndResetCreditsIfNeeded()
        }
    }
    
    // MARK: - Credit Management
    func checkAndResetCreditsIfNeeded() {
        print("🔍 [UsageTracker] checkAndResetCreditsIfNeeded() called")
        print("🔍 [UsageTracker] SubscriptionManager.shared.isPro = \(SubscriptionManager.shared.isPro)")
        print("🔍 [UsageTracker] Current credits before check: \(imageGenerationCredits)")
        
        guard SubscriptionManager.shared.isPro else {
            // Free users don't get monthly credits, but keep their purchased credits
            // Don't clear credits - free users can purchase credits
            print("🔍 [UsageTracker] User is NOT Pro - keeping purchased credits: \(imageGenerationCredits)")
            // Remove reset date since they're not Pro (but keep credits)
            UserDefaults.standard.removeObject(forKey: lastCreditResetDateKey)
            return
        }
        
        print("🔍 [UsageTracker] User is Pro - checking credit reset")
        let lastResetDate = UserDefaults.standard.object(forKey: lastCreditResetDateKey) as? Date
        
        // Check if we need to reset credits (monthly reset)
        if let lastReset = lastResetDate {
            print("🔍 [UsageTracker] Last reset date: \(lastReset)")
            let calendar = Calendar.current
            if let daysSinceReset = calendar.dateComponents([.day], from: lastReset, to: Date()).day {
                print("🔍 [UsageTracker] Days since reset: \(daysSinceReset)")
                if daysSinceReset >= 30 {
                    // Reset credits for new month
                    print("🔍 [UsageTracker] 30+ days passed - resetting credits")
                    resetMonthlyCredits()
                } else {
                    print("🔍 [UsageTracker] Less than 30 days - keeping current credits: \(imageGenerationCredits)")
                }
            }
        } else {
            // First time becoming Pro - ADD monthly credits to existing purchased credits
            print("🔍 [UsageTracker] No reset date found - first time Pro, adding monthly credits to existing balance")
            addInitialProCredits()
        }
    }
    
    private func addInitialProCredits() {
        // Determine credits based on subscription type
        let creditsToAdd = getCreditsForSubscription()
        let existingCredits = imageGenerationCredits
        imageGenerationCredits = existingCredits + creditsToAdd
        UserDefaults.standard.set(imageGenerationCredits, forKey: imageGenerationCreditsKey)
        UserDefaults.standard.set(Date(), forKey: lastCreditResetDateKey)
        print("➕ Added initial Pro credits: \(creditsToAdd) credits added to existing \(existingCredits) credits = \(imageGenerationCredits) total")
    }
    
    private func resetMonthlyCredits() {
        // Determine credits based on subscription type
        // This is for monthly resets - replace credits with monthly allowance
        let creditsToGive = getCreditsForSubscription()
        imageGenerationCredits = creditsToGive
        UserDefaults.standard.set(imageGenerationCredits, forKey: imageGenerationCreditsKey)
        UserDefaults.standard.set(Date(), forKey: lastCreditResetDateKey)
        print("🔄 Reset monthly credits: \(creditsToGive) credits (monthly reset)")
    }
    
    private func getCreditsForSubscription() -> Int {
        let subscriptionManager = SubscriptionManager.shared
        
        // Check which subscription type user has
        if let productID = subscriptionManager.activeProductID {
            if productID == "com.facemaxxing.LooksmaxAI.Weekly" {
                return weeklyMonthlyCredits
            } else if productID == "com.facemaxxing.LooksmaxAI.Yearly" {
                return yearlyMonthlyCredits
            }
        }
        
        // Default/fallback
        return defaultMonthlyCredits
    }
    
    func canUseImageGeneration() -> Bool {
        // All users can use image generation if they have credits
        // Pro users get monthly credits, free users can purchase credits
        return imageGenerationCredits > 0
    }
    
    func getImageGenerationCreditsRemaining() -> Int {
        // Return credits for all users (free users can purchase credits)
        return imageGenerationCredits
    }
    
    func useImageGenerationCredit() -> Bool {
        // All users can use credits (Pro users get monthly credits, free users can purchase)
        guard imageGenerationCredits > 0 else {
            return false
        }
        
        imageGenerationCredits -= 1
        UserDefaults.standard.set(imageGenerationCredits, forKey: imageGenerationCreditsKey)
        print("💳 Used 1 credit. Remaining: \(imageGenerationCredits)")
        return true
    }
    
    func addImageGenerationCredits(_ amount: Int) {
        imageGenerationCredits += amount
        UserDefaults.standard.set(imageGenerationCredits, forKey: imageGenerationCreditsKey)
        print("➕ Added \(amount) credits. Total: \(imageGenerationCredits)")
    }
    
    // MARK: - Face Ratings
    func canUseFaceRating() -> Bool {
        if SubscriptionManager.shared.isPro {
            return true
        }
        return faceRatingsCount < maxFreeFaceRatings
    }
    
    func incrementFaceRating() {
        guard !SubscriptionManager.shared.isPro else { return }
        faceRatingsCount += 1
        UserDefaults.standard.set(faceRatingsCount, forKey: faceRatingsCountKey)
    }
    
    func getFaceRatingRemaining() -> Int {
        if SubscriptionManager.shared.isPro {
            return -1 // Unlimited
        }
        return max(0, maxFreeFaceRatings - faceRatingsCount)
    }
    
    // MARK: - Rizz Coach
    func canUseRizzCoach() -> Bool {
        if SubscriptionManager.shared.isPro {
            return true
        }
        return rizzCoachCount < maxFreeRizzCoach
    }
    
    func incrementRizzCoach() {
        guard !SubscriptionManager.shared.isPro else { return }
        rizzCoachCount += 1
        UserDefaults.standard.set(rizzCoachCount, forKey: rizzCoachCountKey)
    }
    
    func hasUsedRizzCoach() -> Bool {
        return rizzCoachCount >= maxFreeRizzCoach
    }
    
    // MARK: - Looksmaxxing
    func canUseLooksmaxxing() -> Bool {
        if SubscriptionManager.shared.isPro {
            return true
        }
        return looksmaxxingCount < maxFreeLooksmaxxing
    }
    
    func incrementLooksmaxxing() {
        guard !SubscriptionManager.shared.isPro else { return }
        looksmaxxingCount += 1
        UserDefaults.standard.set(looksmaxxingCount, forKey: looksmaxxingCountKey)
    }
    
    func hasUsedLooksmaxxing() -> Bool {
        return looksmaxxingCount >= maxFreeLooksmaxxing
    }
    
    // MARK: - History
    func getMaxHistoryResults() -> Int {
        if SubscriptionManager.shared.isPro {
            return Int.max // Unlimited
        }
        return maxFreeHistoryResults
    }
    
    // MARK: - Reset (for testing)
    func resetUsage() {
        faceRatingsCount = 0
        rizzCoachCount = 0
        looksmaxxingCount = 0
        imageGenerationCredits = 0
        UserDefaults.standard.removeObject(forKey: faceRatingsCountKey)
        UserDefaults.standard.removeObject(forKey: rizzCoachCountKey)
        UserDefaults.standard.removeObject(forKey: looksmaxxingCountKey)
        UserDefaults.standard.removeObject(forKey: imageGenerationCreditsKey)
        UserDefaults.standard.removeObject(forKey: lastCreditResetDateKey)
    }
}

