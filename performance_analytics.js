// 🔹 FIXED FUNCTION: Fetch from localStorage instead of API
async function fetchFromPortfolio() {
    try {
        showLoading();
        hideError();
        
        // 🔹 NEW: Read from localStorage (set by Portfolio Management)
        const savedData = localStorage.getItem('portfolio_analysis');
        
        if (!savedData) {
            throw new Error('Portfolio data not found. Please analyze your portfolio in Portfolio Management first.');
        }
        
        const portfolioData = JSON.parse(savedData);
        
        // Check if data is recent (within last 24 hours)
        const dataAge = new Date() - new Date(portfolioData.timestamp);
        const hoursOld = Math.floor(dataAge / (1000 * 60 * 60));
        
        if (hoursOld > 24) {
            showInfo(`Portfolio data is ${hoursOld} hours old. Consider re-analyzing for latest prices.`);
        }
        
        // Transform portfolio data into performance analytics format
        const performanceData = transformPortfolioData(portfolioData);
        
        // Analyze
        await analyzePerformance(performanceData);
        
        // Update status
        document.getElementById('portfolioStatus').textContent = 'Connected ✓';
        document.getElementById('portfolioStatus').className = 'font-bold text-green-600';
        document.getElementById('lastAnalysis').textContent = new Date(portfolioData.timestamp).toLocaleString();
        
    } catch (error) {
        hideLoading();
        showError(error.message);
    }
}

// 🔹 NEW: Transform Portfolio data to Performance Analytics format
function transformPortfolioData(savedData) {
    const result = savedData.result;
    const holdings = savedData.holdings;
    
    if (!result || !result.success) {
        throw new Error('Invalid portfolio data');
    }
    
    const summary = result.summary || {};
    const currentValue = summary.total_current_value || 0;
    const initialValue = summary.total_invested || 0;
    
    // Generate historical data (simulated 365 days)
    const holdingsHistory = generateSimulatedHistory(initialValue, currentValue, 365);
    
    // Extract holdings detail
    const holdingsDetail = (result.holdings || []).map(h => ({
        symbol: h.symbol,
        quantity: h.quantity,
        buy_price: h.buy_price,
        current_price: h.current_price,
        buy_date: h.buy_date || '2024-01-01',
        gain_loss: h.gain_loss || 0,
        gain_loss_pct: h.gain_loss_pct || 0,
        current_value: h.current_value || 0,
        invested_value: h.invested_value || 0
    }));
    
    // Generate cash flows based on buy dates
    const cashFlows = generateCashFlows(holdings, initialValue);
    
    return {
        holdings_history: holdingsHistory,
        holdings_detail: holdingsDetail,
        cash_flows: cashFlows,
        trades: null,
        signals: null
    };
}

// 🔹 NEW: Generate simulated historical data
function generateSimulatedHistory(initialValue, currentValue, days) {
    if (initialValue === 0 || currentValue === 0) {
        return [];
    }
    
    const history = [];
    const totalReturn = (currentValue - initialValue) / initialValue;
    const startDate = new Date();
    startDate.setDate(startDate.getDate() - days);
    
    for (let i = 0; i <= days; i++) {
        const date = new Date(startDate);
        date.setDate(date.getDate() + i);
        
        const progress = i / days;
        
        // Add realistic volatility
        const volatility = (Math.random() - 0.5) * 0.04; // ±2% daily
        const trendReturn = totalReturn * progress;
        let value = initialValue * (1 + trendReturn + volatility);
        
        // Ensure value doesn't go too low
        value = Math.max(value, initialValue * 0.6);
        
        history.push({
            date: date.toISOString().split('T')[0],
            total_value: Math.round(value * 100) / 100
        });
    }
    
    // Ensure last value matches current value exactly
    history[history.length - 1].total_value = currentValue;
    
    return history;
}

// 🔹 NEW: Generate cash flows from holdings
function generateCashFlows(holdings, totalInvestment) {
    const cashFlows = [];
    
    // Group by buy date
    const dateGroups = {};
    
    holdings.forEach(h => {
        const date = h.buy_date || '2024-01-01';
        if (!dateGroups[date]) {
            dateGroups[date] = 0;
        }
        dateGroups[date] += h.quantity * h.buy_price;
    });
    
    // Convert to cash flow array
    Object.entries(dateGroups).forEach(([dateStr, amount]) => {
        cashFlows.push({
            date: new Date(dateStr),
            amount: Math.round(amount * 100) / 100
        });
    });
    
    // Sort by date
    cashFlows.sort((a, b) => a.date - b.date);
    
    // If no cash flows, create one initial investment
    if (cashFlows.length === 0) {
        const oneYearAgo = new Date();
        oneYearAgo.setFullYear(oneYearAgo.getFullYear() - 1);
        
        cashFlows.push({
            date: oneYearAgo,
            amount: totalInvestment
        });
    }
    
    return cashFlows;
}

// 🔹 NEW: Show info message
function showInfo(message) {
    const statusDiv = document.getElementById('status');
    if (statusDiv) {
        statusDiv.textContent = '⚠️ ' + message;
        statusDiv.className = 'text-center font-semibold mb-4 text-yellow-600';
        statusDiv.classList.remove('hidden');
        
        setTimeout(() => {
            statusDiv.classList.add('hidden');
        }, 5000);
    }
}
