#!/usr/bin/env python3
"""
Script to update all HTML files with consistent navigation
"""

import re
from pathlib import Path

# Define the standard navbar template
NAVBAR_TEMPLATE = '''<!-- ================= NAVBAR ================= -->
<nav class="text-white px-6 py-4 shadow-md mb-8" style="background-color:#457b9d;">
  <div class="max-w-7xl mx-auto flex justify-between items-center">
    <div class="text-xl font-bold">💼 Financial Analyzer</div>
    <div class="flex space-x-4">
      <a href="/"
         class="px-4 py-2 rounded-lg {HOME_ACTIVE} hover:bg-white hover:bg-opacity-20 transition">
        Home
      </a>
      <a href="/dcf"
         class="px-4 py-2 rounded-lg {DCF_ACTIVE} hover:bg-white hover:bg-opacity-20 transition">
        DCF Valuation
      </a>
      <a href="/financial-modelling"
         class="px-4 py-2 rounded-lg {FM_ACTIVE} hover:bg-white hover:bg-opacity-20 transition">
        Financial Modelling
      </a>
      <a href="/technical-analysis"
         class="px-4 py-2 rounded-lg {TA_ACTIVE} hover:bg-white hover:bg-opacity-20 transition">
        Technical Analysis
      </a>
      <a href="/portfolio-management"
         class="px-4 py-2 rounded-lg {PM_ACTIVE} hover:bg-white hover:bg-opacity-20 transition">
        Portfolio Management
      </a>
      <a href="/performance-analytics"
         class="px-4 py-2 rounded-lg {PA_ACTIVE} hover:bg-white hover:bg-opacity-20 transition">
        Performance Analytics
      </a>
      <a href="/strategy-optimization"
         class="px-4 py-2 rounded-lg {SO_ACTIVE} hover:bg-white hover:bg-opacity-20 transition">
        Strategy Optimization
      </a>
    </div>
  </div>
</nav>
<!-- =============== NAVBAR END =============== -->'''

# Define active page for each file
PAGE_CONFIG = {
    'financialanalyzerweb.html': {
        'active': 'HOME',
        'input': '/mnt/user-data/uploads/financialanalyzerweb.html',
        'output': '/mnt/user-data/outputs/financialanalyzerweb.html'
    },
    'dcfvaluation.html': {
        'active': 'DCF',
        'input': '/mnt/user-data/uploads/dcfvaluation.html',
        'output': '/mnt/user-data/outputs/dcfvaluation.html'
    },
    'financial_modelling.html': {
        'active': 'FM',
        'input': '/mnt/user-data/uploads/financial_modelling.html',
        'output': '/mnt/user-data/outputs/financial_modelling.html'
    },
    'technical_analysis.html': {
        'active': 'TA',
        'input': '/mnt/user-data/uploads/technical_analysis.html',
        'output': '/mnt/user-data/outputs/technical_analysis.html'
    },
    'portfolio_management.html': {
        'active': 'PM',
        'input': '/mnt/user-data/uploads/portfolio_management.html',
        'output': '/mnt/user-data/outputs/portfolio_management.html'
    },
    'performance_analytics.html': {
        'active': 'PA',
        'input': '/mnt/user-data/uploads/performance_analytics.html',
        'output': '/mnt/user-data/outputs/performance_analytics.html'
    },
    'strategy_optimization.html': {
        'active': 'SO',
        'input': '/mnt/user-data/uploads/strategy_optimization.html',
        'output': '/mnt/user-data/outputs/strategy_optimization.html'
    }
}

def get_navbar_for_page(active_page):
    """Generate navbar with correct active state"""
    active_class = "bg-white bg-opacity-20 font-semibold"
    
    navbar = NAVBAR_TEMPLATE
    navbar = navbar.replace('{HOME_ACTIVE}', active_class if active_page == 'HOME' else '')
    navbar = navbar.replace('{DCF_ACTIVE}', active_class if active_page == 'DCF' else '')
    navbar = navbar.replace('{FM_ACTIVE}', active_class if active_page == 'FM' else '')
    navbar = navbar.replace('{TA_ACTIVE}', active_class if active_page == 'TA' else '')
    navbar = navbar.replace('{PM_ACTIVE}', active_class if active_page == 'PM' else '')
    navbar = navbar.replace('{PA_ACTIVE}', active_class if active_page == 'PA' else '')
    navbar = navbar.replace('{SO_ACTIVE}', active_class if active_page == 'SO' else '')
    
    return navbar

def update_html_file(input_path, output_path, active_page):
    """Update HTML file with new navbar"""
    try:
        with open(input_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # Pattern to match existing navbar
        navbar_pattern = r'<!-- =+\s*NAVBAR\s*=+ -->.*?<!-- =+\s*NAVBAR END\s*=+ -->'
        
        # Get new navbar
        new_navbar = get_navbar_for_page(active_page)
        
        # Replace old navbar with new one
        updated_content = re.sub(navbar_pattern, new_navbar, content, flags=re.DOTALL)
        
        # Write updated content
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(updated_content)
        
        print(f"✅ Updated: {Path(output_path).name}")
        return True
        
    except Exception as e:
        print(f"❌ Error updating {Path(input_path).name}: {str(e)}")
        return False

def main():
    """Main execution"""
    print("=" * 60)
    print("Financial Analyzer - Navbar Update Script")
    print("=" * 60)
    print()
    
    success_count = 0
    fail_count = 0
    
    for filename, config in PAGE_CONFIG.items():
        print(f"Processing {filename} (Active: {config['active']})...")
        if update_html_file(config['input'], config['output'], config['active']):
            success_count += 1
        else:
            fail_count += 1
    
    print()
    print("=" * 60)
    print(f"✅ Successfully updated: {success_count} files")
    if fail_count > 0:
        print(f"❌ Failed: {fail_count} files")
    print("=" * 60)
    print()
    print("Updated files are in: /mnt/user-data/outputs/")
    print()
    print("Next steps:")
    print("1. Review the updated files")
    print("2. Replace the original files in your templates folder")
    print("3. Restart your Flask server")
    print("4. Test navigation across all pages")

if __name__ == '__main__':
    main()
