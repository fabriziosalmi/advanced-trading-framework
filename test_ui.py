#!/usr/bin/env python3
"""
Test UI with Playwright to validate sidebar menu visibility
"""

import asyncio
from playwright.async_api import async_playwright
import time

async def test_sidebar_menu():
    async with async_playwright() as p:
        browser = await p.chromium.launch(headless=False)
        page = await browser.new_page()

        # Set viewport size
        await page.set_viewport_size({"width": 1920, "height": 1080})

        try:
            print("🌐 Navigating to http://localhost:8000...")
            await page.goto("http://localhost:8000", wait_until="networkidle")

            # Wait for page to load completely
            await page.wait_for_timeout(3000)

            # Take initial screenshot
            await page.screenshot(path="screenshot_initial.png", full_page=True)
            print("📸 Initial screenshot saved as screenshot_initial.png")

            # Check if sidebar exists
            sidebar = await page.query_selector("#sidebar")
            if sidebar:
                print("✅ Sidebar element found")

                # Check sidebar classes
                sidebar_classes = await sidebar.get_attribute("class")
                print(f"📝 Sidebar classes: {sidebar_classes}")

                # Check if sidebar is visible
                is_visible = await sidebar.is_visible()
                print(f"👁️  Sidebar visible: {is_visible}")

                # Get sidebar dimensions
                box = await sidebar.bounding_box()
                if box:
                    print(f"📏 Sidebar dimensions: {box}")
                else:
                    print("❌ Sidebar has no bounding box (not rendered)")

            else:
                print("❌ Sidebar element not found")

            # Check for menu sections
            menu_sections = await page.query_selector_all(".menu-section")
            print(f"📋 Found {len(menu_sections)} menu sections")

            # Check for menu items
            menu_items = await page.query_selector_all(".menu-items")
            print(f"📝 Found {len(menu_items)} menu item containers")

            for i, item in enumerate(menu_items):
                classes = await item.get_attribute("class")
                is_visible = await item.is_visible()
                print(f"   Menu item {i+1}: classes='{classes}', visible={is_visible}")

            # Check for navigation links
            nav_links = await page.query_selector_all(".nav-link")
            print(f"🔗 Found {nav_links.__len__()} navigation links")

            # Check if JavaScript loaded
            js_loaded = await page.evaluate("() => window.tradingUI !== undefined")
            print(f"⚡ JavaScript loaded: {js_loaded}")

            # Check for any JavaScript errors
            await page.wait_for_timeout(1000)

            # Try clicking on Trading section to test collapsible
            trading_header = await page.query_selector('[data-section="trading"]')
            if trading_header:
                print("🖱️  Clicking on Trading section header...")
                await trading_header.click()
                await page.wait_for_timeout(1000)

                # Take screenshot after click
                await page.screenshot(path="screenshot_after_click.png", full_page=True)
                print("📸 Screenshot after click saved as screenshot_after_click.png")

            # Wait a bit more and take final screenshot
            await page.wait_for_timeout(2000)
            await page.screenshot(path="screenshot_final.png", full_page=True)
            print("📸 Final screenshot saved as screenshot_final.png")

        except Exception as e:
            print(f"❌ Error: {e}")
            await page.screenshot(path="screenshot_error.png", full_page=True)
            print("📸 Error screenshot saved as screenshot_error.png")

        finally:
            await browser.close()

if __name__ == "__main__":
    asyncio.run(test_sidebar_menu())