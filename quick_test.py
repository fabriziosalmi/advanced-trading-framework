#!/usr/bin/env python3
"""
Quick test to verify sidebar is now visible
"""

import asyncio
from playwright.async_api import async_playwright

async def quick_test():
    async with async_playwright() as p:
        browser = await p.chromium.launch(headless=True)
        page = await browser.new_page()
        await page.set_viewport_size({"width": 1920, "height": 1080})

        try:
            await page.goto("http://localhost:8000", wait_until="networkidle")
            await page.wait_for_timeout(2000)

            # Check sidebar position
            sidebar = await page.query_selector("#sidebar")
            if sidebar:
                box = await sidebar.bounding_box()
                print(f"✅ Sidebar position: {box}")

                if box and box['x'] >= 0:
                    print("🎉 SUCCESS: Sidebar is now visible (x >= 0)!")
                else:
                    print("❌ STILL HIDDEN: Sidebar is off-screen")

                # Check for menu items
                nav_links = await page.query_selector_all(".nav-link")
                print(f"🔗 Found {len(nav_links)} navigation links")

            await page.screenshot(path="sidebar_test.png", full_page=True)
            print("📸 Test screenshot saved as sidebar_test.png")

        finally:
            await browser.close()

if __name__ == "__main__":
    asyncio.run(quick_test())