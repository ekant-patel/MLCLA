from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from webdriver_manager.chrome import ChromeDriverManager
from selenium.webdriver.common.by import By
import time

driver = webdriver.Chrome(service=Service(ChromeDriverManager().install()))

driver.get("https://quotes.toscrape.com/js/")

time.sleep(3)

quotes = driver.find_elements(By.CLASS_NAME, "text")
authors = driver.find_elements(By.CLASS_NAME, "author")

print("\nCollected Quotes from Dynamic Website:\n")

for q, a in zip(quotes, authors):
    print(f"{q.text} — {a.text}")

driver.quit()
