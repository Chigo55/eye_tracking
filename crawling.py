from selenium import webdriver
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.common.by import By
import time
import urllib.request
from datetime import datetime

driver = webdriver.Chrome()
driver.get("https://www.google.co.kr/imghp?hl=ko&ogbl")
elem = driver.find_element(By.NAME, "q")
elem.send_keys("face")
elem.send_keys(Keys.RETURN)
images = driver.find_elements(By.CSS_SELECTOR, ".rg_i.Q4LuWd")

t = datetime.now()
count = 1
for image in images:
    image.click()
    time.sleep(1)
    img_url = driver.find_element(By.CSS_SELECTOR, ".n3VNCb").get_attribute("src")

    img_name = str(t.year) + "." + str(t.month) + "." + str(t.day) + " - " + str(t.hour) + "." + str(t.minute) + "." + str(count)
    path = 'C:/Users/user/PycharmProjects/pythonProject/picture/'  # Folder
    full_name = img_name + ".png"
    urllib.request.urlretrieve(img_url, path+full_name)

    print(count)
    count += 1

    if count > 50:
        break

driver.close()