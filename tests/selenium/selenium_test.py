import unittest
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.common.action_chains import ActionChains
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC


class SplitBrainTests(unittest.TestCase):

    def setUp(self):
        self.driver = webdriver.Chrome()
        self.driver.get("http://127.0.0.1:8000/")

    def tearDown(self):
        self.driver.quit()

    def test_element_locators(self):
        driver = self.driver

        generate_button = driver.find_element(By.ID, "generate-matrix")
        self.assertIsNotNone(generate_button)

        size_input = driver.find_element(By.XPATH, '//input[@id="matrix-size-input"]')
        self.assertIsNotNone(size_input)

        nav_button = driver.find_element(By.CLASS_NAME, "button")
        self.assertTrue(nav_button.is_displayed())

    def test_assertions(self):
        driver = self.driver

        size_input = driver.find_element(By.ID, "matrix-size-input")
        size_input.clear()
        size_input.send_keys("3")

        generate_button = driver.find_element(By.ID, "generate-matrix")
        generate_button.click()

        table = driver.find_element(By.TAG_NAME, "table")
        self.assertIsNotNone(table)

        rows = table.find_elements(By.TAG_NAME, "tr")
        self.assertEqual(len(rows), 4)

    def test_waits(self):
        driver = self.driver
        wait = WebDriverWait(driver, 10)

        size_input = driver.find_element(By.ID, "matrix-size-input")
        size_input.clear()
        size_input.send_keys("3")

        generate_button = driver.find_element(By.ID, "generate-matrix")
        generate_button.click()

        table = wait.until(EC.presence_of_element_located((By.TAG_NAME, "table")))
        self.assertIsNotNone(table)

        submit_button = driver.find_element(By.ID, "submit-matrix")
        submit_button.click()

        result_div = wait.until(EC.presence_of_element_located((By.ID, "result")))

        self.assertIn("Кластер мертвий", result_div.text)


if __name__ == "__main__":
    unittest.main()
