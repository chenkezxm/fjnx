from time import sleep
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from loginconf import loginconf


class Driver(object):
    def __init__(self, driver=None):
        if not driver:
            chrome_options = Options()
            chrome_options.add_argument('--disable-gpu')
            chrome_options.add_argument('--disable-blink-features=AutomationControlled')
            self.driver = webdriver.Chrome(executable_path="chromedriver.exe", options=chrome_options)
            #self.driver = webdriver.PhantomJS(executable_path="phantomjs.exe")
        else:
            self.driver = driver
        self.browser = []

    def get_driver(self):
        return self.driver

    def switch_window_by_title(self, title):
        for handle in self.driver.window_handles:
            self.driver.switch_to.window(handle)
            if self.driver.title == title:
                return True
        return False

    def left_one(self, url=""):
        if url:
            self.driver.get(url)
        if len(self.driver.window_handles) > 1:
            current = self.driver.current_window_handle
            for handle in self.driver.window_handles:
                if handle != current:
                    self.driver.switch_to.window(handle)
                    self.driver.close()
            self.driver.switch_to.window(current)
        self.browser = []
        self.browser_add()

    def browser_switch_new(self):
        for handle in self.driver.window_handles:
            if handle not in self.browser:
                self.driver.switch_to.window(handle)
        self.browser_add()

    def browser_switch_by_number(self, number):
        self.driver.switch_to.window(self.browser[number])

    def browser_add(self):
        self.browser.append(self.driver.current_window_handle)

    def browser_pop(self):
        self.driver.switch_to.window(self.browser.pop())
        self.driver.close()
        self.driver.switch_to.window(self.browser[-1])

    def login(self):
        self.driver.get("https://eln.fjnx.com.cn/")
        self.browser_add()
        self.driver.find_element_by_id("loginName").send_keys(loginconf["loginname"])
        self.driver.find_element_by_id("swInput").send_keys(loginconf["loginpwd"])
        self.captcha()
        self.driver.find_element_by_class_name("ulogin").click()
        sleep(5)

    def logout(self):
        self.browser_switch_by_number(0)
        self.driver.find_element_by_xpath("//button[@title='退出平台']").click()
        sleep(5)
        self.driver.find_element_by_class_name("new-tbc-btn").click()
        sleep(5)
        self.driver.quit()

    def captcha(self):
        self.driver.find_element_by_id("securityCode").send_keys(input("验证码:"))
        sleep(5)

    def close(self):
        self.driver.quit()
