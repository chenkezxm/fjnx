import random
from time import time, sleep

from selenium import webdriver
from selenium.common.exceptions import NoSuchElementException, NoSuchFrameException, StaleElementReferenceException, \
    ElementClickInterceptedException
from selenium.webdriver.chrome.options import Options

from PIL import Image, ImageFont, ImageDraw, ImageEnhance
from MyCaptcha import GenerateCaptcha, UseModel

loginconf = {
    'loginname': "chenke",
    'loginpwd': "Cshzxm100100"
}


class FjnxGenerateCaptcha(GenerateCaptcha):
    @staticmethod
    def get_width():
        return 58

    @staticmethod
    def get_height():
        return 22

    @staticmethod
    def rndColor():
        return random.randint(100, 255), random.randint(100, 255), random.randint(100, 255)

    def get_captcha(self, num, path):
        for i in range(num):
            image = Image.new('RGB', (self.get_width(), self.get_height()), (255, 255, 255))
            font = ImageFont.truetype('C:/Windows/Fonts/Arial.ttf', 15)
            draw = ImageDraw.Draw(image)
            name = self.get_string()
            for t in range(4):
                draw.text((12 * t + 6, 5), name[t], font=font, fill=(0, 0, 0))
            image.save(path + name + ".jpg", 'jpeg')

    def build_data(self):
        self.get_captcha(5000, "captcha/train/")
        self.get_captcha(500, "captcha/test/")


class Driver(object):
    def __init__(self):
        chrome_options = Options()
        chrome_options.add_argument('--disable-gpu')
        chrome_options.add_argument('--disable-blink-features=AutomationControlled')
        self.driver = webdriver.Chrome(executable_path="chromedriver.exe", options=chrome_options)
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

    def get_captcha(self, file):
        board = 1
        self.driver.save_screenshot("pic/temp.png")
        yzm = self.driver.find_element_by_class_name("yzmImg")
        img = Image.open("pic/temp.png")
        img = img.crop((
            yzm.location['x'] + board,
            yzm.location['y'] + board,
            yzm.location['x'] + yzm.size['width'] - board,
            yzm.location['y'] + yzm.size['height'] - board
        ))
        img = img.convert("RGB")
        img = ImageEnhance.Color(img).enhance(0.0)
        img = ImageEnhance.Brightness(img).enhance(2.0)
        for x in range(img.size[0]):
            for y in range(img.size[1]):
                if img.getpixel((x, y)) < (200, 200, 200):
                    img.putpixel((x, y), (0, 0, 0))
                else:
                    img.putpixel((x, y), (255, 255, 255))
        img.save(file)

    def captcha(self):
        incorrect = False
        self.get_captcha("pic/screenshot.jpg")
        yzm = UseModel().run_file("pic/screenshot.jpg")
        self.driver.find_element_by_id("securityCode").send_keys(yzm)
        sleep(5)
        self.driver.find_element_by_class_name("ulogin").click()
        sleep(5)
        while self.driver.title == '福万通网络学院':
            if self.driver.find_element_by_id("error").text == '验证码不正确!':
                incorrect = True
                yzm = input("验证码:")
                self.get_captcha("pic/" + yzm + ".jpg")
                self.driver.find_element_by_id("securityCode").send_keys(yzm)
                self.driver.find_element_by_class_name("ulogin").click()
                sleep(5)
            sleep(5)
        if incorrect:
            from shutil import move
            move("pic/" + yzm + ".jpg", "train/" + yzm + ".jpg")

    def close(self):
        self.driver.quit()


class StrTime(object):
    @staticmethod
    def verify(text):
        if isinstance(text, str):
            return True
        return False

    def second(self, text):
        if not self.verify(text):
            raise ValueError
        second_num = 0
        for x in text.split(":"):
            second_num = second_num * 60 + int(x)
        return second_num

    def minute(self, text):
        return self.second(text) / 60

    def hour(self, text):
        return self.minute(text) / 60

    def minus(self, text1, text2):
        return self.second(text1) - self.second(text2)


class Study:
    def __init__(self, controller):
        for x in Study.StudyBase.__subclasses__():
            self.study = x(controller)
            if self.study.test():
                print(x)
                self.study.study()
                break

    class StudyBase:
        def __init__(self, controller):
            self.controller = controller
            self.cal_time = StrTime()
            self.last_move_time = time()
            self.time_diff = 29000

        def test(self):
            pass

        def study(self):
            pass

        def stop_quit(self):
            self.controller.refresh()
            self.last_move_time = time()

    class StudyOne(StudyBase):
        def test(self):
            try:
                self.controller.find_element_by_class_name("cl-catalog-item-sub")
                return True
            except NoSuchElementException:
                return False

        def study(self):
            while self.study_chapter_3_start():
                self.study_chapter_3_block_enter()
                self.study_chapter_3_next()
                self.study_chapter_3_block_exit()

        def study_chapter_3_start(self):
            block_list = self.controller.find_elements_by_xpath("//div[@class='cl-catalog-item-sub']/a")
            if block_list:
                for x in block_list:
                    if "done" not in x.get_attribute("class"):
                        x.click()
                        return True
            return False

        def study_chapter_3_next(self):
            while True:
                try:
                    self.controller.find_element_by_xpath("//div[@class='prism-progress-loaded']")
                    break
                except NoSuchElementException:
                    print("wait for loading...")
                    sleep(10)
            while self.controller.find_element_by_xpath("//div[@class='prism-progress-loaded']").get_attribute(
                    "style") != "width: 100%;":
                if time() - self.last_move_time > self.time_diff:
                    self.stop_quit()
                sleep(5)

        def study_chapter_3_block_enter(self):
            self.controller.switch_to.frame("iframe_aliplayer")

        def study_chapter_3_block_exit(self):
            self.controller.switch_to.default_content()

    class StudyTwo(StudyBase):
        def test(self):
            try:
                self.study_chapter_1_block_enter()
                self.controller.find_element_by_class_name("section-item")
                self.study_chapter_1_block_exit()
                return True
            except (NoSuchElementException, NoSuchFrameException):
                return False

        def study(self):
            self.study_chapter_1_block_enter()
            while self.study_chapter_1_block_start():
                while self.study_chapter_1_block_next():
                    sleep(1)
                    if time() - self.last_move_time > self.time_diff:
                        self.stop_quit()
                sleep(5)
            self.study_chapter_1_block_exit()
            while self.study_chapter_1_block_finish():
                sleep(5)

        def study_chapter_1_block_enter(self):
            self.controller.switch_to.frame("aliPlayerFrame")

        def study_chapter_1_block_exit(self):
            self.controller.switch_to.default_content()

        def study_chapter_1_block_start(self):
            try:
                self.controller.find_elements_by_xpath("//li[@class='section-item']")[0].click()
                return True
            except (IndexError, NoSuchElementException):
                return False

        def study_chapter_1_block_next(self):
            try:
                current = self.controller.find_element_by_xpath("//span[@class='current-time']").text
                duration = self.controller.find_element_by_xpath("//span[@class='duration']").text
                if duration == "" or self.cal_time.second(duration) == 0:
                    return True
                else:
                    return self.cal_time.minus(duration, current) > 5
            except NoSuchElementException:
                return False
            except StaleElementReferenceException:
                return True

        def study_chapter_1_block_finish(self):
            try:
                self.controller.find_element_by_xpath(
                    "/html/body/div[@id='newViewerPlaceHolder']/nav[@id='courseInfoSteps']/ul/li[@id='goNextStep']/a")
                return False
            except NoSuchElementException:
                return True

    class StudyThree(StudyBase):
        def test(self):
            try:
                self.controller.find_element_by_id("minStudyTime")
                return True
            except NoSuchElementException:
                return False

        def study(self):
            while True:
                sleep(10)
                if time() - self.last_move_time > self.time_diff:
                    self.stop_quit()
                if self.controller.find_element_by_id("studiedTime").text == self.controller.find_element_by_id(
                        "minStudyTime").text:
                    break
            sleep(5)

    class StudyFour(StudyBase):
        def test(self):
            try:
                self.controller.find_element_by_id("rms-studyRate")
                return True
            except NoSuchElementException:
                return False

        def study(self):
            while True:
                sleep(10)
                if time() - self.last_move_time > self.time_diff:
                    self.stop_quit()
                if self.controller.find_element_by_id("rms-studyRate").text in ['100', '100.0', '100.00']:
                    break
            sleep(5)


class StudyList:
    def __init__(self, my_driver):
        self.driver = my_driver
        self.left = []

    def test(self):
        try:
            for each in self.driver.find_elements_by_xpath("//li[@class='nc-course-card  nc-mycourse-card  ']"):
                if '课程学习' in each.text:
                    self.left.append(each)
            if len(self.left) < 1:
                return False
            return True
        except NoSuchElementException:
            return False

    def enter(self):
        if len(self.left) > 0:
            self.left.pop().click()
            return True
        else:
            return False


def download_captcha():
    download_driver = Driver().get_driver()
    download_driver.get("https://eln.fjnx.com.cn/")
    yzm = download_driver.find_element_by_class_name("yzmImg")
    download_driver.maximize_window()
    download_driver.save_screenshot("pic/test.png")
    from PIL import Image
    img = Image.open("pic/test.png")
    img = img.crop((
        yzm.location['x'],
        yzm.location['y'],
        yzm.location['x'] + yzm.size['width'],
        yzm.location['y'] + yzm.size['height']
    ))
    img = img.convert("RGB")
    img.show()


if __name__ == '__main__':
    driver = Driver()

    driver.get_driver().get("https://eln.fjnx.com.cn/")
    driver.get_driver().maximize_window()
    driver.browser_add()
    driver.get_driver().find_element_by_id("loginName").send_keys(loginconf["loginname"])
    driver.get_driver().find_element_by_id("swInput").send_keys(loginconf["loginpwd"])
    driver.captcha()

    while True:
        try:
            driver.get_driver().find_element_by_xpath("//div[@title='课程中心']").click()
            break
        except (NoSuchElementException, ElementClickInterceptedException):
            sleep(5)
    driver.browser_switch_new()
    sleep(5)
    while True:
        try:
            driver.get_driver().find_element_by_id("loadStudyTask").click()
            break
        except NoSuchElementException:
            sleep(5)
    sleep(5)
    driver.get_driver().find_element_by_xpath("//label[@data-type='STUDY']").click()
    driver.get_driver().find_element_by_xpath("//label[@data-type='NOT_STARTED']").click()
    sleep(5)

    studyList = StudyList(driver.get_driver())
    while studyList.test():
        if studyList.enter():
            driver.browser_switch_new()
            sleep(5)
            Study(driver.get_driver())
            driver.browser_pop()
    driver.logout()
