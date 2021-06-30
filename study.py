from selenium.common.exceptions import NoSuchElementException, NoSuchFrameException, StaleElementReferenceException, \
    ElementNotInteractableException
from time import sleep
from driver import Driver
from strtime import StrTime


class Study(Driver):
    def __init__(self, driver=None):
        super().__init__(driver)
        self.cal_time = StrTime()

    def prepare(self):
        while True:
            try:
                self.driver.find_element_by_xpath("//div[@title='课程中心']").click()
                break
            except NoSuchElementException:
                sleep(5)
        self.browser_switch_new()
        sleep(5)
        while True:
            try:
                self.driver.find_element_by_id("loadStudyTask").click()
                break
            except NoSuchElementException:
                sleep(5)
        sleep(5)
        self.driver.find_element_by_xpath("//label[@data-type='STUDY']").click()
        self.driver.find_element_by_xpath("//label[@data-type='NOT_STARTED']").click()
        sleep(5)

    def study_loop(self):
        while True:
            if not self.study_exists():
                break
            else:
                self.study_prepare()
                self.study_chapter()
                self.study_exit()
                self.study_refresh()

    def study_exists(self):
        try:
            self.driver.find_element_by_xpath("//li[@class='nc-course-card  nc-mycourse-card  ']")
            return True
        except NoSuchElementException:
            return False

    def study_prepare(self):
        self.driver.find_element_by_xpath("//li[@class='nc-course-card  nc-mycourse-card  ']").click()
        self.browser_switch_new()
        sleep(5)

    def study_chapter(self):
        chapter_dict = {0: self.study_chapter_1, 1: self.study_chapter_2, 2: self.study_chapter_3}
        if self.study_chapter_type() >= 0:
            chapter_dict.get(self.study_chapter_type())()
        else:
            self.study_chapter_unknown()
        sleep(5)

    def study_chapter_type(self):
        if self.study_chapter_type1_test():
            return 0
        elif self.study_chapter_type2_test():
            return 1
        elif self.study_chapter_type3_test():
            return 2
        else:
            return -1

    def study_chapter_type1_test(self):
        try:
            self.study_chapter_1_block_enter()
            self.driver.find_element_by_class_name("section-item")
            self.study_chapter_1_block_exit()
            return True
        except (NoSuchElementException, NoSuchFrameException):
            return False

    def study_chapter_type2_test(self):
        try:
            self.driver.find_element_by_id("minStudyTime")
            return True
        except NoSuchElementException:
            return False

    def study_chapter_type3_test(self):
        try:
            self.driver.find_element_by_class_name("cl-catalog-item-sub")
            return True
        except NoSuchElementException:
            return False

    def study_chapter_1(self):
        self.study_chapter_1_block_enter()
        while self.study_chapter_1_block_start():
            while self.study_chapter_1_block_next():
                sleep(1)
            sleep(5)
        self.study_chapter_1_block_exit()
        print(2)
        while self.study_chapter_1_block_finish():
            sleep(5)
        print(3)

    def study_chapter_1_block_enter(self):
        self.driver.switch_to.frame("aliPlayerFrame")

    def study_chapter_1_block_exit(self):
        self.driver.switch_to.default_content()

    def study_chapter_1_block_start(self):
        try:
            self.driver.find_elements_by_xpath("//li[@class='section-item']")[0].click()
            return True
        except (IndexError, NoSuchElementException):
            return False

    def study_chapter_1_block_next(self):
        try:
            current = self.driver.find_element_by_xpath("//span[@class='current-time']").text
            duration = self.driver.find_element_by_xpath("//span[@class='duration']").text
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
            self.driver.find_element_by_xpath(
                "/html/body/div[@id='newViewerPlaceHolder']/nav[@id='courseInfoSteps']/ul/li[@id='goNextStep']/a")
            return False
        except NoSuchElementException:
            return True

    def study_chapter_2(self):
        while True:
            sleep(10)
            if self.driver.find_element_by_id("studiedTime").text == self.driver.find_element_by_id(
                    "minStudyTime").text:
                break
        sleep(5)

    def study_chapter_3(self):
        while self.study_chapter_3_start():
            self.study_chapter_3_block_enter()
            self.study_chapter_3_next()
            self.study_chapter_3_block_exit()

    def study_chapter_3_start(self):
        block_list = self.driver.find_elements_by_xpath("//div[@class='cl-catalog-item-sub']/a")
        if block_list:
            for x in block_list:
                if "done" not in x.get_attribute("class"):
                    x.click()
                    return True
        return False

    def study_chapter_3_next(self):
        while True:
            try:
                self.driver.find_element_by_xpath("//div[@class='prism-progress-loaded']")
                break
            except NoSuchElementException:
                print("wait for loading...")
                sleep(10)
        while self.driver.find_element_by_xpath("//div[@class='prism-progress-loaded']").get_attribute(
                "style") != "width: 100%;":
            sleep(5)

    def study_chapter_3_block_enter(self):
        self.driver.switch_to.frame("iframe_aliplayer")

    def study_chapter_3_block_exit(self):
        self.driver.switch_to.default_content()

    def study_chapter_unknown(self):
        while True:
            try:
                self.driver.find_element_by_xpath(
                    "/html/body/div[@id='newViewerPlaceHolder']/nav[@id='courseInfoSteps']/ul/li[@id='goNextStep']/a")
                break
            except (NoSuchElementException, ElementNotInteractableException):
                sleep(20)

    def study_exit(self):
        if self.study_exit_1_test():
            self.study_exit_1()
        self.browser_pop()
        sleep(5)

    def study_exit_1_test(self):
        try:
            self.driver.find_element_by_xpath(
                "/html/body/div[@id='newViewerPlaceHolder']/nav[@id='courseInfoSteps']/ul/li[@id='goNextStep']/a")
            return True
        except (NoSuchElementException, ElementNotInteractableException):
            return False

    def study_exit_1(self):
        self.driver.find_element_by_xpath(
            "/html/body/div[@id='newViewerPlaceHolder']/nav[@id='courseInfoSteps']/ul/li[@id='goNextStep']/a").click()
        sleep(5)
        self.driver.find_element_by_xpath("//input[@value='5']").click()
        sleep(2)
        try:
            self.driver.find_element_by_xpath("//button[@id='courseEvaluateSubmit']").click()
            sleep(5)
            self.driver.find_element_by_class_name("layui-layer-btn1").click()
        except NoSuchElementException:
            try:
                self.driver.find_element_by_xpath("/html/body/header/div/button").click()
            except NoSuchElementException:
                self.study_exit_unknown()

    @staticmethod
    def study_exit_unknown():
        while input("人工介入:") != 'ok':
            sleep(5)

    def study_refresh(self):
        self.driver.refresh()
        sleep(5)
        self.driver.find_element_by_xpath("//label[@data-type='STUDY']").click()
        self.driver.find_element_by_xpath("//label[@data-type='NOT_STARTED']").click()
        sleep(5)

    def run(self):
        self.login()
        self.prepare()
        self.study_loop()
        self.logout()


if __name__ == '__main__':
    Study().run()
