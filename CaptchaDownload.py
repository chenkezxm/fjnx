import asyncio
import uuid

import requests
import cv2
import threading

from requests.exceptions import ProxyError, ConnectTimeout


class Download:
    def __init__(self, proxy):
        self.proxy = proxy
        self.header = {
            'Host': 'eln.fjnx.com.cn',
            'User-Agent': 'Mozilla/5.0 (Windows NT 6.1; Win64; x64; rv:90.0) Gecko/20100101 Firefox/90.0',
            'Accept': 'image/webp,*/*',
            'Accept-Language': 'zh-CN,zh;q=0.8,zh-TW;q=0.7,zh-HK;q=0.5,en-US;q=0.3,en;q=0.2',
            'Accept-Encoding': 'gzip, deflate, br',
            'Connection': 'keep-alive',
            'Referer': 'https://eln.fjnx.com.cn/login/login.init.do?returnUrl=https%3A%2F%2Feln.fjnx.com.cn%2Fos%2Fhtml%2Findex.init.do&elnScreen=1366*768elnScreen',
            'Sec-Fetch-Dest': 'image',
            'Sec-Fetch-Mode': 'no-cors',
            'Sec-Fetch-Site': 'same-origin'
        }

    def run(self):
        respond = requests.get(
            url="https://eln.fjnx.com.cn/login/login.securityCode.do",
            headers=self.header,
            proxies=self.proxy
        )
        file = "train/" + str(uuid.uuid4()) + ".png"
        with open(file, 'wb') as fp:
            fp.write(respond.content)
        img = cv2.imread(file)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        h, w = img.shape[:2]
        for y in range(0, w):
            for x in range(0, h):
                if y < 5 or y > w - 5:
                    img[x, y] = 255
                if x < 5 or x > h - 5:
                    img[x, y] = 255
        for y in range(1, w - 1):
            for x in range(1, h - 1):
                count = 0
                if img[x, y - 1] > 245:
                    count = count + 1
                if img[x, y + 1] > 245:
                    count = count + 1
                if img[x - 1, y] > 245:
                    count = count + 1
                if img[x + 1, y] > 245:
                    count = count + 1
                if count > 2:
                    img[x, y] = 255
        for y in range(0, w):
            for x in range(0, h):
                if img[x, y] > 100:
                    img[x, y] = 255
        cv2.imwrite(file, img)


class Test:
    def __init__(self, proxy):
        self.lock = threading.Lock()
        self.proxy = proxy

    def test(self, proxy):
        try:
            requests.get(
                url="https://www.baidu.com",
                proxies=proxy,
                timeout=5
            )
        except (ProxyError, ConnectTimeout):
            with self.lock:
                self.proxy.remove(proxy)

    def run(self):
        t_list = []
        for x in self.proxy:
            t_ = threading.Thread(target=self.test, args=(x,))
            t_list.append(t_)
            t_.start()
        for x in t_list:
            x.join()
        return self.proxy


class Process:
    def __init__(self, func):
        self.is_run = False
        self.process = func

    async def run_once(self):
        cur = None
        while self.is_run:
            if len(self.process) > 0:
                cur = self.process.pop()
                break
            else:
                await asyncio.sleep(0)
        if cur is not None:
            cur.run()
            self.process.append(cur)

    async def run(self, times):
        task = []
        for x in range(times):
            task.append(asyncio.create_task(self.run_once()))
        await asyncio.gather(*task)

    def start(self, times):
        self.is_run = True
        asyncio.run(self.run(times))


if __name__ == '__main__':
    proxy_list = [
        {'https': '210.26.124.143:808'},
        {'https': '106.75.226.36:808'},
        {'https': '101.37.79.125:3128'},
        {'https': '114.113.126.87:80'}
    ]
    proxy_list = Test(proxy_list).run()
    if len(proxy_list) == 0:
        process = Process([Download(None)])
    else:
        process_list = []
        for each in proxy_list:
            process_list.append(Download(each))
        process = Process(process_list)
    t = threading.Thread(target=process.start, args=(100,))
    t.start()
    t.join()
