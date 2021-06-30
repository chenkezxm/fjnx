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
