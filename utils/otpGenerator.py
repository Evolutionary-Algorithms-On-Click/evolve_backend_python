import math, random


def generateOTP():

    digits = "0123456789"
    OTP = ""

    for _ in range(6):
        OTP += digits[math.floor(random.random() * 10)]

    return OTP
