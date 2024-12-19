import validators


def validateUrl(url):
    return validators.url(url)


def validateEmail(email):
    return validators.email(email)


def validatePassword(password):
    return len(password) >= 8
