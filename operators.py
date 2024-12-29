class Operators:
    def protectedDiv(left, right):
        try:
            return left / right
        except ZeroDivisionError:
            return 1
    