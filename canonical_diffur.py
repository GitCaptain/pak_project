from random import randint as rand
from

def generate_coefficients():
    A = [[rand(0, 10) for _ in range(2)] for i in range(2)]  # коэффициенты для производных второго порядка 0..10
    A[0][1] = A[1][0]  # матрица А - симметричная
    B = [rand(0, 1) for _ in range(2)]  # коэффициенты для первых производных 0 или 1
    F = rand(0, 1)  # есть ли правая часть
    return A, B, F


def type_identify(coefficients):
    """
    :param coefficients: кортеж коеффициентов
    :return: тип уравнения
    0 - эллиптическое
    1 - параболическое
    2 - гиперболическое
    """
    A = coefficients[0]
    a, b, c = *A[0], A[1][1]
    type = a*c - b*b
    if type < 0:
        return 2
    if not type:
        return 1
    return 0


def make_canonical(coefficients):
    # step_by_step_solution = []
    A = coefficients[0]
    a, b, c = A[0][0], 2 * A[0][1], A[1][1]
    discrimenant = b*b - 4 * a * c


def repetitor():
    pass

if __name__ == '__main__':
    pass