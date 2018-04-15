from random import randint, choice


class Equation:

    def __init__(self, coefficients=None, right_part=None, variables=('x','y')):
        if not coefficients:
            self.coefficients = Equation.generate_coefficients()
        else:
            self.coefficients = coefficients
        if not self.right_part:
            self.right_part = Equation.generate_right_part()
        else:
            self.right_part = right_part
        self.variables = variables
        self.equation = (self.coefficients, self.right_part, self.variables)

    def __str__(self):
        pass

    @staticmethod
    def generate_coefficients():
        A = [[randint(1, 10) for _ in range(2)] for i in range(2)]  # коэффициенты для производных второго порядка 1..10
        A[0][1] = A[1][0]  # матрица А - симметричная
        B = [randint(0, 1) for _ in range(2)]  # коэффициенты для первых производных 0 или 1
        F = randint(0, 1)  # есть ли правая часть
        return A, B, F

    @staticmethod
    def generate_right_part():
        funcs = ['cos', 'sin', 'ln']
        x_coefficient = randint(1, 5)
        y_coefficient = randint(0, 5)
        return choice(funcs), x_coefficient, y_coefficient

    def type_identify(self):
        """
        :param coefficients: кортеж коеффициентов
        :return: тип уравнения
        0 - эллиптическое
        1 - параболическое
        2 - гиперболическое
        """
        A = self.coefficients[0]
        a, b, c = *A[0], A[1][1]
        type = a*c - b*b
        if type < 0:
            return 2
        if not type:
            return 1
        return 0

    def make_canonical(self):
        """
        # общий вид уравнения: a*Uxx + b*Uxy + c*Uyy + B[0]*Ux + B[1]*Uy = C[0]*F(x,y)
        # рассмотрим только производные второго порядка a*Uxx + b*Uxy + c*Uyy = 0
        # отсюда получаем уравнение нужное для замены переменных, хз как оно называется м.б. характеристическое
        # a*dy^2 - b*dx*dy + c * dx^2 = 0 (/dx^2) -> a(y'^2) + by' + c = 0
        # y' - корни
        # нужно рассмотреть случаи: корни равны, корни различны, корни мнимые
        # (Это зависит от типа уравнения и нужно для правильной замены)
        #
        # y'i = -ci  ->  y_i + ci*x = const
        # замена h = y_1 + c1*x, p = y_2 + c2*x - т.н. линии уровня (но это не важно)
        # H,P - новые переменные, h[0], p[0], h[1], p[1] - их производные по x и y соответственно
        # нужно переписать производные в новых переменных
        # Ux = Uh*Hx + Up*Px и т.д., тоесть обновить матрицу коээффициентов
        # A = [[C(Uxx), 0.5C(Uxy)], [0.5C(Uxy), C(Uyy)]] -> A' = [[C(Uhh), 0], [0, C(Upp)]]
        #                                                || A' = [[0, 0.5C(Uhp)], [0.5C(Uhp), 0]]
        # B = [C(Ux), C(Uy)] -> B' = [C(Uh), C(Up)]
        # F(x,y) -> F(c11*h + c12*p, c21*h+ c22*p)
        # cij - коэффициенты константы, C(X) - коэффициент при Х тоже константа
        #
        # Ux = Uh*Hx + Up*Px = Uh*h[0] + Up*p[0]
        # Uy = Uh*Hy + Up*Py = Uh*h[1] + Up*p[1]
        #
        # Uxx = (Uh*h[0] + Up*p[0])h * Hx + (Uh*h[0] + Up*p[0])p * Px =
        # = Uhh*h[0]*h[0] + Uph *p[0]*h[0] + Uhp*h[0]*p[0] + Upp*p[0]*p[0] = Uhh*h[0]^2 + 2*Uhp*h[0]*p[0] + Upp*p[0]^2
        #
        # Uxy = (Uh*h[0] + Up*p[0])h * Hx + (Uh*h[0] + Up*p[0])p * Px =
        # = Uhh*h[0]*h[1] + Uph*p[0]*h[1] + Uhp*h[0]*p[1] + Upp*p[0]*p[1] = Uhh*h[0]*h[1] + Uhp*(h[0]*p[1]+p[0]*h[1])
        #                                                                                 + Upp*p[0]*p[1]
        #
        # Uyy = (Uh*h[1] + Up*p[1])h * Hy + (Uh*h[1] + Up*p[1])p * Py =
        # = Uhh*h[1]*h[1] + Uph*p[1]*h[1] + Uhp*h[1]*p[1] + Upp*p[1]*p[1] = Uhh*h[1]^2 + 2*Uhp*p[1]*h[1] + Upp*p[1]^2
        #
        # h[0]*x+h[1]*y = H   |
        #                     | - система уравнений, нужно выразить x и y через H и P, чтоб подставить в правую часть
        # p[0]*x+p[1]*y = P   |
        # determinant = (h[0]*p[1]-h[1]*p[0])
        # по формулам крамера если система совместна, т.е. определитель determinant отличен от нуля
        # (иначе решения нет или их бесконечно много, нам такого не надо, лучше сгенерируем новые коэффицинты)
        # x = (H*p[1]-P*h[1])/determinant
        # y = (P*h[0]-H*p[0])/determinant
        #
        # F(x,y) -> F((H*p[1]-P*h[1])/determinant, (P*h[0]-H*p[0])/determinant)
        # C(H) = (p[1]*C(x)-p[0]*C(y))/determinant, C(P) = (C(y)*h[0]-C(x)*h[1])/determinant
        # C(x), C(y) - коэффициенты перед x и y в F, пример: cos(2*x-3*y) -> C(x) = 2, c(y) = -3
        #
        # a*Uxx + b*Uxy + c*Uyy + B[0]*Ux + B[1]*Uy = C[0]*F(x,y)
        #
        # отсюда:
        #
        # -> C(Uh) = (С(Ux)*h[0] + C(Uy)*h[1]) = B[0]*h[0]+B[1]*h[1]
        # -> C(Up) = (C(Ux)*p[0] + C(Uy)*p[1]) = B[0]*p[0]+B[1]*p[1]
        # -> C(Uhh) = a*h[0]^2+b*h[0]*h[1]+c*h[1]^2
        # -> C(Uhp) = 2*a*p[0]*h[0]+b*(h[0]*p[1]+h[1]*p[0])+2*c*p[1]*h[1]
        # -> C(Upp) = a*p[0]^2+b*p[0]*p[1]+c*p[1]^2
        #
        # Если я нигде не проебался, то либо C(Uhp) = 0, либо C(Upp) = C(Uhh) = 0, т.о. получим канонический вид:
        # C(Upp)*Upp + C(Uhh)*Uhh = F
        # или
        # С(Uhp) * Uhp = F
        """
        step_by_step_solution = []
        coefficients = self.equation[0]
        F = self.equation[1]
        A = coefficients[0]
        B = coefficients[1]
        C = coefficients[2]
        a, b, c = A[0][0], 2 * A[0][1], A[1][1]
        discrimenant = complex(b*b - 4 * a * c)
        y_derivative_1, y_derivative_2 = (-b+discrimenant**0.5)/2/a, (-b-discrimenant**0.5)/2/a  # y' - корни

        if not y_derivative_2.imag:
            if y_derivative_2 == y_derivative_1:
                h = (-y_derivative_1, 0)  # коэффициент при x, y
                p = (-y_derivative_2, 1)
            else:
                h = (-y_derivative_1, 1)
                p = (-y_derivative_2, 1)
        else:
            h = (-y_derivative_1.real, 1)
            p = (-y_derivative_1.imag, 0)

        determinant = h[0]*p[1]-h[1]*p[0]

        if not determinant:
            # нужно перегенирировать уравнение и привести его к каноническому виду
            return Equation().make_canonical

        newA = [[a*h[0]*h[0] + b*h[0]*h[1] + c*h[1]*h[1], a*p[0]*h[0] + b/2*(h[0]*p[1] + h[1]*p[0]) + c*p[1]*h[1]],
                [a*p[0]*h[0] + b/2*(h[0]*p[1] + h[1]*p[0]) + c*p[1]*h[1], a*p[0]*p[0] + b*p[0]*p[1] + c*p[1]*p[1]]]

        newB = [B[0]*h[0]+B[1]*h[1], B[0]*p[0] + B[1]*p[1]]

        new_Right_part = (F[0], (p[1]*F[1]-p[0]*F[2])/determinant, (F[2]*h[0]-F[1]*h[1])/determinant)

        return self, Equation((newA, newB, C), new_Right_part, ('h', 'p')), step_by_step_solution


class Student:

    def __init__(self, name):
        self.skill_level = 0
        self.name = name



def repeater():
    pass


if __name__ == '__main__':
    pass
