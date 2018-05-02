from random import randint, choice


class Equation:

    def __init__(self, skill=1, coefficients=None, right_part=None, variables=('x', 'y')):
        self.for_skill = skill
        if not coefficients:
            self.coefficients = Equation.generate_coefficients(self.for_skill)
        else:
            self.coefficients = coefficients
        if not right_part:
            self.right_part = Equation.generate_right_part(self.for_skill)
        else:
            self.right_part = right_part
        self.variables = variables
        self.equation = (self.coefficients, self.right_part, self.variables)

    def __str__(self):
        A, B, F_c = self.coefficients
        F, v1_c, v2_c = self.right_part
        v1, v2 = self.variables
        result = ""

        def add_sign(x, s):
            sgn = ''
            if x < 0:
                sgn = ' - '
            elif s:
                sgn = ' + '
            return sgn

        def add_val_and_multiply_sign(x):
            if abs(x) == 1:
                return ''
            return str(abs(x)) + ' * '


        # Производные второго порядка
        if A[0][0]:
            result += add_val_and_multiply_sign(A[0][0]) + 'd^2(U)/(d' + v1 + '^2)'
        if A[0][1]:
            result += add_sign(A[0][1], result)
            result += add_val_and_multiply_sign(2*A[0][1]) + 'd^2(U)/(d' + v1 + '*d' + v2 + ')'
        if A[1][1]:
            result += add_sign(A[1][1], result)
            result += add_val_and_multiply_sign(A[1][1]) + 'd^2(U)/(d' + v2 + '^2)'

        # производные первого порядка
        if B[0]:
            result += add_sign(B[0], result)
            result += add_val_and_multiply_sign(B[0]) + 'dU/d' + v1
        if B[1]:
            result += add_sign(B[1], result)
            result += add_val_and_multiply_sign(B[1]) + 'dU/d' + v2

        if not result:
            result = '0'

        # правая часть
        result += ' = '
        if F_c:
            result += F + '('
            if v1_c:
                result += add_sign(v1_c, result)
                result += add_val_and_multiply_sign(v1_c) + v1
            if v2_c:
                result += add_sign(v2_c, result)
                result += add_val_and_multiply_sign(v2_c) + v2
            result += ')'
        else:
            result += '0'
        return result

    @staticmethod
    def generate_coefficients(skill=1):
        A = [[randint(1, skill) for _ in range(2)] for i in range(2)]  # коэффициенты для производных второго порядка 1..skill
        A[0][1] = A[1][0]  # матрица А - симметричная
        B = [randint(0, 1) for _ in range(2)]  # коэффициенты для первых производных 0 или 1
        F = randint(0, 1)  # есть ли правая часть
        return A, B, F

    @staticmethod
    def generate_right_part(skill=1):
        funcs = ['cos', 'sin', 'ln', 'exp', 'tg']
        x_coefficient = randint(1, skill)
        y_coefficient = randint(0, skill)
        return choice(funcs), x_coefficient, y_coefficient

    def type_identify(self):
        """
        :return: тип уравнения
        0 - эллиптическое
        1 - параболическое
        2 - гиперболическое
        """
        A = self.coefficients[0]
        a, b, c = *A[0], A[1][1]
        type_id = a*c - b*b
        if type_id < 0:
            return 2, "гиперболическое"
        if not type_id:
            return 1, "параболическое"
        return 0, "эллиптическое"

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
        # Uxy = (Uh*h[0] + Up*p[0])h * Hy + (Uh*h[0] + Up*p[0])p * Py =
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
        step_by_step_solution.append("Начальный вид уравнения:\n" + str(self))
        coefficients = self.equation[0]
        F = self.equation[1]
        A = coefficients[0]
        B = coefficients[1]
        C = coefficients[2]
        a, b, c = A[0][0], 2 * A[0][1], A[1][1]
        step_by_step_solution.append("От исходного уравнения перейдем к уравнению характеристик.\nДля этого Uxx заменим"
                                     " на dy^2, Uxy на dxdy и поменяем знак у коэффициента перед ним, а Uyy заменим на "
                                     "dx^2, остальные части уравнения отбросим.")
        b = -b  # меняем знак
        step_by_step_solution.append("Имеем:\n")
        step_by_step_solution.append("{0}*dy^2 - {1}*dxdy + {2}*dx^2 = 0".format(A[0][0], 2*A[0][1], A[1][1]))
        step_by_step_solution.append("Разделим данное уравнение на dx^2.")
        step_by_step_solution.append("Получим:\n{0}*y'^2 - {1}*y' + {2} = 0".format(A[0][0], 2*A[0][1], A[1][1]))
        step_by_step_solution.append("Решим данное квадратное уравнение относительно y'")
        discrimenant = b*b - 4 * a * c
        step_by_step_solution.append("Дискрименант = " + str(discrimenant))
        discrimenant = complex(discrimenant)
        y_derivative_1, y_derivative_2 = (-b+discrimenant**0.5)/2/a, (-b-discrimenant**0.5)/2/a  # y' - корни
        if not y_derivative_2.imag:
            y_derivative_2 = y_derivative_2.real
            y_derivative_1 = y_derivative_1.real
        step_by_step_solution.append("y' = " + str(y_derivative_1) + " или y' = " + str(y_derivative_2))
        step_by_step_solution.append("Проинтегрируем полученные уравнения по dx, имеем:")
        step_by_step_solution.append("y + " + str(-y_derivative_1) + "*x = const1, или y + " + str(-y_derivative_2) +
                                     "*x = const2")
        step_by_step_solution.append("Теперь сделаем замену переменных в зависимости от типа уравнения")
        step_by_step_solution.append("Тип нашего уравнения - " + self.type_identify()[1])
        step_by_step_solution.append("Имеем следующую замену:")
        if not y_derivative_2.imag:
            if y_derivative_2 == y_derivative_1:
                step_by_step_solution.append("h = " + str(-y_derivative_1) + "*x, "
                                             "p = " + str(-y_derivative_2) + "*x + y")
                h = (-y_derivative_1.real, 0)  # коэффициент при x, y
                p = (-y_derivative_2.real, 1)
            else:
                step_by_step_solution.append("h = " + str(-y_derivative_1) + "*x + y, "
                                             "p = " + str(-y_derivative_2) + "*x + y")
                h = (-y_derivative_1.real, 1)
                p = (-y_derivative_2.real, 1)
        else:
            step_by_step_solution.append("h = " + str(-y_derivative_1.real) + "*x + y, "
                                         "p = " + str(-y_derivative_1.imag) + "*x")
            h = (-y_derivative_1.real, 1)
            p = (-y_derivative_1.imag, 0)

        determinant = h[0]*p[1]-h[1]*p[0]
        b = -b  # характеристическое уравнение решено, меняем знак b обратно
        if not determinant:
            # нужно перегенирировать уравнение и привести его к каноническому виду
            return Equation(self.for_skill).make_canonical()
        step_by_step_solution.append("Пересчитываем производные в новых переменных: ")

        step_by_step_solution.append("Ux = Uh*Hx + Up*Px = Uh*{0} + Up*{1}".format(h[0], p[0]))
        step_by_step_solution.append("Uy = Uh*Hy + Up*Py = Uh*{0} + Up*{1}".format(h[1], p[1]))
        step_by_step_solution.append(("Uxx = (Uh*{0} + Up*{1})h * Hx + (Uh*{0} + Up*{1})p * Px = "
                                      "Uhh*{2} + Uph*{3} + Uhp*{3} + Upp*{4} = Uhh*{2} + Uhp*{5} + Upp*{4}")
                                     .format(h[0], p[0], h[0]*h[0], h[0]*p[0], p[0]*p[0], 2*h[0]*p[0]))
        step_by_step_solution.append(("Uxy = (Uh*{0} + Up*{1})h * Hy + (Uh*{0} + Up*{1})p * Py = "
                                     "Uhh*{2} + Uph*{3} + Uhp*{4} + Upp*{5} = Uhh*{2} + Uhp*{6} + Upp*{5}")
                                     .format(h[0], p[0], h[0] * h[1], p[0] * h[1], h[0] * p[1], p[0] * p[1],
                                             h[0]*p[1]+p[0]*h[1]))
        step_by_step_solution.append(("Uyy = (Uh*{0} + Up*{1})h * Hy + (Uh*{0} + Up*{1})p * Py = "
                                      "Uhh*{2} + Uph*{3} + Uhp*{3} + Upp*{4} = Uhh*{2} + Uhp{5} + Upp*{4}")
                                     .format(h[1], p[1], h[1] * h[1], h[1] * p[1], p[1] * p[1], 2 * h[1] * p[1]))

        newA = [[a*h[0]*h[0] + b*h[0]*h[1] + c*h[1]*h[1], a*p[0]*h[0] + b/2*(h[0]*p[1] + h[1]*p[0]) + c*p[1]*h[1]],
                [a*p[0]*h[0] + b/2*(h[0]*p[1] + h[1]*p[0]) + c*p[1]*h[1], a*p[0]*p[0] + b*p[0]*p[1] + c*p[1]*p[1]]]

        newB = [B[0]*h[0]+B[1]*h[1], B[0]*p[0] + B[1]*p[1]]

        # для упрощения проверки округлим все коэффициенты до sign_cnt знаков после запятой
        sign_cnt = 2
        newA[0][0] = round(newA[0][0], sign_cnt)
        newA[0][1] = round(newA[0][1], sign_cnt)
        newA[1][0] = round(newA[1][0], sign_cnt)
        newA[1][1] = round(newA[1][1], sign_cnt)
        newB[0] = round(newB[0], sign_cnt)
        newB[1] = round(newB[1], sign_cnt)

        if C:
            step_by_step_solution.append("Выразим x и y через новые переменные h и p для подставновки "
                                         "их в правую часть уравнения")
            step_by_step_solution.append(("коэффициент при h = {0}\n"
                                          "коэффициент при p = {1}").format((p[1]*F[1]-p[0]*F[2])/determinant,
                                                                            (F[2]*h[0]-F[1]*h[1])/determinant))

        new_Right_part = (F[0], (p[1]*F[1]-p[0]*F[2])/determinant, (F[2]*h[0]-F[1]*h[1])/determinant)

        step_by_step_solution.append("Подставим новые производные и правую часть в исходное уравнение и получим "
                                     "канонический вид:")

        newEquation = Equation(self.for_skill, (newA, newB, C), new_Right_part, ('h', 'p'))

        step_by_step_solution.append(str(newEquation))

        return self, newEquation, step_by_step_solution


class Student:

    def __init__(self, name):
        self.skill_level = 1
        self.name = name


def repeater(student):
    print("Здравствуйте,", student.name)
    while True:
        print("Введите 0, если хотите завершить работу")
        print("Введите 1, если хотите определять тип уравнения")
        print("Введите 2, если хотите приводить уравнения к каноническому виду")
        print("Введите 3, если хотите посмотреть ментальную карту")
        activity_type = int(input().strip())
        if not activity_type:
            print("До свидания, сегодня вы хорошо поработали!")
            break

        if activity_type == 1:
            to_solve = Equation(student.skill_level)
            print("Введите тип данного уравнения:")
            print(to_solve)
            eq_type = input().strip()
            if eq_type == to_solve.type_identify()[1]:
                print("Верно, вы молодец!")
                student.skill_level += 1
            else:
                print("Ошибка! Попробуйте еще раз!")
                student.skill_level = max(student.skill_level - 1, 1)

        if activity_type == 2:
            to_solve = Equation(student.skill_level).make_canonical()
            print("Приведите следующее уравнение к каноническому типу:", to_solve[0])
            print("Введите какие коэффициенты (с точностью до 2х знаков) у вас получились при")
            A = [[0, 0], [0, 0]]
            A[0][0] = float(input("d^2(U)/(dh^2): "))
            A[0][1] = A[1][0] = float(input("d^2(U)/(dh*dp): ")) / 2
            A[1][1] = float(input("d^2(U)/(dp^2): "))
            B = [0, 0]
            B[0] = float(input("dU/dh: "))
            B[1] = float(input("dU/dp: "))
            if (A, B) == to_solve[1].coefficients[:-1]:
                print("Верно! Вы молодец!")
                print("Итоговое уравнение:")
                print(to_solve[1])
                student.skill_level += 1
            else:
                print("Ошибка! Попробуйте еще раз!")
                student.skill_level = max(student.skill_level-1, 1)
                need_help = int(input("Введите 1, если хотите посмотреть пошаговое решение: ").strip())
                if need_help == 1:
                    print('\n'.join(to_solve[2]))

        if activity_type == 3:
            pass


if __name__ == '__main__':
    """
    test = Equation(1, ([[1, 1], [1, -3]], [0, 0], 0))  # гиперболический тип
    #print('\n'.join(test.make_canonical()[2]))

    test = Equation(1, ([[3, -2], [-2, 1]], [-3, 1], 0))  # гиперболический тип
    #print('\n'.join(test.make_canonical()[2]))

    test = Equation(1, ([[2, -2.5], [-2.5, 3]], [0, 0], 0))  # гиперболический тип
    #print('\n'.join(test.make_canonical()[2]))

    test = Equation(1, ([[1, -1], [-1, 1]], [0, 0], 1), ('exp', 0, 1))  # параболический тип (вывод правой части?? х,у)
    #print('\n'.join(test.make_canonical()[2]))

    test = Equation()  # рандом
    #print('\n'.join(test.make_canonical()[2]))
    """
    student = Student(input("Введите имя, чтобы начать: "))
    repeater(student)

