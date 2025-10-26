"""Các ví dụ lập trình từ tài liệu 'tài liệu bài 1-2-3 về toán học cơ bản'.

Mỗi hàm tương ứng với một ví dụ trong tài liệu, trả về kết quả tính toán
hoặc đối tượng đồ thị để dễ dàng tái sử dụng khi ôn tập.
"""
'''
Chủ đề 0. Hello World
Chủ đề 1. Tính toán trên trường số thực
Chủ đề 2. Vectơ, ma trận và định thức
Chủ đề 3. Hệ phương trình tuyến tính
Chủ đề 4. Dạng toàn phương
Chủ đề 5. Đồ thị của hàm số
Chủ đề 6. Giới hạn
Chủ đề 7. Đạo hàm
Chủ đề 8. Giá trị lớn nhất – nhỏ nhất
Chủ đề 9. Tích phân hàm một biến
Chủ đề 10. Phương trình vi phân
Chủ đề 11. Phương trình sai phân
Phụ lục: Download, cài đặt, và chạy “Hello World”
'''
from __future__ import annotations

import math
from typing import Dict, Tuple

import numpy as np
import sympy
from matplotlib import pyplot as plt
from scipy import integrate
from scipy import linalg as la
from scipy.integrate import odeint
from scipy.optimize import fmin


# Chủ đề 0 --------------------------------------------------------------------
def example_hello_world() -> Tuple[str, float]:
    """In thông điệp Hello World và sin(0)."""
    message = "Hello World !"
    sin_zero = math.sin(0.0)
    print(message)
    print(sin_zero)
    return message, sin_zero


# Chủ đề 1 --------------------------------------------------------------------
def example_real_number_operations() -> Dict[str, Dict[str, float]]:
    """So sánh các phép tính cơ bản bằng math, sympy, numpy."""
    x_sym = sympy.Symbol("x")

    sqrt_sym = sympy.lambdify(x_sym, sympy.sqrt(x_sym))
    log_sym = sympy.lambdify(x_sym, sympy.log(x_sym, 3))
    arcsin_sym = sympy.lambdify(x_sym, sympy.asin(x_sym) ** 2)

    results = {
        "sqrt(5.1)": {
            "math": math.sqrt(5.1),
            "sympy": float(sqrt_sym(5.1)),
            "numpy": float(np.sqrt(5.1)),
        },
        "log_base_3(4)": {
            "math": math.log(4, 3),
            "sympy": float(log_sym(4)),
            "numpy": float(np.log(4) / np.log(3)),
        },
        "arcsin(1/2)^2": {
            "math": math.asin(0.5) ** 2,
            "sympy": float(arcsin_sym(0.5)),
            "numpy": float(np.arcsin(0.5) ** 2),
        },
    }
    return results


# Chủ đề 2 --------------------------------------------------------------------
def example_vectors_and_matrices() -> Dict[str, np.ndarray]:
    """Thực hiện các phép toán vectơ, ma trận cơ bản."""
    row_vector = np.array([1, 2, 3])
    col_vector = np.array([[1], [2], [3]])
    matrix_b = np.array([[11, 4, 20], [4, 9, 8], [3, 6, 9]])
    matrix_c = np.array([[0, 4, 17], [-2, 5, 8], [3.5, 8, -9.2]])

    results = {
        "row_vector": row_vector,
        "col_vector": col_vector,
        "matrix_b": matrix_b,
        "matrix_c": matrix_c,
        "b_plus_c": matrix_b + matrix_c,
        "b_times_c_elementwise": matrix_b * matrix_c,
        "matrix_product": matrix_b.dot(matrix_c),
        "matrix_power_b3": np.linalg.matrix_power(matrix_b, 3),
        "transpose_b": matrix_b.T,
        "rank_b": np.linalg.matrix_rank(matrix_b),
        "inverse_b": np.linalg.inv(matrix_b),
        "determinant_b": np.linalg.det(matrix_b),
    }
    return results


# Chủ đề 3 --------------------------------------------------------------------
def example_linear_system_unique() -> np.ndarray:
    """Giải hệ tuyến tính có nghiệm duy nhất."""
    a = np.array([[1, 1, 1], [1, -1, 0], [1, 1, 2]], dtype=float)
    b = np.array([6, -1, 9], dtype=float)
    solution = np.linalg.solve(a, b)
    return solution


def example_linear_system_singular() -> Tuple[np.ndarray, np.ndarray]:
    """Ví dụ hệ tuyến tính suy biến gây lỗi Singular matrix."""
    a = np.array([[1, 1], [1, 1]], dtype=float)
    b = np.array([6, 9], dtype=float)
    return a, b


# Chủ đề 4 --------------------------------------------------------------------
def example_quadratic_form_eigendecomposition() -> Dict[str, np.ndarray]:
    """Tính giá trị riêng, vectơ riêng của dạng toàn phương."""
    matrix = np.array([[4, -5, 2], [-5, 4, 2], [2, 2, -8]], dtype=float)
    eigenvalues, eigenvectors = la.eig(matrix)
    return {"eigenvalues": eigenvalues, "eigenvectors": eigenvectors}


def example_quadratic_form_definiteness() -> np.ndarray:
    """Kiểm tra dấu xác định của dạng toàn phương bằng phổ eigen."""
    matrix = np.array([[-2, 1, 1], [1, -4, -4], [1, -4, -9]], dtype=float)
    eigenvalues = la.eigvals(matrix)
    return eigenvalues


# Chủ đề 5 --------------------------------------------------------------------
def example_plot_function(show: bool = False) -> Tuple[np.ndarray, np.ndarray]:
    """Vẽ đồ thị 2D y = x^2 + 5 sin x."""
    xs = np.arange(-10.0, 10.0, 0.05)
    ys = xs ** 2 + 5 * np.sin(xs)
    plt.figure()
    plt.plot(xs, ys)
    plt.xlabel("Truc x")
    plt.ylabel("Truc y")
    plt.title("Toan cao cap - 2D")
    if show:
        plt.show()
    plt.close()
    return xs, ys


def example_plot_surface(show: bool = False) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Vẽ đồ thị 3D z = sin(x) * y."""
    xs = np.arange(0.0, 2 * np.pi, 0.1)
    ys = np.arange(0.0, 5.0, 0.1)
    x_grid, y_grid = np.meshgrid(xs, ys)
    z_grid = np.sin(x_grid) * y_grid
    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")
    ax.plot_surface(x_grid, y_grid, z_grid)
    ax.set_xlabel("Truc x")
    ax.set_ylabel("Truc y")
    ax.set_title("Toan cao cap - 3D")
    if show:
        plt.show()
    plt.close(fig)
    return x_grid, y_grid, z_grid


# Chủ đề 6 --------------------------------------------------------------------
def example_limits() -> Dict[str, sympy.Expr]:
    """Tính các giới hạn điển hình bằng sympy."""
    x = sympy.symbols("x", real=True)
    results = {
        "sin(x)/x as x->0": sympy.limit(sympy.sin(x) / x, x, 0),
        "x^2*(1-cos(1/x)) as x->oo": sympy.limit(x ** 2 * (1 - sympy.cos(1 / x)), x, sympy.oo),
        "sin(x) as x->+oo": sympy.limit(sympy.sin(x), x, sympy.oo),
        "x^3 as x->-oo": sympy.limit(x ** 3, x, -sympy.oo),
    }
    return results


# Chủ đề 7 --------------------------------------------------------------------
def example_derivatives_single_variable() -> Dict[str, sympy.Expr]:
    """Đạo hàm cấp một, cấp hai và giá trị tại x = 2."""
    x = sympy.Symbol("x", real=True)
    y = x ** 3 + sympy.sin(x)
    first = sympy.diff(y, x)
    second = sympy.diff(first, x)
    value_at_2 = sympy.lambdify(x, first)(2)
    return {"first_derivative": first, "second_derivative": second, "first_at_2": value_at_2}


def example_derivatives_multivariable() -> Dict[str, object]:
    """Đạo hàm riêng cấp một, cấp hai tại điểm (1,2)."""
    x, y = sympy.symbols("x y", real=True)
    z = x ** 3 + sympy.sin(x * y)

    zx = sympy.diff(z, x)
    zy = sympy.diff(z, y)
    zxx = sympy.diff(zx, x)
    zxy = sympy.diff(zx, y)
    zyy = sympy.diff(zy, y)

    zx_val = sympy.lambdify((x, y), zx)(1, 2)
    zy_val = sympy.lambdify((x, y), zy)(1, 2)
    zxx_val = sympy.lambdify((x, y), zxx)(1, 2)
    zxy_val = sympy.lambdify((x, y), zxy)(1, 2)
    zyy_val = sympy.lambdify((x, y), zyy)(1, 2)

    return {
        "partials": {"zx": zx, "zy": zy},
        "second_partials": {"zxx": zxx, "zxy": zxy, "zyy": zyy},
        "partials_at_1_2": {"zx": zx_val, "zy": zy_val},
        "second_partials_at_1_2": {"zxx": zxx_val, "zxy": zxy_val, "zyy": zyy_val},
    }


# Chủ đề 8 --------------------------------------------------------------------
def example_extrema() -> Dict[str, object]:
    """Tìm cực trị của hàm f(x) = sin^2(x) + x^2 + 1."""
    def f(x_val: float) -> float:
        return math.sin(x_val) ** 2 + x_val ** 2 + 1

    # fmin trả về nghiệm, cần unpack tuple.
    min_location = fmin(f, 1.0, disp=False)
    min_value = f(*min_location)

    # Hàm không có giá trị lớn nhất trên R; minh họa cách tiếp cận và kết quả phân tích.
    return {
        "minimum_point": float(min_location[0]),
        "minimum_value": float(min_value),
        "maximum_exists": False,
        "reason": "Hàm tăng vô hạn khi |x| -> +inf.",
    }


# Chủ đề 9 --------------------------------------------------------------------
def example_integrals() -> Dict[str, Tuple[float, float]]:
    """Tính tích phân xác định và suy rộng bằng scipy.integrate.quad."""
    integral_1 = integrate.quad(lambda x: x * math.exp(x), 1, 10)
    integral_2 = integrate.quad(lambda x: 1 / x ** 2, 1, math.inf)
    return {
        "integral_x_exp_x_1_10": integral_1,
        "integral_1_over_x2_1_inf": integral_2,
    }


# Chủ đề 10 -------------------------------------------------------------------
def example_ode_first_order(show: bool = False) -> Tuple[np.ndarray, np.ndarray]:
    """Giải PTVP dy/dx = x^2, y(0)=5 trên [0,100]."""
    def model(y_val: float, x_val: float) -> float:
        return x_val ** 2

    xspan = np.linspace(0, 100, num=200)
    y0 = 5.0
    solution = odeint(model, y0, xspan).ravel()
    plt.figure()
    plt.plot(xspan, solution)
    plt.xlabel("Truc x")
    plt.ylabel("Truc y")
    plt.title("dy/dx = x^2, y(0)=5")
    if show:
        plt.show()
    plt.close()
    return xspan, solution


def example_ode_second_order(show: bool = False) -> Tuple[np.ndarray, np.ndarray]:
    """Giải hệ tương đương của y'' - (1-y^2)y' + y = 0, y(5)=2, y'(5)=0."""
    def model(u_vec: np.ndarray, x_val: float) -> np.ndarray:
        # u_vec[0] = y, u_vec[1] = y'
        return np.array([u_vec[1], (1 - u_vec[0] ** 2) * u_vec[1] - u_vec[0]])

    xspan = np.linspace(5, 100, num=200)
    u0 = np.array([2.0, 0.0])
    solution = odeint(model, u0, xspan)
    y_values = solution[:, 0]
    plt.figure()
    plt.plot(xspan, y_values)
    plt.xlabel("Truc x")
    plt.ylabel("Truc y")
    plt.title("y'' - (1-y^2)y' + y = 0")
    if show:
        plt.show()
    plt.close()
    return xspan, y_values


def example_ode_second_order_forced(show: bool = False) -> Tuple[np.ndarray, np.ndarray]:
    """Giải y'' - (1-y^2)y' + y = e^x sin x, y(4)=2, y'(4)=0 trên [4,10]."""
    def model(u_vec: np.ndarray, x_val: float) -> np.ndarray:
        forcing = math.exp(x_val) * math.sin(x_val)
        return np.array([u_vec[1], (1 - u_vec[0] ** 2) * u_vec[1] - u_vec[0] + forcing])

    xspan = np.linspace(4, 10, num=200)
    u0 = np.array([2.0, 0.0])
    solution = odeint(model, u0, xspan)
    y_values = solution[:, 0]
    plt.figure()
    plt.plot(xspan, y_values)
    plt.xlabel("Truc x")
    plt.ylabel("Truc y")
    plt.title("y'' - (1-y^2)y' + y = e^x sin x")
    if show:
        plt.show()
    plt.close()
    return xspan, y_values


# Chủ đề 11 -------------------------------------------------------------------
def example_difference_equation_first_order(coeff: float = -2.0, n_steps: int = 10) -> np.ndarray:
    """Giải x(n+1) + 2 x(n) = 0 với x(0)=3."""
    x = np.zeros(n_steps, dtype=float)
    x[0] = 3.0
    for n in range(1, n_steps):
        x[n] = coeff * x[n - 1]
    return x


def example_difference_equation_factorial(n_steps: int = 10) -> np.ndarray:
    """Giải x(n+1) = (n+1)x(n) + (n+1)! * n với x(0)=3."""
    x = np.zeros(n_steps, dtype=float)
    x[0] = 3.0
    for n in range(1, n_steps):
        x[n] = n * x[n - 1] + math.factorial(n) * (n - 1)
    return x


def example_difference_equation_second_order_homogeneous(n_steps: int = 10) -> np.ndarray:
    """Giải x(n+2) - 5 x(n+1) + 6 x(n) = 0 với x(0)=2, x(1)=5."""
    x = np.zeros(n_steps, dtype=float)
    x[0] = 2.0
    x[1] = 5.0
    for n in range(2, n_steps):
        x[n] = 5 * x[n - 1] - 6 * x[n - 2]
    return x


def example_difference_equation_second_order_forced(n_steps: int = 10) -> np.ndarray:
    """Giải x(n+2) - 5 x(n+1) + 6 x(n) = n^2 + 2n + 3 với x(0)=2, x(1)=5."""
    x = np.zeros(n_steps, dtype=float)
    x[0] = 2.0
    x[1] = 5.0
    for n in range(0, n_steps - 2):
        x[n + 2] = 5 * x[n + 1] - 6 * x[n] + n ** 2 + 2 * n + 3
    return x


if __name__ == "__main__":
    print("== Chủ đề 0 ==")
    example_hello_world()

    print("\n== Chủ đề 1 ==")
    for name, libs in example_real_number_operations().items():
        print(name, libs)

    print("\n== Chủ đề 2 ==")
    matrix_results = example_vectors_and_matrices()
    print("determinant_b =", matrix_results["determinant_b"])

    print("\n== Chủ đề 3 ==")
    print("Unique solution:", example_linear_system_unique())

    print("\n== Chủ đề 4 ==")
    eig_results = example_quadratic_form_eigendecomposition()
    print("Eigenvalues:", eig_results["eigenvalues"])

    print("\n== Chủ đề 6 ==")
    for desc, value in example_limits().items():
        print(desc, "->", value)

    print("\n== Chủ đề 7 ==")
    print(example_derivatives_single_variable())

    print("\n== Chủ đề 8 ==")
    print(example_extrema())

    print("\n== Chủ đề 9 ==")
    print(example_integrals())

    print("\n== Chủ đề 11 ==")
    print("First-order difference:", example_difference_equation_first_order())
