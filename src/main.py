import random
import time
import os
import matplotlib.pyplot as plt
import numpy as np
from generators import generate_polygon, visualize_polygon
from algorithms import gauss_area, monte_carlo_area, calculate_relative_error


def ensure_images_dir():
    """Створює папку images, якщо вона не існує"""
    images_dir = os.path.join(os.path.dirname(
        os.path.dirname(__file__)), 'images')
    if not os.path.exists(images_dir):
        os.makedirs(images_dir)
        print(f"Створено папку: {images_dir}")
    return images_dir


def test_polygon_generation():
    """
    Завдання 1.3: Генерація та візуалізація тестових полігонів
    """
    print("=" * 60)
    print("ЗАВДАННЯ 1: Генерація та візуалізація полігонів")
    print("=" * 60)

    # Переконуємось що папка images існує
    ensure_images_dir()

    random.seed(42)  # Для відтворюваності результатів

    test_cases = [10, 50, 100]

    for num_vertices in test_cases:
        poly = generate_polygon(num_points=num_vertices, radius=50.0)
        filename = f"./images/polygon_N{num_vertices}.png"
        visualize_polygon(poly, filename=filename)

        print(f"\nПолігон з N={num_vertices} вершин:")
        print(f"  Площа (Shapely): {poly.area:.4f}")
        print(f"  Зображення збережено: {filename}")


def test_area_calculations():
    """
    Завдання 2: Тестування реалізованих алгоритмів
    """
    print("\n" + "=" * 60)
    print("ЗАВДАННЯ 2: Порівняння методів обчислення площі")
    print("=" * 60)

    random.seed(42)
    poly = generate_polygon(num_points=50, radius=50.0)

    # Еталонна площа (Shapely)
    area_shapely = poly.area

    # Метод Гауса
    area_gauss = gauss_area(poly)

    # Метод Монте-Карло
    area_mc = monte_carlo_area(poly, num_points=100000)

    print(f"\nПолігон з N=50 вершин:")
    print(f"  Площа (Shapely):      {area_shapely:.4f}")
    print(f"  Площа (Гаус):         {area_gauss:.4f}")
    print(f"  Площа (Монте-Карло):  {area_mc:.4f}")
    print(
        f"\nПохибка методу Гауса: {calculate_relative_error(area_gauss, area_shapely):.6f}%")
    print(
        f"Похибка Монте-Карло:  {calculate_relative_error(area_mc, area_shapely):.4f}%")


def test_monte_carlo_convergence():
    """
    Завдання 3: Дослідження точності Монте-Карло
    """
    print("\n" + "=" * 60)
    print("ЗАВДАННЯ 3: Дослідження збіжності методу Монте-Карло")
    print("=" * 60)

    # Переконуємось що папка images існує
    ensure_images_dir()

    random.seed(42)
    poly = generate_polygon(num_points=50, radius=50.0)
    area_reference = poly.area

    # Різна кількість точок
    num_points_list = [100, 1000, 10000, 100000]
    errors = []

    print(f"\nЕталонна площа (Shapely): {area_reference:.4f}\n")
    print(f"{'M (точок)':<15} {'Площа (MC)':<15} {'Похибка (%)':<15}")
    print("-" * 45)

    for M in num_points_list:
        area_mc = monte_carlo_area(poly, num_points=M)
        error = calculate_relative_error(area_mc, area_reference)
        errors.append(error)

        print(f"{M:<15} {area_mc:<15.4f} {error:<15.4f}")

    # Побудова графіка
    plt.figure(figsize=(10, 6))
    plt.plot(num_points_list, errors, marker='o', linewidth=2, markersize=8)
    plt.xscale('log')
    plt.xlabel('Кількість точок M', fontsize=12)
    plt.ylabel('Відносна похибка (%)', fontsize=12)
    plt.title(
        'Залежність похибки методу Монте-Карло від кількості точок', fontsize=14)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('./images/error_plot.png', dpi=150)
    print("\nГрафік збіжності збережено: ./images/error_plot.png")
    plt.close()


def benchmark_performance():
    """
    Завдання 4: Аналіз продуктивності (Benchmark)
    """
    print("\n" + "=" * 60)
    print("ЗАВДАННЯ 4: Benchmark продуктивності методів")
    print("=" * 60)

    # Переконуємось що папка images існує
    ensure_images_dir()

    random.seed(42)
    num_vertices_list = [10, 50, 100, 1000]

    results = {
        'N': [],
        'Shapely (s)': [],
        'Gauss (s)': [],
        'Monte-Carlo (s)': []
    }

    print(
        f"\n{'N':<10} {'Shapely (s)':<15} {'Gauss (s)':<15} {'Monte-Carlo (s)':<20}")
    print("-" * 60)

    for N in num_vertices_list:
        poly = generate_polygon(num_points=N, radius=50.0)

        # Benchmark Shapely
        start = time.perf_counter()
        for _ in range(100):  # Повторюємо для більш точного вимірювання
            _ = poly.area
        time_shapely = (time.perf_counter() - start) / 100

        # Benchmark Gauss
        start = time.perf_counter()
        for _ in range(100):
            _ = gauss_area(poly)
        time_gauss = (time.perf_counter() - start) / 100

        # Benchmark Monte-Carlo (M=100000)
        start = time.perf_counter()
        _ = monte_carlo_area(poly, num_points=100000)
        time_mc = time.perf_counter() - start  # Одна ітерація, бо довго

        results['N'].append(N)
        results['Shapely (s)'].append(time_shapely)
        results['Gauss (s)'].append(time_gauss)
        results['Monte-Carlo (s)'].append(time_mc)

        print(f"{N:<10} {time_shapely:<15.6f} {time_gauss:<15.6f} {time_mc:<20.6f}")

    # Побудова графіка
    plt.figure(figsize=(12, 6))

    x_pos = np.arange(len(num_vertices_list))
    width = 0.25

    plt.bar(x_pos - width, results['Shapely (s)'],
            width, label='Shapely', alpha=0.8)
    plt.bar(x_pos, results['Gauss (s)'], width, label='Gauss', alpha=0.8)
    plt.bar(x_pos + width, results['Monte-Carlo (s)'],
            width, label='Monte-Carlo (M=10⁵)', alpha=0.8)

    plt.xlabel('Кількість вершин N', fontsize=12)
    plt.ylabel('Час виконання (секунди)', fontsize=12)
    plt.title('Порівняння продуктивності методів обчислення площі', fontsize=14)
    plt.xticks(x_pos, num_vertices_list)
    plt.legend()
    plt.yscale('log')
    plt.grid(True, alpha=0.3, axis='y')
    plt.tight_layout()
    plt.savefig('./images/time_benchmark.png', dpi=150)
    print("\nГрафік benchmark збережено: ./images/time_benchmark.png")
    plt.close()

    return results


def main():
    """
    Основна функція для запуску всіх експериментів
    """
    print("\n" + "=" * 60)
    print("ЛАБОРАТОРНА РОБОТА: Алгоритмічні та евристичні методи")
    print("обчислення площі геометричних фігур")
    print("=" * 60)

    # Завдання 1: Генерація полігонів
    test_polygon_generation()

    # Завдання 2: Тестування алгоритмів
    test_area_calculations()

    # Завдання 3: Дослідження збіжності Монте-Карло
    test_monte_carlo_convergence()

    # Завдання 4: Benchmark
    results = benchmark_performance()

    print("\n" + "=" * 60)
    print("ВИКОНАННЯ ЗАВЕРШЕНО!")
    print("=" * 60)
    print("\nРезультати збережено у папці images/")
    print("Перегляньте README.md для детального звіту.")


if __name__ == "__main__":
    main()
