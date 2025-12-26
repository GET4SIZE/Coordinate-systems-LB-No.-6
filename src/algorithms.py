import random
from shapely.geometry import Polygon


def gauss_area(polygon: Polygon) -> float:
    """
    Обчислює площу полігону за методом Гауса (Shoelace formula).
    
    Формула: S = 1/2 * |Σ(x_i * y_{i+1} - x_{i+1} * y_i)|
    
    Args:
        polygon (Polygon): Полігон Shapely
    
    Returns:
        float: Площа полігону
    """
    coords = list(polygon.exterior.coords)
    n = len(coords) - 1  # Остання точка дублює першу
    
    area_sum = 0.0
    for i in range(n):
        x_i, y_i = coords[i]
        x_next, y_next = coords[i + 1]
        area_sum += (x_i * y_next - x_next * y_i)
    
    return abs(area_sum) / 2.0


def monte_carlo_area(polygon: Polygon, num_points: int = 10000) -> float:
    """
    Обчислює площу полігону методом Монте-Карло.
    
    Алгоритм:
    1. Визначається bounding box полігону
    2. Генеруються випадкові точки всередині bounding box
    3. Підраховується кількість точок, що потрапили в полігон
    4. Площа обчислюється як: S_poly ≈ S_box × (K/M)
    
    Args:
        polygon (Polygon): Полігон Shapely
        num_points (int): Кількість випадкових точок для генерації
    
    Returns:
        float: Приблизна площа полігону
    """
    # Отримуємо bounding box
    minx, miny, maxx, maxy = polygon.bounds
    
    # Площа bounding box
    box_area = (maxx - minx) * (maxy - miny)
    
    # Генеруємо випадкові точки та рахуємо ті, що всередині полігону
    points_inside = 0
    for _ in range(num_points):
        x = random.uniform(minx, maxx)
        y = random.uniform(miny, maxy)
        
        # Використовуємо метод .contains() з Shapely
        from shapely.geometry import Point
        if polygon.contains(Point(x, y)):
            points_inside += 1
    
    # Обчислюємо площу
    estimated_area = box_area * (points_inside / num_points)
    
    return estimated_area


def calculate_relative_error(estimated: float, reference: float) -> float:
    """
    Обчислює відносну похибку у відсотках.
    
    δ = |S_estimated - S_reference| / S_reference × 100%
    
    Args:
        estimated (float): Обчислена площа
        reference (float): Еталонна площа
    
    Returns:
        float: Відносна похибка у відсотках
    """
    return abs(estimated - reference) / reference * 100.0
