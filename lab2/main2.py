import matplotlib
matplotlib.use('TkAgg')  # или 'Qt5Agg', если TkAgg не работает

import numpy as np
import matplotlib.pyplot as plt

def triangular_membership(x, a, b, c):
    if a <= x <= b:
        return (x - a) / (b - a)
    elif b <= x <= c:
        return (c - x) / (c - b)
    else:
        return 0.0

def fuzzy_intersection(set1_values, set2_values):
    return [min(mu1, mu2) for mu1, mu2 in zip(set1_values, set2_values)]

def generate_membership_values(x_values, a, b, c):
    return [triangular_membership(x, a, b, c) for x in x_values]

x_values = np.linspace(0, 150, 150)

a_low, b_low, c_low = 0, 25, 50
a_mid, b_mid, c_mid = 25, 50, 75
a_high, b_high, c_high = 50, 75, 100
a_critical, b_critical, c_critical = 75, 100, 125

low_vals = generate_membership_values(x_values, a_low, b_low, c_low)
mid_vals = generate_membership_values(x_values, a_mid, b_mid, c_mid)
high_vals = generate_membership_values(x_values, a_high, b_high, c_high)
critical_vals = generate_membership_values(x_values, a_critical, b_critical, c_critical)

# Пересечение вместо объединения
low_intersection = fuzzy_intersection(low_vals, mid_vals)
mid_intersection = fuzzy_intersection(mid_vals, high_vals)
high_intersection = fuzzy_intersection(high_vals, critical_vals)

def match_consumption(consumption: int):
    high = low_intersection[consumption]
    mid = mid_intersection[consumption]
    low = high_intersection[consumption]
    if max(high, mid, low) == high:
        print(f"Потребление {consumption} - Высокая энергоэффективность")
    elif max(high, mid, low) == mid:
        print(f"Потребление {consumption} - Средняя энергоэффективность")
    elif max(high, mid, low) == low:
        print(f"Потребление {consumption} - Низкая энергоэффективность")
    else:
        print("Значение вне диапазона")

match_consumption(100)

plt.plot(x_values, low_vals, label='Низкое потребление')
plt.plot(x_values, mid_vals, label='Среднее потребление')
plt.plot(x_values, high_vals, label='Высокое потребление')
plt.plot(x_values, critical_vals, label='Критическое потребление')
plt.plot(x_values, low_intersection, label='Высокая энергоэффективность')
plt.plot(x_values, mid_intersection, label='Средняя энергоэффективность')
plt.plot(x_values, high_intersection, label='Низкая энергоэффективность')

plt.xlabel('Уровень потребления')
plt.ylabel('Степень принадлежности')
plt.legend()
plt.title('Пересечение нечетких множеств уровней потребления')
plt.grid(True)
plt.savefig('fuzzy_intersection.png', dpi=300, bbox_inches='tight')
plt.show()