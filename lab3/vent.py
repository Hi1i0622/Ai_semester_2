import psycopg2
import numpy as np
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import pandas as pd


class VentilationControlSystem:
    def __init__(self, dbname, user, password, host='localhost', port=5432):
        self.conn = psycopg2.connect(
            dbname=dbname,
            user=user,
            password=password,
            host=host,
            port=port
        )

    def close(self):
        self.conn.close()

    def get_action(self, temperature, humidity):
        temp_condition = self.fuzzify_temperature(temperature)
        hum_condition = self.fuzzify_humidity(humidity)

        query = """
            SELECT action
            FROM rules
            WHERE temperature_condition = %s AND humidity_condition = %s
            LIMIT 1;
        """

        with self.conn.cursor() as cur:
            cur.execute(query, (temp_condition, hum_condition))
            result = cur.fetchone()

        if result:
            return result[0]
        return "No action"

    def fuzzify_temperature(self, temperature):
        if temperature > 30:
            return "High"
        elif 20 <= temperature <= 30:
            return "Optimal"
        else:
            return "Low"

    def fuzzify_humidity(self, humidity):
        if humidity > 70:
            return "High"
        elif 40 <= humidity <= 70:
            return "Optimal"
        else:
            return "Low"

    # Дефаззификация
    def defuzzify_action(self, action):
        if action == "IncreaseVentilation":
            return 0.8
        elif action == "DecreaseVentilation":
            return 0.3
        elif action == "StableVentilation":
            return 0.5
        else:
            return 0.0

    def simulate(self, initial_temp, initial_humidity, steps=10):
        temp = initial_temp
        humidity = initial_humidity

        history = []  # список для хранения результатов

        for step in range(steps):
            action = self.get_action(temp, humidity)
            power = self.defuzzify_action(action)
            history.append((step, temp, humidity, action, power))

            print(f"Step {step}: Temp={temp:.2f}°C, Humidity={humidity:.2f}%, Action={action}, Power={power}")

            if temp <= 5 or humidity <= 30:
                print("Остановлено: Достигнут минимальный предел")
                break

            if power == 0.8:  # IncreaseVentilation
                temp -= np.random.uniform(1, 3)
                humidity -= np.random.uniform(2, 5)

            elif power == 0.3:  # DecreaseVentilation
                temp += np.random.uniform(1, 3)
                humidity += np.random.uniform(2, 5)

            elif power == 0.5:  # StableVentilation
                temp += np.random.uniform(-0.5, 0.5)
                humidity += np.random.uniform(-1, 1)

            temp = round(temp, 2)
            humidity = round(humidity, 2)

        # === Выводим результаты ===
        df = pd.DataFrame(history, columns=["Step", "Temperature", "Humidity", "Action", "Power"])
        print("\n📋 Итоговая таблица:")
        print(df.to_string(index=False))

        # === Строим графики ===
        plt.figure(figsize=(10, 5))
        plt.plot(df["Step"], df["Temperature"], marker='o', label="Temperature (°C)")
        plt.plot(df["Step"], df["Humidity"], marker='s', label="Humidity (%)")
        plt.title("Изменение температуры и влажности во времени")
        plt.xlabel("Шаг симуляции")
        plt.ylabel("Значение")
        plt.legend()
        plt.grid(True)
        plt.show()


# === Запуск ===
if __name__ == "__main__":
    dbname = "ventilation"
    user = "postgres"
    password = "1234"

    system = VentilationControlSystem(dbname, user, password)

    try:
        system.simulate(initial_temp=10, initial_humidity=45, steps=15)
    finally:
        system.close()