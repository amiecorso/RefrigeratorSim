# A class representing the refrigerator/smart plug object.

class Refrigerator:
    def __init__(self):
        self.on = False
        self.current_temp = 33  # Fahrenheit
        self.current_timestamp = 0  # minutes
        self.warming_rate = 5/60  # degrees/minute
        self.cooling_rate = -10/60  # degrees/minute
        self.wattage = 200

    def current_rate_temp_change(self):
        if self.on:
            return self.cooling_rate
        else:
            return self.warming_rate

    def expected_temp(self, timestamp):
        elapsed = timestamp - self.current_timestamp
        return self.current_temp + (elapsed * self.current_rate_temp_change())

    def turn_on(self):
        self.on = True

    def turn_off(self):
        self.on = False