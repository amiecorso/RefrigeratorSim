class Refrigerator:
    """ A class representing the refrigerator/smart plug object. """

    def __init__(self):
        self.on = False
        self.max_temp = 43
        self.min_temp = 33
        self.current_temp = 33  # Fahrenheit, starts here at beginning of sim
        self.current_timestamp = 0  # minutes
        self.warming_rate = 5/60  # degrees/minute
        self.cooling_rate = -10/60  # degrees/minute
        self.wattage = 200  # watts

    def _current_rate_temp_change(self):
        """ Returns the current rate of temperature change for the refrigerator, which depends on its on/off status. """
        if self.on:
            return self.cooling_rate
        else:
            return self.warming_rate

    def expected_temp(self, timestamp):
        """ Returns the expected temperature of the refrigerator at the provided timestep, based on current
        temperature and assuming no change to current on/off status.

        :param timestamp: The timestep at which to determine the refrigerator's temperature
        :return: The expected temperature of the fridge at the given timestep
        """
        elapsed = timestamp - self.current_timestamp
        return self.current_temp + (elapsed * self._current_rate_temp_change())

    def turn_on(self):
        self.on = True

    def turn_off(self):
        self.on = False