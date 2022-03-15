class ParameterSchedule:
    def __init__(self, starting_parameter, step_change, end_parameter=0, increase: bool = False):
        self._parameter = starting_parameter
        self._step_change = step_change if increase else -step_change
        self._end_parameter = end_parameter

    def __repr__(self) -> str:
        d = {'type': self.__class__.__name__, 'starting_parameter': self._parameter, 'step_change': self._step_change,
             'end_parameter': self._end_parameter}
        return str(d)

    def get_parameter(self, take_step: bool = True):
        if take_step is False or self._step_change == 0:
            return self._parameter

        self._parameter = self._parameter + self._step_change

        if self._step_change > 0:
            self._parameter = min(self._parameter, self._end_parameter)
        else:
            self._parameter = max(self._parameter, self._end_parameter)

        return self._parameter
