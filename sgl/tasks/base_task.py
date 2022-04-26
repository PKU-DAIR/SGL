class BaseTask:
    def __init__(self):
        pass

    def _execute(self):
        return NotImplementedError

    def _evaluate(self):
        return NotImplementedError

    def _train(self):
        return NotImplementedError
