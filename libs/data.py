class data:

    def __init__(self):
        self._ = {}

    def set(self, key, data_):
        self._[key] = data_
        return data_

    def get(self, key):
        if key in self._:
            return self._[key]
        else:
            return False
