class filters:

    def __init__(self):
        self.filters = {}

    def set (self, name, function):
        if not name in self.filters:
            self.filters[name] = []
            self.filters[name].append(function)
        else:
            self.filters[name].append(function)

    def get (self, name):
        if name in self.filters:
            return self.filters[name]

    def call (self, name, argv):

        r = argv

        if (not self.get(name) == None):

            for _ in self.get(name):
                try:
                    if callable(_):
                        r = _(r)
                except: pass

            return r

        else:

            return argv
