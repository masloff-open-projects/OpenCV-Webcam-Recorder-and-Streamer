import requests

class requests_:
    
    def __init__(self):
        self.requests = {}
        self.r = requests

    def set (self, name, function):
        if not name in self.requests:
            self.requests[name] = []
            self.requests[name].append(function)
        else:
            self.requests[name].append(function)

    def get (self, name):
        if name in self.requests:
            return self.requests[name]

    def call (self, name, argv):

        if (not self.get(name) == None):

            for _ in self.get(name):
                try:

                    if (_['method'] == 'GET'):
                        self.r.get(_['url'], auth=_['auth'], params=_['params'])

                    elif (_['method'] == 'POST'):
                        self.r.post(_['url'], auth=_['auth'], params=_['params'])

                    elif (_['method'] == 'PUT'):
                        self.r.post(_['url'], auth=_['auth'], params=_['params'])

                except: pass
