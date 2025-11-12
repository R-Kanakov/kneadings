def register(registry, jobType):
    def decoratorRegisterJob(func):
        if jobType in ['worker', 'init', 'post']:
            registry[jobType] = func
        else:
            raise KeyError("type must be either 'worker', 'init' or 'post'")
        return func

    return decoratorRegisterJob