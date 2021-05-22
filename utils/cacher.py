class Cacher:
    def __init__(self, fn, cache=None):
        self._fn = fn
        self._cache = cache if cache else dict()
        self._current_args = tuple()
        self._current_kwargs = dict()

    def __call__(self, *args, **kwargs):
        self._current_args = args
        self._current_kwargs = kwargs
        return self._check_cache(args[0])

    @classmethod
    def hashable(cls, obj):
        return hasattr(obj, "__hash__")

    def _check_cache(self, vector):
        hashable_vector = vector if self.hashable(vector) else tuple(vector)
        cache_res = self._cache.get(hashable_vector)
        if cache_res is None:
            cache_res = self._call_fn()
            self._cache[hashable_vector] = cache_res
        return cache_res

    def _call_fn(self):
        return self._fn(*self._current_args, **self._current_kwargs)
