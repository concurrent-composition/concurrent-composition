from dataclasses import dataclass
from typing import Callable, Any

# FRAMEWORK TYPES


@dataclass
class Domain(object):
    descriptor: str
    member: Callable[[Any], bool] = None


@dataclass
class Metric(object):
    descriptor: str
    distance: Callable[[Any, Any], float] = None


@dataclass
class Measure(object):
    descriptor: str

    def concurrent_composition(self, d_mids):
        if self.descriptor == "RenyiDivergence":
            return lambda alpha: sum(d_mid(alpha) for d_mid in d_mids)
        if self.descriptor.startswith("FixedRenyiDivergence"):
            return sum(d_mids)
        raise ValueError("Unknown measure of privacy")
    
    def sequential_composition(self, d_mids):
        if self.descriptor == "RenyiDivergence":
            return lambda alpha: sum(d_mid(alpha) for d_mid in d_mids)
        if self.descriptor.startswith("FixedRenyiDivergence"):
            return sum(d_mids)
        raise ValueError("Unknown measure of privacy")


WRAPPER: Callable[["Queryable"], "Queryable"] = None


def with_wrapper(new_wrapper: Callable[["Queryable"], "Queryable"], f: Callable[[], Any]):
    global WRAPPER
    prev_wrapper = WRAPPER

    if prev_wrapper is None:
        WRAPPER = new_wrapper
    else:
        def WRAPPER(q): return prev_wrapper(new_wrapper(q))

    try:
        answer = f()
    finally:
        WRAPPER = prev_wrapper

    return answer


class Queryable(object):
    # for wrappers to work, we sometimes need to return a different, wrapped `self`
    # to do this, we intercept new when WRAPPER is not None.
    # This fiddling with constructors is an unfortunate artifact of Python that isn't needed in Rust.
    #
    # when you try to construct a queryable, and WRAPPER is set,
    # then the wrapped queryable is returned instead
    def __new__(cls, transition: Callable[["Queryable", Any], Any]):

        global WRAPPER
        if WRAPPER is not None:
            wrapper = WRAPPER
            WRAPPER = None
            queryable = wrapper(Queryable(transition))
            WRAPPER = wrapper
            return queryable

        return object.__new__(cls)

    # class initialization is only performed if the queryable
    # was not already initialized via a wrapper in __new__
    def __init__(self, transition: Callable[["Queryable", Any], Any]):
        global WRAPPER
        if WRAPPER is None:
            self.transition = transition

    def eval(self, input):
        """invoke the transition function with the given input
        
        `self` is passed in so that queryables can create children that know how to communicate with their parents"""
        return self.transition(self, input)


@dataclass
class Odometer(object):
    input_domain: Domain
    function: Callable[[Any], Any]
    input_metric: Metric
    output_measure: Measure


@dataclass
class Measurement(object):
    input_domain: Domain
    function: Callable[[Any], Any]
    input_metric: Metric
    output_measure: Measure
    privacy_map: Callable[[Any], Any]


# MESSAGE TYPES
# This is an internal message that is used to communicate pending privacy changes
@dataclass
class ChildChange(object):
    pending_map: Callable[[Any], Any]
    id: int


@dataclass
class Map(object):
    d_in: float


@dataclass
class MapAfter(object):
    proposed_query: Any


@dataclass
class GetId(object):
    pass


@dataclass
class AskPermission(object):
    id: int


# CONSTRUCTORS


def make_base_gaussian(scale, output_measure):
    """Make a measurement representing the gaussian mechanism"""
    import numpy as np

    def function(data):
        # not floating-point safe, but this is not the purpose of this paper
        return np.random.normal(loc=data, scale=scale)

    if output_measure.descriptor == "RenyiDivergence":
        def privacy_map(d_in):
            return lambda alpha: d_in * alpha / (2 * scale**2)
    else:
        raise ValueError("unsupported output_measure")

    return Measurement(
        input_domain=Domain(
            descriptor="AllNonNullFloats",
            member=lambda v: isinstance(v, float) and not np.isnan(v)),
        function=function,
        input_metric=Metric(
            descriptor="AbsoluteDistance",
            distance=lambda a, b: abs(a - b)),
        output_measure=output_measure,
        privacy_map=privacy_map
    )


def make_fix_alpha(measurement, alpha):
    """Make a measurement that fixes `alpha` in the privacy map of a Renyi-DP measurement"""
    assert measurement.output_measure.descriptor == "RenyiDivergence"
    assert alpha >= 1.
    return Measurement(
        input_domain=measurement.input_domain,
        function=measurement.function,
        input_metric=measurement.input_metric,
        output_measure=Measure(f"FixedRenyiDivergence(alpha={alpha})"),
        privacy_map=lambda d_in: measurement.privacy_map(d_in)(alpha)
    )


def make_sequential_odometer(input_domain, input_metric, output_measure):
    def function(data):
        # queryable state
        child_maps = []

        def sc_transition(this_queryable, query):
            nonlocal data, child_maps, input_domain, input_metric, output_measure

            if isinstance(query, (Measurement, Odometer)):
                assert input_domain == query.input_domain
                assert input_metric == query.input_metric
                assert output_measure == query.output_measure

                child_id = len(child_maps)
                
                getid_wrapper = _new_getid_wrapper(child_id)
                sequentiality_wrapper = _new_sequentiality_wrapper(this_queryable)
                # compose the wrappers
                wrapper = lambda queryable: sequentiality_wrapper(getid_wrapper(queryable))

                # invoke the query, and wrap any resulting queryables
                answer = with_wrapper(wrapper, lambda: query.function(data))

                if isinstance(query, Measurement):
                    child_privacy_map = query.privacy_map
                if isinstance(query, Odometer):
                    child_privacy_map = lambda d_in: answer.eval(Map(d_in))
                
                child_maps.append(child_privacy_map)
                return answer

            if isinstance(query, Map):
                d_mids = [child_map(query.d_in) for child_map in child_maps]
                return output_measure.sequential_composition(d_mids)

            # This is needed to answer queries about the hypothetical privacy consumption after running a query.
            # This gives the filter a way to reject queries that would cause the privacy consumption to exceed the budget.
            if isinstance(query, MapAfter):
                pending_child_maps = [*child_maps]
                if isinstance(query.proposed_query, Measurement):
                    pending_child_maps.push(query.proposed_query.privacy_map)
                elif not isinstance(query.proposed_query, Odometer):
                    raise ValueError("Unknown query")

                def pending_map(d_in):
                    return output_measure.sequential_composition(
                        [child_map(d_in) for child_map in pending_child_maps]
                    )
                return pending_map

            if isinstance(query, AskPermission):
                if query.id != len(child_maps) - 1:
                    raise ValueError("Permission denied")
                return
            
            if isinstance(query, ChildChange):
                if query.id != len(child_maps) - 1:
                    raise ValueError("Permission denied")
                pending_child_maps = [*child_maps]

                pending_child_maps[query.id] = query.pending_map
                def pending_map(d_in):
                    d_mids = [child_map(d_in) for child_map in pending_child_maps]
                    return output_measure.sequential_composition(d_mids)
                return pending_map
            
            raise ValueError("Unknown query", query)
        return Queryable(sc_transition)
    
    return Odometer(
        input_domain=input_domain,
        function=function,
        input_metric=input_metric,
        output_measure=output_measure
    )


def make_concurrent_odometer(input_domain, input_metric, output_measure):
    """Make an odometer that spawns a concurrent odometer queryable when invoked with some data."""
    def function(data):
        # queryable state
        child_maps = []

        def cc_transition(_queryable, query):
            nonlocal data, child_maps, input_domain, input_metric, output_measure

            if isinstance(query, Measurement):
                assert input_domain == query.input_domain
                assert input_metric == query.input_metric
                assert output_measure == query.output_measure
                child_maps.append(query.privacy_map)
                child_id = len(child_maps)

                getid_wrapper = _new_getid_wrapper(child_id)
                answer = with_wrapper(
                    getid_wrapper, lambda: query.function(data))
                return answer

            if isinstance(query, Map):
                d_mids = [child_map(query.d_in) for child_map in child_maps]
                return output_measure.concurrent_composition(d_mids)

            # This is needed to answer queries about the hypothetical privacy consumption after running a query.
            # This gives the filter a way to reject the query before the state is changed.
            if isinstance(query, MapAfter):
                if isinstance(query.proposed_query, Measurement):
                    def pending_map(d_in):
                        pending_maps = [*child_maps,
                                        query.proposed_query.privacy_map]
                        d_mids = [child_map(d_in)
                                  for child_map in pending_maps]
                        return output_measure.concurrent_composition(d_mids)
                    return pending_map
            raise ValueError(f"unrecognized query: {query}")

        return Queryable(cc_transition)

    return Odometer(
        input_domain=input_domain,
        input_metric=input_metric,
        output_measure=output_measure,
        function=function
    )


def make_odometer_to_filter(odometer, d_in, d_out):
    """Construct a filter measurement out of an odometer.
    Limits the privacy consumption of the queryable that the odometer spawns."""
    def function(data):
        nonlocal odometer, d_in, d_out

        # construct a filter queryable
        def filter_transition(_queryable, query):
            nonlocal d_in, d_out

            if isinstance(query, ChildChange):
                if query.pending_map(d_in) > d_out:
                    raise ValueError("privacy budget exceeded")
                return query.pending_map
            raise ValueError("unrecognized query", query)
        filter_queryable = Queryable(filter_transition)

        # the child odometer always identifies itself as id 0
        getid_wrapper = _new_getid_wrapper(0)
        recursive_wrapper = _new_filter_wrapper(filter_queryable)
        def wrapper(q): return recursive_wrapper(getid_wrapper(q))

        return with_wrapper(wrapper, lambda: odometer.function(data))

    def privacy_map(d_in_prime):
        nonlocal d_in, d_out
        if d_in_prime > d_in:
            raise ValueError(
                "d_in may not exceed the d_in passed into the constructor")
        return d_out

    return Measurement(
        input_domain=odometer.input_domain,
        function=function,
        input_metric=odometer.input_metric,
        output_measure=odometer.output_measure,
        privacy_map=privacy_map
    )


# BEGIN WRAPPER HELPERS


def _new_getid_wrapper(id):
    """Adds a wrapper queryable that reports its id when queried"""
    def wrap_logic(inner_queryable) -> Queryable:
        def getid_wrapper_transition(wrapper_queryable, query):
            if isinstance(query, GetId):
                return id
            # the inner queryable may handle any other kind of query
            return inner_queryable.eval(query)
        return Queryable(getid_wrapper_transition)
    return wrap_logic


def _new_sequentiality_wrapper(seq_qbl):
    def wrap_logic(inner_qbl) -> Queryable:
        def sequentiality_wrapper_transition(wrapper_qbl, query):
            # whitelist map queries because they don't change state
            if isinstance(query, Map):
                return inner_qbl.eval(query)
            seq_qbl.eval(AskPermission(inner_qbl.eval(GetId())))
            return inner_qbl.eval(query)
        return Queryable(sequentiality_wrapper_transition)
    return wrap_logic


def _new_filter_wrapper(parent_queryable):
    """Constructs a function that recursively wraps child queryables.
    All queryable descendants recursively report privacy usage to their parents"""
    def wrap_logic(inner_queryable) -> Queryable:
        def filter_wrapper_transition(wrapper_queryable, query):
            if isinstance(query, (Measurement, Odometer)):
                # Determine privacy usage after the proposed query
                pending_privacy_map = inner_queryable.eval(
                    MapAfter(query))
                
                # will throw an exception if the privacy budget is exceeded
                parent_queryable.eval(ChildChange(
                    pending_map=pending_privacy_map,
                    id=inner_queryable.eval(GetId())
                ))

                # recursively wrap any child queryable in the same logic,
                #     but in a way that speaks to this wrapper instead
                return with_wrapper(
                    lambda qbl: _new_filter_wrapper(wrapper_queryable)(qbl),
                    lambda: inner_queryable.eval(query))

            # if a child queryable reports a potential change in its privacy usage,
            #     work out the new privacy consumption at this level,
            #     then report it to the parent queryable
            if isinstance(query, ChildChange):
                pending_map = inner_queryable.eval(query)
                parent_queryable.eval(ChildChange(
                    id=inner_queryable.eval(GetId()),
                    pending_map=pending_map
                ))
                return pending_map

            # the inner queryable may handle any other kind of query
            return inner_queryable.eval(query)

        return Queryable(filter_wrapper_transition)
    return wrap_logic


# BEGIN TESTS


def test_concurrent_odometer():
    """Construct a concurrent odometer and send it a couple queries."""
    gaussian_meas = make_base_gaussian(
        scale=1., output_measure=Measure("RenyiDivergence"))

    odo = make_concurrent_odometer(
        input_domain=gaussian_meas.input_domain,
        input_metric=gaussian_meas.input_metric,
        output_measure=gaussian_meas.output_measure,
    )

    agg = 2.

    odo_qbl = odo.function(agg)
    print("Privacy consumption starts at zero. If sensitivity is 1, and alpha is 3, then the privacy consumption should be 0.")
    print(odo_qbl.eval(Map(d_in=1.))(alpha=3.))

    print("Now we make a release on our 'agg' dataset with the gaussian mechanism.")
    print(odo_qbl.eval(gaussian_meas))

    print("Epsilon is now 1.5")
    print(odo_qbl.eval(Map(d_in=1.))(alpha=3.))

    print("Make a second release on our 'agg' dataset with the gaussian mechanism.")
    print(odo_qbl.eval(gaussian_meas))

    print("Epsilon is now 3.")
    print(odo_qbl.eval(Map(d_in=1.))(alpha=3.))


def test_concurrent_filter():
    """Construct a concurrent filter and send it a couple queries."""
    gaussian_meas = make_base_gaussian(
        scale=1., output_measure=Measure("RenyiDivergence"))
    gaussian_meas = make_fix_alpha(gaussian_meas, alpha=3.)

    odo = make_concurrent_odometer(
        input_domain=gaussian_meas.input_domain,
        input_metric=gaussian_meas.input_metric,
        output_measure=gaussian_meas.output_measure,
    )

    odo = make_odometer_to_filter(odo, d_in=1., d_out=2.)

    agg = 2.

    odo_qbl = odo.function(agg)

    print("Privacy consumption starts at zero. If sensitivity is 1, and alpha is 3, then the privacy consumption should be 0.")
    print(odo_qbl.eval(Map(d_in=1.)))

    print("Now we make a release on our 'agg' dataset with the gaussian mechanism.")
    print(odo_qbl.eval(gaussian_meas))

    print("Epsilon is now 1.5")
    print(odo_qbl.eval(Map(d_in=1.)))

    print("Try to make a second release on our 'agg' dataset with the gaussian mechanism.")
    try:
        odo_qbl.eval(gaussian_meas)
        assert False, "the above statement should fail"
    except ValueError as e:
        print("filter rejected the release because it would exceed the privacy budget:", e)


if __name__ == "__main__":
    test_concurrent_odometer()
    test_concurrent_filter()

    print("All tests passed!")
